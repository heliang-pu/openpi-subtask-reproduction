"""Render a long image with one subtask prediction per second for a full episode."""

import argparse
import copy
import pathlib
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_hwc_uint8(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _prepare_inference_sample(raw_sample: dict, data_config, model_config) -> dict:
    data = copy.deepcopy(raw_sample)
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizeSubtaskInference(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    for transform in transforms:
        data = transform(data)
    return data


def _observation_dict(data: dict) -> dict:
    return {
        "image": data["image"],
        "image_mask": data["image_mask"],
        "state": data["state"],
        "tokenized_prompt": data["tokenized_prompt"],
        "tokenized_prompt_mask": data["tokenized_prompt_mask"],
    }


def _batch_observation(samples: list[dict]) -> _model.Observation:
    obs_dicts = [_observation_dict(sample) for sample in samples]
    stacked = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *obs_dicts)
    return _model.Observation.from_dict(stacked)


def _load_font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = pathlib.Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, width_chars: int, line_gap: int = 4) -> int:
    x, y = xy
    lines = []
    for part in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(part, width=width_chars) or [""])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _episode_index_map(dataset) -> dict[int, list[int]]:
    base_dataset = getattr(dataset, "_dataset", dataset)
    hf_dataset = getattr(base_dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("Expected LeRobotDataset.hf_dataset for fast episode lookup.")

    result: dict[int, list[int]] = {}
    for index, episode in enumerate(hf_dataset["episode_index"]):
        result.setdefault(int(episode), []).append(index)
    return result


def _sample_episode_every_second(dataset, *, episode: int | None, seconds_stride: float) -> tuple[int, list[int], float]:
    base_dataset = getattr(dataset, "_dataset", dataset)
    fps = float(getattr(base_dataset, "fps", 30.0))
    episode_to_indices = _episode_index_map(dataset)
    if episode is None:
        episode = max(episode_to_indices, key=lambda ep: len(episode_to_indices[ep]))
    indices = episode_to_indices[episode]
    if not indices:
        raise ValueError(f"episode={episode} is empty")

    stride_frames = max(1, int(round(seconds_stride * fps)))
    selected = indices[::stride_frames]
    if selected[-1] != indices[-1]:
        selected.append(indices[-1])
    return episode, selected, fps


def _pred_color(gt: str, pred: str) -> tuple[int, int, int]:
    if gt == pred:
        return (22, 163, 74)
    if gt and (gt in pred or pred in gt):
        return (202, 138, 4)
    return (220, 38, 38)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="outputs/subtask_episode_timeline.png")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--seconds-stride", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    episode, indices, fps = _sample_episode_every_second(dataset, episode=args.episode, seconds_stride=args.seconds_stride)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)

    raw_samples = [dataset[index] for index in indices]
    prepared = [_prepare_inference_sample(sample, data_config, train_config.model) for sample in raw_samples]

    generated_text: list[str] = []
    for start in range(0, len(prepared), args.batch_size):
        observation = _batch_observation(prepared[start : start + args.batch_size])
        tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
        generated_text.extend(tokenizer.detokenize(token_ids) for token_ids in np.asarray(tokens))

    rows = []
    for index, raw_sample, generated in zip(indices, raw_samples, generated_text, strict=True):
        prompt = _as_text(raw_sample.get("prompt"))
        advantage = "positive" if "Advantage: positive" in prompt else "negative" if "Advantage: negative" in prompt else "n/a"
        rows.append(
            {
                "index": index,
                "episode": int(raw_sample["episode_index"]),
                "frame": int(raw_sample["frame_index"]),
                "time": float(raw_sample["frame_index"]) / fps,
                "advantage": advantage,
                "prompt": prompt,
                "gt": _as_text(raw_sample.get("subtask")),
                "generated": generated,
                "top": _to_hwc_uint8(raw_sample["observation.images.top"]),
                "wrist": _to_hwc_uint8(raw_sample["observation.images.left_wrist"]),
            }
        )

    title_font = _load_font(30, bold=True)
    label_font = _load_font(19, bold=True)
    text_font = _load_font(18)
    small_font = _load_font(15)

    margin = 24
    img_w, img_h = 240, 180
    gap = 16
    text_w = 860
    row_h = 228
    width = margin * 2 + img_w * 2 + gap * 2 + text_w
    height = margin * 2 + 74 + row_h * len(rows)
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), "Full episode subtask timeline", font=title_font, fill=(15, 23, 42))
    subtitle = (
        f"episode {episode} | checkpoint {checkpoint_dir.name} | fps={fps:g} | "
        f"sample every {args.seconds_stride:g}s | rows={len(rows)}"
    )
    draw.text((margin, margin + 40), subtitle, font=small_font, fill=(71, 85, 105))

    y = margin + 74
    for row in rows:
        x = margin
        top_img = Image.fromarray(row["top"]).resize((img_w, img_h), Image.Resampling.LANCZOS)
        wrist_img = Image.fromarray(row["wrist"]).resize((img_w, img_h), Image.Resampling.LANCZOS)
        draw.text((x, y), "top", font=label_font, fill=(30, 41, 59))
        canvas.paste(top_img, (x, y + 28))
        x += img_w + gap
        draw.text((x, y), "wrist", font=label_font, fill=(30, 41, 59))
        canvas.paste(wrist_img, (x, y + 28))

        x += img_w + gap
        adv_color = (22, 163, 74) if row["advantage"] == "positive" else (220, 38, 38)
        meta = (
            f"t={row['time']:.1f}s | frame {row['frame']} | index {row['index']} | "
            f"Advantage: {row['advantage']}"
        )
        draw.text((x, y), meta, font=label_font, fill=adv_color)
        ty = y + 28
        task_only = row["prompt"].split("\nAdvantage:")[0]
        ty = _draw_wrapped(draw, (x, ty), "Task: " + task_only, small_font, (71, 85, 105), 94)
        ty += 5
        ty = _draw_wrapped(draw, (x, ty), "GT: " + row["gt"], text_font, (15, 23, 42), 78)
        ty += 5
        _draw_wrapped(draw, (x, ty), "Pred: " + row["generated"], text_font, _pred_color(row["gt"], row["generated"]), 78)

        draw.line((margin, y + row_h - 14, width - margin, y + row_h - 14), fill=(226, 232, 240), width=2)
        y += row_h

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)
    for row in rows:
        print(
            f"t={row['time']:.1f}s frame={row['frame']} adv={row['advantage']} "
            f"gt={row['gt']} pred={row['generated']}"
        )


if __name__ == "__main__":
    main()
