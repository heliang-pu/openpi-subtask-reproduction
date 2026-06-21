"""Create a visual sheet of subtask predictions from one random episode."""

import argparse
import copy
import pathlib
import random
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
        _transforms.TokenizeSubtaskInference(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    for transform in transforms:
        data = transform(data)
    return data


def _batch_observation(data: dict) -> _model.Observation:
    obs_dict = {
        "image": data["image"],
        "image_mask": data["image_mask"],
        "state": data["state"],
        "tokenized_prompt": data["tokenized_prompt"],
        "tokenized_prompt_mask": data["tokenized_prompt_mask"],
    }
    return _model.Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None, ...], obs_dict))


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


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, width_chars: int, line_gap: int = 5) -> int:
    x, y = xy
    lines = []
    for part in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(part, width=width_chars) or [""])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _pick_episode_indices(dataset, seed: int, num_frames: int) -> tuple[int, list[int]]:
    rng = random.Random(seed)
    base_dataset = getattr(dataset, "_dataset", dataset)
    hf_dataset = getattr(base_dataset, "hf_dataset", None)

    episode_to_indices: dict[int, list[int]] = {}
    if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
        for index, episode in enumerate(hf_dataset["episode_index"]):
            episode_to_indices.setdefault(int(episode), []).append(index)
    else:
        for index in range(len(dataset)):
            sample = dataset[index]
            episode = int(sample["episode_index"])
            episode_to_indices.setdefault(episode, []).append(index)

    candidates = [episode for episode, indices in episode_to_indices.items() if indices]
    episode = rng.choice(candidates)
    indices = episode_to_indices[episode]
    if len(indices) <= num_frames:
        return episode, indices
    positions = np.linspace(0, len(indices) - 1, num_frames).round().astype(int)
    return episode, [indices[pos] for pos in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="outputs/subtask_prediction_episode.png")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    episode, indices = _pick_episode_indices(dataset, args.seed, args.num_frames)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)

    rows = []
    for index in indices:
        raw_sample = dataset[index]
        data = _prepare_inference_sample(raw_sample, data_config, train_config.model)
        observation = _batch_observation(data)
        tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
        generated = tokenizer.detokenize(np.asarray(tokens[0]))

        rows.append(
            {
                "index": index,
                "episode": int(raw_sample["episode_index"]),
                "frame": int(raw_sample["frame_index"]),
                "prompt": _as_text(raw_sample.get("prompt")),
                "gt": _as_text(raw_sample.get("subtask")),
                "generated": generated,
                "state": np.asarray(raw_sample["observation.state"]).round(3).tolist(),
                "top": _to_hwc_uint8(raw_sample["observation.images.top"]),
                "wrist": _to_hwc_uint8(raw_sample["observation.images.left_wrist"]),
            }
        )

    title_font = _load_font(30, bold=True)
    label_font = _load_font(21, bold=True)
    text_font = _load_font(20)
    small_font = _load_font(17)

    image_w, image_h = 360, 270
    gap = 18
    text_w = 700
    margin = 28
    row_h = image_h + 72
    width = margin * 2 + image_w * 2 + gap * 2 + text_w
    height = margin * 2 + 66 + row_h * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"Subtask predictions | checkpoint {checkpoint_dir.name} | random episode {episode}"
    draw.text((margin, margin), title, font=title_font, fill=(20, 28, 38))
    draw.text((margin, margin + 40), str(checkpoint_dir), font=small_font, fill=(93, 101, 113))

    y = margin + 76
    for row in rows:
        x = margin
        top_img = Image.fromarray(row["top"]).resize((image_w, image_h), Image.Resampling.LANCZOS)
        wrist_img = Image.fromarray(row["wrist"]).resize((image_w, image_h), Image.Resampling.LANCZOS)
        canvas.paste(top_img, (x, y + 34))
        draw.text((x, y + 6), "top camera", font=label_font, fill=(30, 41, 59))
        x += image_w + gap
        canvas.paste(wrist_img, (x, y + 34))
        draw.text((x, y + 6), "left wrist", font=label_font, fill=(30, 41, 59))

        x += image_w + gap
        meta = f"episode {row['episode']} | dataset index {row['index']} | frame {row['frame']}"
        draw.text((x, y + 6), meta, font=label_font, fill=(15, 23, 42))
        ty = y + 40
        ty = _draw_wrapped(draw, (x, ty), "Prompt: " + row["prompt"], text_font, (51, 65, 85), 68)
        ty += 8
        ty = _draw_wrapped(
            draw,
            (x, ty),
            "Raw state in sample (not tokenized by this checkpoint): " + str(row["state"]),
            small_font,
            (100, 116, 139),
            82,
        )
        ty += 8
        ty = _draw_wrapped(draw, (x, ty), "GT: " + row["gt"], text_font, (22, 101, 52), 68)
        ty += 8
        pred_color = (29, 78, 216) if row["generated"] == row["gt"] else (180, 83, 9)
        _draw_wrapped(draw, (x, ty), "Pred: " + row["generated"], text_font, pred_color, 68)

        draw.line((margin, y + row_h - 16, width - margin, y + row_h - 16), fill=(226, 232, 240), width=2)
        y += row_h

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)
    for row in rows:
        print(f"index={row['index']} frame={row['frame']} gt={row['gt']} pred={row['generated']}")


if __name__ == "__main__":
    main()
