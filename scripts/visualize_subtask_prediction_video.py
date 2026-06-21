"""Render a video with subtask predictions over a positive dataset segment."""

import argparse
import copy
import pathlib
import textwrap

import cv2
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


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, width_chars: int, line_gap: int = 5) -> int:
    x, y = xy
    lines = []
    for part in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(part, width=width_chars) or [""])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _find_positive_segment(dataset, *, episode: int | None) -> list[int]:
    base_dataset = getattr(dataset, "_dataset", dataset)
    hf_dataset = getattr(base_dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("Expected LeRobotDataset.hf_dataset for fast positive segment lookup.")

    segments: list[list[int]] = []
    current: list[int] = []
    last_episode = None
    last_index = -2
    for index, (ep, acp) in enumerate(zip(hf_dataset["episode_index"], hf_dataset["complementary_info.acp_indicator_r1"])):
        ep = int(ep)
        acp = int(acp)
        is_positive = acp == 1 and (episode is None or ep == episode)
        if is_positive and ep == last_episode and index == last_index + 1:
            current.append(index)
        else:
            if current:
                segments.append(current)
            current = [index] if is_positive else []
        last_episode = ep
        last_index = index
    if current:
        segments.append(current)

    if not segments:
        raise ValueError(f"No positive segment found for episode={episode}.")
    return max(segments, key=len)


def _find_positive_segment_containing(dataset, target_index: int) -> list[int]:
    base_dataset = getattr(dataset, "_dataset", dataset)
    hf_dataset = getattr(base_dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("Expected LeRobotDataset.hf_dataset for fast positive segment lookup.")

    target_episode = int(hf_dataset[target_index]["episode_index"])
    start = target_index
    while start > 0:
        row = hf_dataset[start - 1]
        if int(row["episode_index"]) != target_episode or int(row["complementary_info.acp_indicator_r1"]) != 1:
            break
        start -= 1

    end = target_index
    while end + 1 < len(hf_dataset):
        row = hf_dataset[end + 1]
        if int(row["episode_index"]) != target_episode or int(row["complementary_info.acp_indicator_r1"]) != 1:
            break
        end += 1

    row = hf_dataset[target_index]
    if int(row["complementary_info.acp_indicator_r1"]) != 1:
        raise ValueError(f"target_index={target_index} is not in a positive segment.")
    return list(range(start, end + 1))


def _sample_indices(indices: list[int], max_frames: int) -> list[int]:
    if len(indices) <= max_frames:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
    return [indices[pos] for pos in positions]


def _render_frame(row: dict, *, width: int, height: int, checkpoint_name: str) -> np.ndarray:
    title_font = _load_font(26, bold=True)
    label_font = _load_font(20, bold=True)
    text_font = _load_font(19)
    small_font = _load_font(16)

    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    margin = 24
    draw.text((margin, 18), f"Successful segment | checkpoint {checkpoint_name}", font=title_font, fill=(15, 23, 42))
    meta = f"episode {row['episode']} | frame {row['frame']} | dataset index {row['index']} | Advantage: positive"
    draw.text((margin, 52), meta, font=small_font, fill=(71, 85, 105))

    image_w, image_h = 512, 384
    gap = 18
    y_img = 82
    top_img = Image.fromarray(row["top"]).resize((image_w, image_h), Image.Resampling.LANCZOS)
    wrist_img = Image.fromarray(row["wrist"]).resize((image_w, image_h), Image.Resampling.LANCZOS)
    canvas.paste(top_img, (margin, y_img + 28))
    draw.text((margin, y_img), "top camera", font=label_font, fill=(30, 41, 59))
    x_wrist = margin + image_w + gap
    canvas.paste(wrist_img, (x_wrist, y_img + 28))
    draw.text((x_wrist, y_img), "left wrist", font=label_font, fill=(30, 41, 59))

    text_x = margin + image_w * 2 + gap * 2
    text_y = y_img
    text_y = _draw_wrapped(draw, (text_x, text_y), "Prompt: " + row["prompt"], text_font, (51, 65, 85), 60)
    text_y += 14
    text_y = _draw_wrapped(draw, (text_x, text_y), "GT: " + row["gt"], text_font, (22, 101, 52), 60)
    text_y += 14
    pred_color = (29, 78, 216) if row["generated"] == row["gt"] else (180, 83, 9)
    text_y = _draw_wrapped(draw, (text_x, text_y), "Pred: " + row["generated"], text_font, pred_color, 60)
    text_y += 14
    _draw_wrapped(
        draw,
        (text_x, text_y),
        "State is included in the subtask prefix when discrete_state_input=True.",
        small_font,
        (100, 116, 139),
        68,
    )

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="outputs/subtask_success_segment.mp4")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--contains-index", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    if args.contains_index is not None:
        segment = _find_positive_segment_containing(dataset, args.contains_index)
    else:
        segment = _find_positive_segment(dataset, episode=args.episode)
    indices = _sample_indices(segment, args.max_frames)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)

    rows = []
    prepared = []
    raw_samples = []
    for index in indices:
        raw_sample = dataset[index]
        raw_samples.append(raw_sample)
        prepared.append(_prepare_inference_sample(raw_sample, data_config, train_config.model))

    generated_text: list[str] = []
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start : start + args.batch_size]
        observation = _batch_observation(batch)
        tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
        for token_ids in np.asarray(tokens):
            generated_text.append(tokenizer.detokenize(token_ids))

    for index, raw_sample, generated in zip(indices, raw_samples, generated_text, strict=True):
        rows.append(
            {
                "index": index,
                "episode": int(raw_sample["episode_index"]),
                "frame": int(raw_sample["frame_index"]),
                "prompt": _as_text(raw_sample.get("prompt")),
                "gt": _as_text(raw_sample.get("subtask")),
                "generated": generated,
                "top": _to_hwc_uint8(raw_sample["observation.images.top"]),
                "wrist": _to_hwc_uint8(raw_sample["observation.images.left_wrist"]),
            }
        )

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1720, 560
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}")
    for row in rows:
        writer.write(_render_frame(row, width=width, height=height, checkpoint_name=checkpoint_dir.name))
    writer.release()

    print(output)
    print(f"positive_segment_len={len(segment)} sampled_frames={len(rows)}")
    for row in rows:
        print(f"ep={row['episode']} frame={row['frame']} index={row['index']} gt={row['gt']} pred={row['generated']}")


if __name__ == "__main__":
    main()
