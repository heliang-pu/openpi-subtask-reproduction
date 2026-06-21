"""Plot policy action predictions against dataset actions for one LeRobot episode."""

import argparse
import csv
import dataclasses
import pathlib
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.training import checkpoints as _checkpoints
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


def _clean_generated_subtask(text: str, subtask_vocab: tuple[str, ...]) -> str:
    text = re.sub(r"\s+", " ", _as_text(text)).strip()
    if not text:
        return text

    text = re.sub(r"^Subtask:\s*", "", text, flags=re.IGNORECASE).strip()
    for label in subtask_vocab:
        if text.lower().startswith(label.lower()):
            return label
    return text


def _load_subtask_vocab(data_config) -> tuple[str, ...]:
    if data_config.local_root is None:
        return ()
    subtasks_path = pathlib.Path(data_config.local_root) / "meta" / "subtasks.parquet"
    if not subtasks_path.exists():
        return ()

    import pandas as pd

    table = pd.read_parquet(subtasks_path)
    labels = [str(value) for value in table["subtask"]] if "subtask" in table else [str(value) for value in table.index]
    return tuple(sorted(set(labels), key=len, reverse=True))


def _episode_indices(dataset, episode: int) -> list[int]:
    base_dataset = getattr(dataset, "_dataset", dataset)
    hf_dataset = getattr(base_dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("Expected LeRobotDataset.hf_dataset for fast episode lookup.")

    indices = [index for index, episode_value in enumerate(hf_dataset["episode_index"]) if int(episode_value) == episode]
    if not indices:
        raise ValueError(f"No frames found for episode={episode}.")
    return indices


def _sample_indices(indices: list[int], stride: int, max_frames: int | None) -> list[int]:
    sampled = indices[::stride]
    if sampled[-1] != indices[-1]:
        sampled.append(indices[-1])
    if max_frames is not None and len(sampled) > max_frames:
        positions = np.linspace(0, len(sampled) - 1, max_frames).round().astype(int)
        sampled = [sampled[pos] for pos in positions]
    return sampled


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_inference_sample(raw_sample: dict, data_config, model_config) -> dict:
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
    data = dict(raw_sample)
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
        "token_highlevel_mask": data["token_highlevel_mask"],
    }
    return _model.Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None, ...], obs_dict))


def _load_checkpoint_norm_stats(data_config, checkpoint_dir: pathlib.Path):
    if data_config.asset_id is None:
        return data_config
    norm_stats_path = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"
    if not norm_stats_path.exists():
        return data_config
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return dataclasses.replace(data_config, norm_stats=norm_stats)


def _unnormalize_actions(data_config, observation: _model.Observation, actions) -> np.ndarray:
    output_transform = _transforms.compose(
        [
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    outputs = {
        "state": np.asarray(observation.state[0]),
        "actions": np.asarray(actions[0]),
    }
    return output_transform(outputs)["actions"]


def _write_csv(path: pathlib.Path, rows: list[dict], action_dim: int) -> None:
    fields = ["index", "frame", "gt_subtask", "generated_subtask"]
    fields += [f"gt_a{dim}" for dim in range(action_dim)]
    fields += [f"pred_a{dim}" for dim in range(action_dim)]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: pathlib.Path, frames: np.ndarray, gt: np.ndarray, pred: np.ndarray, *, title: str) -> None:
    action_dim = gt.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 2.2 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    for dim, axis in enumerate(axes):
        axis.plot(frames, gt[:, dim], label="dataset", color="#15803d", linewidth=1.8)
        axis.plot(frames, pred[:, dim], label="policy", color="#1d4ed8", linewidth=1.5, alpha=0.9)
        axis.set_ylabel(f"a{dim}")
        axis.grid(visible=True, alpha=0.25)
        if dim == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episode", type=int, default=28)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--num-action-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default="outputs/action_curve_episode28.png")
    parser.add_argument("--csv-output", default="outputs/action_curve_episode28.csv")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    data_config = _load_checkpoint_norm_stats(data_config, checkpoint_dir)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    subtask_vocab = _load_subtask_vocab(data_config)

    print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
    rng = jax.random.key(0)

    indices = _sample_indices(_episode_indices(dataset, args.episode), args.stride, args.max_frames)
    rows = []
    gt_actions = []
    pred_actions = []
    frames = []
    for count, index in enumerate(indices, start=1):
        raw_sample = dataset[index]
        print(f"Infer actions {count}/{len(indices)} index={index}", flush=True)
        data = _prepare_inference_sample(raw_sample, data_config, train_config.model)
        observation = _batch_observation(data)
        rng, action_rng = jax.random.split(rng)
        actions, subtask_tokens = model.sample_actions_hierarchical(
            action_rng,
            observation,
            max_tokens=args.max_tokens,
            num_steps=args.num_action_steps,
        )

        gt_action = _to_numpy(raw_sample["action"])[0, :7].astype(np.float32)
        pred_action = _unnormalize_actions(data_config, observation, actions)[0, :7].astype(np.float32)
        frame = int(raw_sample["frame_index"])
        generated = _clean_generated_subtask(tokenizer.detokenize(np.asarray(subtask_tokens[0])), subtask_vocab)
        row = {
            "index": index,
            "frame": frame,
            "gt_subtask": _as_text(raw_sample.get("subtask")),
            "generated_subtask": generated,
        }
        row.update({f"gt_a{dim}": float(gt_action[dim]) for dim in range(gt_action.shape[0])})
        row.update({f"pred_a{dim}": float(pred_action[dim]) for dim in range(pred_action.shape[0])})
        rows.append(row)
        gt_actions.append(gt_action)
        pred_actions.append(pred_action)
        frames.append(frame)

    frames_arr = np.asarray(frames)
    gt_arr = np.stack(gt_actions)
    pred_arr = np.stack(pred_actions)

    output = pathlib.Path(args.output)
    csv_output = pathlib.Path(args.csv_output)
    _plot(
        output,
        frames_arr,
        gt_arr,
        pred_arr,
        title=f"Episode {args.episode}: first-step action prediction vs dataset action",
    )
    _write_csv(csv_output, rows, gt_arr.shape[1])

    mae = np.mean(np.abs(pred_arr - gt_arr), axis=0)
    print(output)
    print(csv_output)
    print("MAE:", ", ".join(f"a{dim}={value:.4f}" for dim, value in enumerate(mae)))
    for row in rows:
        print(
            f"frame={row['frame']} gt_subtask={row['gt_subtask']} "
            f"generated_subtask={row['generated_subtask']}"
        )


if __name__ == "__main__":
    main()
