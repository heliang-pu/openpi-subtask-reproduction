"""Validate pi0.5 subtask generation on real LeRobot samples."""

import argparse
import copy
import dataclasses
import pathlib
import random
import re

import jax
import jax.numpy as jnp
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


def _load_subtask_vocab(data_config) -> tuple[str, ...]:
    if data_config.local_root is None:
        return ()
    subtasks_path = pathlib.Path(data_config.local_root) / "meta" / "subtasks.parquet"
    if not subtasks_path.exists():
        return ()
    try:
        import pandas as pd
    except ImportError:
        return ()

    table = pd.read_parquet(subtasks_path)
    labels = [str(value) for value in table["subtask"]] if "subtask" in table else [str(value) for value in table.index]
    return tuple(sorted(set(labels), key=len, reverse=True))


def _clean_generated_subtask(text: str, subtask_vocab: tuple[str, ...]) -> str:
    text = re.sub(r"\s+", " ", _as_text(text)).strip()
    text = re.sub(r"^Subtask:\s*", "", text, flags=re.IGNORECASE).strip()
    for label in subtask_vocab:
        if text.lower().startswith(label.lower()):
            return label
    return text.strip(" .")


def _batch_observation(data: dict) -> _model.Observation:
    obs_dict = {
        "image": data["image"],
        "image_mask": data["image_mask"],
        "state": data["state"],
        "tokenized_prompt": data["tokenized_prompt"],
        "tokenized_prompt_mask": data["tokenized_prompt_mask"],
    }

    def add_batch(x):
        x = np.asarray(x)
        return x[None, ...]

    return _model.Observation.from_dict(jax.tree.map(add_batch, obs_dict))


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


def _resolve_checkpoint_dir(checkpoint_dir: str | None, config_name: str) -> pathlib.Path:
    root = pathlib.Path(checkpoint_dir) if checkpoint_dir is not None else pathlib.Path("checkpoints") / config_name
    if (root / "params").exists():
        return root

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")

    direct_steps = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit() and (path / "params").exists()]
    if direct_steps:
        return max(direct_steps, key=lambda path: int(path.name))

    nested_steps = [
        path
        for path in root.glob("*/*")
        if path.is_dir() and path.name.isdigit() and (path / "params").exists()
    ]
    if nested_steps:
        return max(nested_steps, key=lambda path: (path.stat().st_mtime, int(path.name)))

    raise FileNotFoundError(f"No complete checkpoint step with a params/ directory found under: {root}")


def _with_checkpoint_norm_stats(data_config, checkpoint_dir: pathlib.Path):
    if data_config.asset_id is None:
        return data_config

    norm_stats_path = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"
    if not norm_stats_path.exists():
        return data_config

    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return dataclasses.replace(data_config, norm_stats=norm_stats)


def _select_indices(dataset, indices: str | None, num_samples: int, seed: int) -> list[int]:
    if indices:
        selected = [int(x) for x in indices.split(",") if x.strip()]
    else:
        rng = random.Random(seed)
        selected = rng.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    invalid = [index for index in selected if index < 0 or index >= len(dataset)]
    if invalid:
        raise IndexError(f"indices out of range for dataset of length {len(dataset)}: {invalid}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument(
        "--checkpoint-dir",
        help="A step directory, an experiment directory, or a config checkpoint root. Defaults to the latest step.",
    )
    parser.add_argument("--indices", help="Comma-separated dataset indices. Overrides --num-samples when set.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--show-token-ids", action="store_true")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir, args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_config = _with_checkpoint_norm_stats(data_config, checkpoint_dir)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    subtask_vocab = _load_subtask_vocab(data_config)

    print(f"Loading checkpoint: {checkpoint_dir}")
    print(f"Dataset length:     {len(dataset)}")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)

    indices = _select_indices(dataset, args.indices, args.num_samples, args.seed)
    print(f"Sample indices:     {','.join(str(index) for index in indices)}")
    correct = 0
    for index in indices:
        raw_sample = dataset[index]
        prompt = _as_text(raw_sample.get("prompt"))
        gt_subtask = _as_text(raw_sample.get("subtask"))
        data = _prepare_inference_sample(raw_sample, data_config, train_config.model)
        observation = _batch_observation(data)

        tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
        token_ids = np.asarray(tokens[0])
        generated = _clean_generated_subtask(tokenizer.detokenize(token_ids), subtask_vocab)
        is_correct = generated == gt_subtask
        correct += int(is_correct)
        hit_max_tokens = len(token_ids) >= args.max_tokens and _tokenizer.PALIGEMMA_EOS_TOKEN not in token_ids

        print("\n" + "=" * 80)
        print(f"index:     {index}")
        print(f"prompt:    {prompt}")
        print(f"gt:        {gt_subtask}")
        print(f"generated: {generated}")
        print(f"exact:     {is_correct} tokens={len(token_ids)} hit_max_tokens={hit_max_tokens}")
        if args.show_token_ids:
            print(f"token_ids: {token_ids.tolist()}")
    print("=" * 80)
    print(f"exact_match: {correct}/{len(indices)} = {correct / max(1, len(indices)):.2%}")


if __name__ == "__main__":
    main()
