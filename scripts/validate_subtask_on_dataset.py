"""Validate pi0.5 subtask generation on real LeRobot samples."""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

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
    data = raw_sample
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_subtask_pickup_round1_50ep_lora")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--indices", default="0,100,500")
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_dir}")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    tokenizer = _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    for index in indices:
        raw_sample = dataset[index]
        prompt = _as_text(raw_sample.get("prompt"))
        gt_subtask = _as_text(raw_sample.get("subtask"))
        data = _prepare_inference_sample(raw_sample, data_config, train_config.model)
        observation = _batch_observation(data)

        tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
        token_ids = np.asarray(tokens[0])
        generated = tokenizer.detokenize(token_ids)

        print("\n" + "=" * 80)
        print(f"index:     {index}")
        print(f"prompt:    {prompt}")
        print(f"gt:        {gt_subtask}")
        print(f"generated: {generated}")
        print(f"token_ids: {token_ids.tolist()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
