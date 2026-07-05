#!/usr/bin/env python3
"""openpi 归一化统计（支持 --repo-id/--local-root 覆盖数据集——原版 compute_norm_stats.py 不支持）。

用法（huichuan-openpi 环境）:
  python openpi_norm_stats.py --config-name pi05_huichuan_eef \
      --repo-id local/tissue_sort_clean --local-root /path/to/tissue_sort_clean

统计写入 openpi-subtask/assets/<config>/<repo_id>/norm_stats.json，train.py 按同键查找。
"""
import dataclasses
import os

os.environ.setdefault("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
os.environ.setdefault("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(
    config_name: str,
    repo_id: str | None = None,
    local_root: str | None = None,
    max_frames: int | None = None,
):
    config = _config.get_config(config_name)
    if repo_id is not None or local_root is not None:
        overrides = {}
        if repo_id is not None:
            overrides["repo_id"] = repo_id
        if local_root is not None:
            overrides["local_root"] = local_root
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **overrides))
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("data config 缺 repo_id")

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
