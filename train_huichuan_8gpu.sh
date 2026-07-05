#!/usr/bin/env bash
# 汇川 tissue_sort 任务 pi0.5(JAX) 多卡训练（默认按 8 卡配置, 单卡也能跑）。
# 用法（在 openpi-subtask 仓库根目录, conda 环境 huichuan-openpi）:
#   DATA_ROOT=~/data/tissue_sort_final ./train_huichuan_8gpu.sh
# 常用覆盖:
#   EXP=tissue_sort_8gpu   实验名(checkpoint 目录名)
#   BATCH=256              总 batch(必须能被卡数整除; 8x80G 推荐 256, 8x24G 用 64 并配 FSDP=8)
#   STEPS=6000             训练步数    SAVE=1000 保存间隔(步)
#   FSDP=1                 参数分片卡数(单卡显存<80G 时设 8, 显存充足保持 1 最快)
#   WANDB=1                开 wandb(先 wandb login)
#   PI05_BASE_PARAMS=...   pi05 基座权重 params 目录(默认 /workspace/models/openpi-assets/checkpoints/pi05_base/params)
#   GPUS=0,1,2,3           只用部分卡(CUDA_VISIBLE_DEVICES)
set -e
cd "$(dirname "$0")"

CONFIG="${CONFIG:-pi05_huichuan_eef}"
EXP="${EXP:-tissue_sort_8gpu}"
DATA_REPO="${DATA_REPO:-local/tissue_sort_final}"
DATA_ROOT="${DATA_ROOT:?用法: DATA_ROOT=<tissue_sort_final数据集路径> $0}"
BATCH="${BATCH:-256}"
STEPS="${STEPS:-6000}"
SAVE="${SAVE:-1000}"
FSDP="${FSDP:-1}"

# 仓库代码 setdefault 了原作者的缓存路径，任何新机器都要覆盖到当前用户可写目录
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

# 基座权重存在性检查（12G, 需事先 rsync 到位或设 PI05_BASE_PARAMS）
BASE="${PI05_BASE_PARAMS:-/workspace/models/openpi-assets/checkpoints/pi05_base/params}"
[ -d "$BASE" ] || { echo "缺基座权重: $BASE  (rsync 过来或用 PI05_BASE_PARAMS= 指定)"; exit 1; }

# 归一化统计: 仓库已带 tissue_sort_final 的(assets/), 换数据集时自动重算
NORM="assets/$CONFIG/$DATA_REPO/norm_stats.json"
if [ ! -f "$NORM" ]; then
  echo "== 归一化统计不存在, 计算中: $NORM =="
  python scripts/compute_norm_stats_override.py \
    --config-name "$CONFIG" --repo-id "$DATA_REPO" --local-root "$DATA_ROOT"
fi

EXTRA_ARGS=""
[ "${WANDB:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --wandb-enabled"

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM:-0.9}" exec python scripts/train.py "$CONFIG" \
  --exp-name "$EXP" --overwrite \
  --data.repo-id "$DATA_REPO" --data.local-root "$DATA_ROOT" \
  --batch-size "$BATCH" --num-train-steps "$STEPS" \
  --save-interval "$SAVE" --keep-period "$SAVE" --max-checkpoints 10 \
  --lr-schedule.decay-steps "$STEPS" \
  --fsdp-devices "$FSDP" \
  $EXTRA_ARGS
