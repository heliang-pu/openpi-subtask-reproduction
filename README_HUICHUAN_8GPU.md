# 汇川 tissue_sort × pi0.5(JAX) 8 卡服务器训练指南

从 pro6000（单卡 RTX PRO 6000）迁移到 8 卡服务器的完整操作。分支 `huichuan` 已包含：
汇川数据配置（`pi05_huichuan_eef`，14 维 EEF）、tissue_sort_final 的归一化统计（`assets/`）、
多卡训练脚本（`train_huichuan_8gpu.sh`）、conda 环境定义（`environment_huichuan.yml`）。

## 前置要求

- NVIDIA 驱动 ≥ 550（jax 0.5.3 + CUDA12 wheel），`nvidia-smi` 能看到 8 卡
- conda / miniconda
- 磁盘：代码 ~1G + 基座权重 12G + 数据集 0.7G + 每个 checkpoint 31G（存 N 个算 N×31G）

## 1. 拉代码（huichuan 分支）

```bash
git clone -b huichuan git@github.com:heliang-pu/openpi-subtask-reproduction.git openpi-subtask
cd openpi-subtask
```

## 2. 建环境

```bash
conda env create -f environment_huichuan.yml    # 环境名 huichuan-openpi
conda activate huichuan-openpi
# 三个非 PyPI 包单独装（依赖已全在 yml 里，--no-deps 避免版本被搅动）：
pip install -e packages/openpi-client --no-deps
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@33cad37054c2b594ceba57463e8f11ee374fa93c"
pip install -e . --no-deps
# 验证：应打印 8 个 CudaDevice
python -c "import jax; print(jax.devices())"
```

若服务器不能直连 github，先在能上网的机器 `pip download` 或直接把 pro6000 上
`/opt/miniconda3/envs/huichuan-openpi` 用 conda-pack 打包过去。

## 3. 拷数据与基座权重（从 pro6000 rsync）

```bash
# 数据集(0.7G) —— 放哪都行，训练时 DATA_ROOT= 指过去
rsync -a fmc3-1@<pro6000-ip>:~/workspace/dataset/huichuan/tissue_sort_final ~/data/

# pi05 基座权重(12G) —— 默认路径 /workspace/models/openpi-assets/checkpoints/pi05_base，
# 放别处则训练时用 PI05_BASE_PARAMS= 指定
rsync -a fmc3-1@<pro6000-ip>:/workspace/models/openpi-assets/checkpoints/pi05_base \
  ~/models/openpi-assets/checkpoints/

# 分词器缓存(4M, 可选; 没有会自动从 gs:// 下载, 国内网络建议直接拷)
rsync -a fmc3-1@<pro6000-ip>:~/.cache/openpi ~/.cache/
```

## 4. 训练

```bash
conda activate huichuan-openpi
cd openpi-subtask
# 8x80G(A800/H800 等):
DATA_ROOT=~/data/tissue_sort_final \
PI05_BASE_PARAMS=~/models/openpi-assets/checkpoints/pi05_base/params \
BATCH=256 STEPS=6000 SAVE=1000 WANDB=1 ./train_huichuan_8gpu.sh

# 8x24G(4090 等) —— 总 batch 降到 64 并开 FSDP 参数分片:
DATA_ROOT=~/data/tissue_sort_final PI05_BASE_PARAMS=... \
BATCH=64 FSDP=8 STEPS=20000 SAVE=2000 ./train_huichuan_8gpu.sh
```

要点：
- **BATCH 必须能被卡数整除**（openpi 自动做数据并行切分）
- batch 变大 N 倍，一般步数可减为 1/N 左右（参考：单卡 bs32 × 6000 步已收敛，
  8 卡 bs256 对应 ~1000-2000 步就要盯 wandb 看 loss，别死跑满）
- `WANDB=1` 前先 `wandb login`；`GPUS=0,1,2,3` 可只用部分卡
- 换数据集：改 `DATA_REPO=local/<名> DATA_ROOT=<路径>`，归一化统计会自动重算
- 断线安全：`tmux new -s train` 里跑

## 5. 产物与取回

checkpoint 落在 `checkpoints/pi05_huichuan_eef/<EXP>/<step>/`（每档 31G，
含 `params/`+`train_state/`+`assets/`）。**推理只需要 `params/` + `assets/`（≈12G）**：

```bash
# 从 8 卡机取回到 pro6000（只拷推理所需）
rsync -a <server>:~/openpi-subtask/checkpoints/pi05_huichuan_eef/<EXP>/<step>/params \
         <server>:~/openpi-subtask/checkpoints/pi05_huichuan_eef/<EXP>/<step>/assets \
  /workspace/shared/openpi-subtask/checkpoints/pi05_huichuan_eef/<EXP>/<step>/
# 然后在 pro6000: cd /workspace/shared/lerobot/docs/huichuan/scripts
# CKPT=/workspace/shared/openpi-subtask/checkpoints/pi05_huichuan_eef/<EXP>/<step> \
#   TASK="..." ./09_infer_jax.sh
```

## 常见问题

| 现象 | 原因/处理 |
|---|---|
| `Batch size ... must be divisible` | BATCH 改成卡数的整数倍 |
| 启动时 OOM | 加 `FSDP=8`；或降 BATCH；或 `XLA_MEM=0.8` |
| 卡在下载 gs://big_vision | 分词器缓存没拷，见第 3 步最后一条 |
| `Norm stats file not found` | 换了 DATA_REPO 但 assets 里没有对应统计——脚本会自动算；若手动跑 train.py 需先跑 `scripts/compute_norm_stats_override.py` |
| PermissionError /storages/liweile | 缓存环境变量没设——用 `train_huichuan_8gpu.sh`（已内置覆盖），别裸跑 train.py |
| loss 出现 NaN | 立即停：降 peak_lr（配置里 1e-5）或查坏数据；**发散后保存的 checkpoint 不可用** |

## 与单卡（pro6000）的对应关系

| | pro6000 | 8 卡服务器 |
|---|---|---|
| 入口 | `docs/huichuan/scripts/07_train_pi05_jax.sh`（lerobot 仓库） | `./train_huichuan_8gpu.sh`（本仓库） |
| 实测速度 | bs32: 3.6s/步 | 视卡型，wandb 看 throughput |
| 已出模型 | `checkpoints/pi05_huichuan_eef/tissue_sort_2/{4000,5000,6000}` | — |
