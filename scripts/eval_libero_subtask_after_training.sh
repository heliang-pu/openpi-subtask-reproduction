#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-pi05_subtask_libero_tasks0_4_full_infer}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/pi05_subtask_libero_tasks0_4_full}"
EXP_NAME="${EXP_NAME:-libero_tasks0_4_subtask_full_b16_sparse_until_0622_1100}"
EPISODE="${EPISODE:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_FRAMES="${MAX_FRAMES:-120}"
OUT_DIR="${OUT_DIR:-outputs/libero_tasks0_4_full_eval}"
RUN_DATASET_CHECKS="${RUN_DATASET_CHECKS:-1}"
RUN_SIM="${RUN_SIM:-1}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
TASK_IDS="${TASK_IDS:-4,6,9,2,7}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
RANDOM_INIT="${RANDOM_INIT:-0}"
SERVER_PORT="${SERVER_PORT:-8000}"
PYTHON="${PYTHON:-/opt/miniconda3/envs/openpi-jax/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/workspace/dataset/libero_config}"
LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" ]]; then
  if [[ -d "${PWD}/third_party/libero/libero/libero" ]]; then
    LIBERO_ROOT="${PWD}/third_party/libero"
  elif [[ -d "/workspace/shared/openpi_jax/third_party/libero/libero/libero" ]]; then
    LIBERO_ROOT="/workspace/shared/openpi_jax/third_party/libero"
  else
    LIBERO_ROOT="${PWD}/third_party/libero"
  fi
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${LIBERO_CONFIG_PATH}"

CKPT="${CHECKPOINT_DIR:-}"
if [[ -z "${CKPT}" ]]; then
  CKPT="$(find "${CHECKPOINT_ROOT}/${EXP_NAME}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | sort -V | tail -n 1)"
fi

if [[ -z "${CKPT}" || ! -d "${CKPT}/params" ]]; then
  echo "No complete checkpoint found. Looked under: ${CHECKPOINT_ROOT}/${EXP_NAME}" >&2
  exit 1
fi

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "CHECKPOINT_DIR=${CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "RUN_DATASET_CHECKS=${RUN_DATASET_CHECKS}"
echo "RUN_SIM=${RUN_SIM}"
echo "LIBERO_ROOT=${LIBERO_ROOT}"
echo "MAX_STEPS_OVERRIDE=${MAX_STEPS_OVERRIDE:-none}"
echo "RANDOM_INIT=${RANDOM_INIT}"

if [[ "${RUN_DATASET_CHECKS}" == "1" ]]; then
  "${PYTHON}" scripts/validate_subtask_on_dataset.py \
    --config-name "${CONFIG_NAME}" \
    --checkpoint-dir "${CKPT}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed 7 \
    --max-tokens 48 \
    2>&1 | tee "${OUT_DIR}/subtask_accuracy.txt"

  "${PYTHON}" scripts/visualize_subtask_prediction_video.py \
    --config-name "${CONFIG_NAME}" \
    --checkpoint-dir "${CKPT}" \
    --output "${OUT_DIR}/episode${EPISODE}_subtask_predictions.mp4" \
    --episode "${EPISODE}" \
    --full-episode \
    --max-frames "${MAX_FRAMES}" \
    --fps 10 \
    --max-tokens 48 \
    --batch-size 8 \
    --video-writer imageio \
    2>&1 | tee "${OUT_DIR}/episode${EPISODE}_subtask_video.log"

  "${PYTHON}" scripts/plot_action_predictions_on_dataset.py \
    --config-name "${CONFIG_NAME}" \
    --checkpoint-dir "${CKPT}" \
    --episode "${EPISODE}" \
    --stride 20 \
    --max-frames 24 \
    --num-action-steps 5 \
    --max-tokens 48 \
    --output "${OUT_DIR}/episode${EPISODE}_action_curve.png" \
    --csv-output "${OUT_DIR}/episode${EPISODE}_action_curve.csv" \
    2>&1 | tee "${OUT_DIR}/episode${EPISODE}_action_curve.log"
fi

if [[ "${RUN_SIM}" == "1" ]]; then
  SERVER_LOG="${OUT_DIR}/policy_server.log"
  SIM_LOG="${OUT_DIR}/libero_sim.log"
  SIM_VIDEO_DIR="${OUT_DIR}/sim_videos"
  mkdir -p "${SIM_VIDEO_DIR}"
  CLIENT_ARGS=(
    --args.host 127.0.0.1
    --args.port "${SERVER_PORT}"
    --args.task-suite-name "${TASK_SUITE_NAME}"
    --args.task-ids "${TASK_IDS}"
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
    --args.video-out-path "${SIM_VIDEO_DIR}"
    --args.save-all-videos
  )
  if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
    CLIENT_ARGS+=(--args.max-steps-override "${MAX_STEPS_OVERRIDE}")
  fi
  if [[ "${RANDOM_INIT}" == "1" ]]; then
    CLIENT_ARGS+=(--args.random-init)
  fi

  "${PYTHON}" scripts/serve_policy.py \
    --port "${SERVER_PORT}" \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CKPT}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  cleanup() {
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill "${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  echo "Waiting for policy server on port ${SERVER_PORT}..."
  SERVER_READY=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Policy server exited early. See ${SERVER_LOG}" >&2
      exit 1
    fi
    if "${PYTHON}" - "${SERVER_PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
    then
      SERVER_READY=1
      break
    fi
    sleep 5
  done
  if [[ "${SERVER_READY}" != "1" ]]; then
    echo "Timed out waiting for policy server. See ${SERVER_LOG}" >&2
    exit 1
  fi

  PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}" \
  MUJOCO_GL=egl \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
  "${PYTHON}" examples/libero/main.py "${CLIENT_ARGS[@]}" \
    2>&1 | tee "${SIM_LOG}"
fi

cat <<EOF

Evaluation complete.

Outputs:
  ${OUT_DIR}/subtask_accuracy.txt
  ${OUT_DIR}/episode${EPISODE}_subtask_predictions.mp4
  ${OUT_DIR}/episode${EPISODE}_action_curve.png
  ${OUT_DIR}/episode${EPISODE}_action_curve.csv
  ${OUT_DIR}/sim_videos/*.mp4
EOF
