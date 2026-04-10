#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"

mkdir -p logs responses results/model

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"

VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
NUM_GPUS="${NUM_GPUS:-4}"
TAG="simpo_llama32_1b"
OUTPUT_DIR="results/model/${TAG}"
MODEL_PATH="${OUTPUT_DIR}/simpo"

conda activate "${TRAIN_ENV:-hypo-train}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  --num_processes "${NUM_GPUS}" \
  scripts/run_dpo.py \
  --trainer simpo \
  --config recipes/llama32-1b/simpo/config.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 1 \
  2>&1 | tee "logs/train_${TAG}.log"

conda activate "${TEST_ENV:-hypo-test}"

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/${TAG}.json" \
  --model_path "${MODEL_PATH}" \
  --generator_name "${TAG}" \
  --tensor_parallel_size "${VLLM_TP_SIZE}" \
  --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  2>&1 | tee "logs/gen_${TAG}.log"

python eval/score.py \
  --input_json "responses/${TAG}.json" \
  --output_json "responses/${TAG}-reward.json" \
  2>&1 | tee "logs/score_${TAG}.log"

if [[ -f "responses/calculate_winrate.py" && -f "responses/gpt4-reward.json" ]]; then
  python responses/calculate_winrate.py \
    --data_path "responses/${TAG}-reward.json" \
    --baseline_path "responses/gpt4-reward.json" \
    --save_path "responses/${TAG}-winrate.json"
fi
