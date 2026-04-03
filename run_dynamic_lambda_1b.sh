#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ACC_CONFIG="${ACC_CONFIG:-recipes/accelerate_configs/zero3.yaml}"
GEN_CONFIG="${GEN_CONFIG:-eval/alpacaeval/configs/llama3-instruct.yaml}"
BASELINE_JSON="${BASELINE_JSON:-responses/gpt4-reward.json}"
TRAIN_ENV="${TRAIN_ENV:-hypo-train}"
TEST_ENV="${TEST_ENV:-hypo-test}"
GEN_TP_SIZE="${GEN_TP_SIZE:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

TRAINER="${TRAINER:-dynamic_lambda_dpo}"
RECIPE_CONFIG="${RECIPE_CONFIG:-recipes/llama32-1b/dynamic_lambda_dpo/config.yaml}"
MODEL_DIR="${MODEL_DIR:-results/model/dynamic_lambda_dpo_llama32_1b/dynamic_lambda_dpo}"
RUN_TAG="${RUN_TAG:-dynamic_lambda_dpo_llama32_1b}"

mkdir -p responses results/model

if [[ ! -f "${ACC_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACC_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${GEN_CONFIG}" ]]; then
  echo "Missing generation config: ${GEN_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${RECIPE_CONFIG}" ]]; then
  echo "Missing recipe config: ${RECIPE_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "scripts/run_dpo.py" ]]; then
  echo "Missing training entrypoint: scripts/run_dpo.py" >&2
  exit 1
fi

OUTPUT_DIR="${MODEL_DIR%/*}"
RESPONSE_TAG="${RUN_TAG}"
GEN_ARGS=()

if [[ -n "${RUN_SUFFIX}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_${RUN_SUFFIX}"
  MODEL_DIR="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")"
  RESPONSE_TAG="${RUN_TAG}_${RUN_SUFFIX}"
fi

conda activate "${TRAIN_ENV}"

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file "${ACC_CONFIG}" \
  scripts/run_dpo.py \
  --trainer "${TRAINER}" \
  --config "${RECIPE_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "train_${RESPONSE_TAG}.log"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Expected model directory not found after training: ${MODEL_DIR}" >&2
  exit 1
fi

conda activate "${TEST_ENV}"
if [[ -n "${GEN_TP_SIZE}" ]]; then
  GEN_ARGS+=(--tensor_parallel_size "${GEN_TP_SIZE}")
fi

python scripts/gen.py \
  --config_yaml "${GEN_CONFIG}" \
  --output_file "responses/${RESPONSE_TAG}.json" \
  --model_path "${MODEL_DIR}" \
  --generator_name "${RESPONSE_TAG}" \
  "${GEN_ARGS[@]}"

python eval/score.py \
  --input_json "responses/${RESPONSE_TAG}.json" \
  --output_json "responses/${RESPONSE_TAG}-reward.json"

if [[ -f "responses/calculate_winrate.py" && -f "${BASELINE_JSON}" ]]; then
  python responses/calculate_winrate.py \
    --data_path "responses/${RESPONSE_TAG}-reward.json" \
    --baseline_path "${BASELINE_JSON}" \
    --save_path "responses/${RESPONSE_TAG}-winrate.json"
else
  echo "Skipping winrate for ${RESPONSE_TAG}: missing responses/calculate_winrate.py or baseline ${BASELINE_JSON}" >&2
fi
