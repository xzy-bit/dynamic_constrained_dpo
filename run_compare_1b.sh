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

DPO_ENABLE_GRAD_METRICS="${DPO_ENABLE_GRAD_METRICS:-1}"
DPO_GRAD_METRICS_INTERVAL="${DPO_GRAD_METRICS_INTERVAL:-5}"
SPDPO_ENABLE_GRAD_METRICS="${SPDPO_ENABLE_GRAD_METRICS:-1}"
SPDPO_GRAD_METRICS_INTERVAL="${SPDPO_GRAD_METRICS_INTERVAL:-5}"
SIMPO_ENABLE_GRAD_METRICS="${SIMPO_ENABLE_GRAD_METRICS:-1}"
SIMPO_GRAD_METRICS_INTERVAL="${SIMPO_GRAD_METRICS_INTERVAL:-5}"

mkdir -p responses results/model

if [[ ! -f "${ACC_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACC_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${GEN_CONFIG}" ]]; then
  echo "Missing generation config: ${GEN_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "scripts/run_dpo.py" ]]; then
  echo "Missing training entrypoint: scripts/run_dpo.py" >&2
  exit 1
fi

run_method() {
  local trainer="$1"
  local config="$2"
  local model_dir="$3"
  local tag="$4"
  local gen_args=()
  local output_dir="${model_dir%/*}"
  local train_tag="${tag}"
  local response_tag="${tag}"

  if [[ -n "${RUN_SUFFIX}" ]]; then
    output_dir="${output_dir}_${RUN_SUFFIX}"
    model_dir="${output_dir}/$(basename "${model_dir}")"
    response_tag="${tag}_${RUN_SUFFIX}"
  fi

  if [[ ! -f "${config}" ]]; then
    echo "Missing recipe config: ${config}" >&2
    return 1
  fi
  
  
  conda activate "${TRAIN_ENV}"

  ACCELERATE_LOG_LEVEL=info \
  DPO_ENABLE_GRAD_METRICS="${DPO_ENABLE_GRAD_METRICS}" \
  DPO_GRAD_METRICS_INTERVAL="${DPO_GRAD_METRICS_INTERVAL}" \
  SPDPO_ENABLE_GRAD_METRICS="${SPDPO_ENABLE_GRAD_METRICS}" \
  SPDPO_GRAD_METRICS_INTERVAL="${SPDPO_GRAD_METRICS_INTERVAL}" \
  SIMPO_ENABLE_GRAD_METRICS="${SIMPO_ENABLE_GRAD_METRICS}" \
  SIMPO_GRAD_METRICS_INTERVAL="${SIMPO_GRAD_METRICS_INTERVAL}" \
  accelerate launch \
    --config_file "${ACC_CONFIG}" \
    scripts/run_dpo.py \
    --trainer "${trainer}" \
    --config "${config}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "train_${response_tag}.log"

  if [[ ! -d "${model_dir}" ]]; then
    echo "Expected model directory not found after training: ${model_dir}" >&2
    return 1
  fi

  conda activate "${TEST_ENV}"
  if [[ -n "${GEN_TP_SIZE}" ]]; then
    gen_args+=(--tensor_parallel_size "${GEN_TP_SIZE}")
  fi
  python scripts/gen.py \
    --config_yaml "${GEN_CONFIG}" \
    --output_file "responses/${response_tag}.json" \
    --model_path "${model_dir}" \
    --generator_name "${response_tag}" \
    "${gen_args[@]}"

  python eval/score.py \
    --input_json "responses/${response_tag}.json" \
    --output_json "responses/${response_tag}-reward.json"

  if [[ -f "responses/calculate_winrate.py" && -f "${BASELINE_JSON}" ]]; then
    python responses/calculate_winrate.py \
      --data_path "responses/${response_tag}-reward.json" \
      --baseline_path "${BASELINE_JSON}" \
      --save_path "responses/${response_tag}-winrate.json"
  else
    echo "Skipping winrate for ${response_tag}: missing responses/calculate_winrate.py or baseline ${BASELINE_JSON}" >&2
  fi
}

run_method "dpo" "recipes/llama32-1b/dpo/config.yaml" "results/model/dpo_llama32_1b/dpo" "dpo_llama32_1b"
run_method "sp_dpo" "recipes/llama32-1b/sp_dpo/config.yaml" "results/model/sp_dpo_llama32_1b/sp_dpo" "sp_dpo_llama32_1b"
#run_method "simpo" "recipes/llama32-1b/simpo/config.yaml" "results/model/simpo_llama32_1b/simpo" "simpo_llama32_1b"
run_method "sp_simpo" "recipes/llama32-1b/sp_simpo/config.yaml" "results/model/sp_simpo_llama32_1b/sp_simpo" "sp_simpo_llama32_1b"
