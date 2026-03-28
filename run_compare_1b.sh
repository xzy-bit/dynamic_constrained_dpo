#!/bin/bash
set -euo pipefail

source /home/llm/miniforge3/etc/profile.d/conda.sh

cd /home/llm/hypo_reproduction
export PYTHONPATH="/home/llm/hypo_reproduction/src"

ACC_CONFIG="recipes/accelerate_configs/zero3.yaml"
GEN_CONFIG="eval/alpacaeval/configs/llama3-instruct.yaml"
BASELINE_JSON="responses/gpt4-reward.json"

run_method() {
  local trainer="$1"
  local config="$2"
  local model_dir="$3"
  local tag="$4"

  conda activate hypo-train
  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACC_CONFIG}" \
    scripts/run_dpo.py \
    --trainer "${trainer}" \
    --config "${config}" \
    2>&1 | tee "train_${tag}.log"

  conda activate hypo-test
  python scripts/gen.py \
    --config_yaml "${GEN_CONFIG}" \
    --output_file "responses/${tag}.json" \
    --model_path "${model_dir}" \
    --generator_name "${tag}"

  python eval/score.py \
    --input_json "responses/${tag}.json" \
    --output_json "responses/${tag}-reward.json"

  python responses/calculate_winrate.py \
    --data_path "responses/${tag}-reward.json" \
    --baseline_path "${BASELINE_JSON}" \
    --save_path "responses/${tag}-winrate.json"
}

run_method "dpo" "recipes/llama32-1b/dpo/config.yaml" "results/model/dpo_llama32_1b/dpo" "dpo_llama32_1b"
run_method "sp_dpo" "recipes/llama32-1b/sp_dpo/config.yaml" "results/model/sp_dpo_llama32_1b/sp_dpo" "sp_dpo_llama32_1b"
run_method "simpo" "recipes/llama32-1b/simpo/config.yaml" "results/model/simpo_llama32_1b/simpo" "simpo_llama32_1b"
run_method "sp_simpo" "recipes/llama32-1b/sp_simpo/config.yaml" "results/model/sp_simpo_llama32_1b/sp_simpo" "sp_simpo_llama32_1b"
