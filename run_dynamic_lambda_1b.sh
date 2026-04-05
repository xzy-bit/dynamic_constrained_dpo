#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source /home/llm/miniforge3/etc/profile.d/conda.sh
conda activate hypo-train

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  scripts/run_dpo.py \
  --trainer dynamic_lambda_dpo \
  --config recipes/llama32-1b/dynamic_lambda_dpo/config.yaml \
  --output_dir results/model/dynamic_lambda_dpo_llama32_1b_zero1 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
