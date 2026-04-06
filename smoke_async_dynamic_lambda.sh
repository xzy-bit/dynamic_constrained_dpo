#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source /home/llm/miniforge3/etc/profile.d/conda.sh
conda activate hypo-train

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  scripts/run_dpo.py \
  --trainer async_dynamic_lambda_dpo \
  --config recipes/llama32-1b/dynamic_lambda_dpo/config.yaml \
  --output_dir results/smoke_async_dynamic_lambda \
  --max_steps 1 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --max_length 512 \
  --max_prompt_length 256 \
  --gradient_checkpointing false \
  --dlambda_grad_target last_three_layers \
  --dlambda_logp_aggregation sum

echo "Async smoke test finished."
