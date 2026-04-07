#!/bin/bash

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  scripts/run_dpo.py \
  --trainer async_dynamic_lambda_dpo \
  --config recipes/llama32-1b/dynamic_lambda_dpo/config.yaml \
  --output_dir results/smoke_async_dynamic_lambda \
  --max_steps 10 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --gradient_checkpointing false \
  --dlambda_grad_target last_three_layers \
  --dlambda_logp_aggregation sum

echo "Async smoke test finished."
