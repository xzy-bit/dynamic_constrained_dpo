#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --num_processes 1 \
  --config_file recipes/accelerate_configs/zero1.yaml \
  scripts/run_dpo.py \
  --trainer dynamic_lambda_dpo \
  --config recipes/llama32-1b/dynamic_lambda_dpo/config.yaml \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --output_dir results/smoke_dynamic_lambda \
  --max_steps 10 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
