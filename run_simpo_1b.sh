#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

TAG="simpo_llama32_1b"

conda activate hypo-train
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  scripts/run_dpo.py \
  --trainer simpo \
  --config recipes/llama32-1b/simpo/config.yaml \
  2>&1 | tee "logs/train_${TAG}.log"

conda activate hypo-test
python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/${TAG}.json" \
  --model_path "results/model/${TAG}/simpo" \
  --generator_name "${TAG}" \
  2>&1 | tee "logs/gen_${TAG}.log"

python eval/score.py \
  --input_json "responses/${TAG}.json" \
  --output_json "responses/${TAG}-reward.json" \
  2>&1 | tee "logs/score_${TAG}.log"
