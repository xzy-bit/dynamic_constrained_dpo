#!/bin/bash
source /home/llm/miniforge3/etc/profile.d/conda.sh

for aggregation in sum mean; do
  tag="dynamic_lambda_dpo_llama31_8b_${aggregation}"
  output_dir="results/model/${tag}"
  model_dir="${output_dir}/dynamic_lambda_dpo"

  conda activate hypo-train
  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    scripts/run_dpo.py \
    --trainer dynamic_lambda_dpo \
    --config recipes/llama31-8b/dynamic_lambda_dpo/config.yaml \
    --output_dir "${output_dir}" \
    --dlambda_logp_aggregation "${aggregation}" \
    2>&1 | tee "train_${tag}.log"

  conda activate hypo-test
  python scripts/gen.py \
    --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
    --output_file "responses/${tag}.json" \
    --model_path "${model_dir}" \
    --generator_name "${tag}"

  python eval/score.py \
    --input_json "responses/${tag}.json" \
    --output_json "responses/${tag}-reward.json"
done
