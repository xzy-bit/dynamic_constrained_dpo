#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source /home/llm/miniforge3/etc/profile.d/conda.sh

mkdir -p responses results/model

conda activate hypo-train

<<ref_logp
python scripts/precompute_ref_logps.py \
  --config recipes/llama31-8b/dynamic_lambda_dpo/config.yaml \
  --per_device_train_batch_size 4 \
  --precompute_ref_batch_size 32 \
  --report_to none \
  --save_to_disk_dir results/ref_logps_llama31_8b_dataset \
  --summary_json_path results/ref_logps_llama31_8b_summary_full.json \
  2>&1 | tee train_ref_logps_llama31_8b.log
ref_logp

for aggregation in sum mean; do
  if [[ "${aggregation}" == "mean" ]]; then
    tag="dynamic_lambda_dpo_llama31_8b_avg"
  else
    tag="dynamic_lambda_dpo_llama31_8b_sum"
  fi

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero1.yaml \
    scripts/run_dpo.py \
    --trainer dynamic_lambda_dpo \
    --config recipes/llama31-8b/dynamic_lambda_dpo/config.yaml \
    --output_dir "results/model/${tag}" \
    --offline_ref_logps_dir results/ref_logps_llama31_8b_dataset \
    --dlambda_logp_aggregation "${aggregation}" \
    --report_to wandb \
    --save_strategy epoch \
    2>&1 | tee "train_${tag}.log"

  conda activate hypo-test
  python scripts/gen.py \
    --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
    --output_file "responses/${tag}.json" \
    --model_path "results/model/${tag}" \
    --generator_name "${tag}"

  python eval/score.py \
    --input_json "responses/${tag}.json" \
    --output_json "responses/${tag}-reward.json"

  if [[ -f "responses/calculate_winrate.py" && -f "responses/gpt4-reward.json" ]]; then
    python responses/calculate_winrate.py \
      --data_path "responses/${tag}-reward.json" \
      --baseline_path "responses/gpt4-reward.json" \
      --save_path "responses/${tag}-winrate.json"
  fi

  conda activate hypo-train
done
