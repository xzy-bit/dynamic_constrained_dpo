#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source /home/ziyang/anaconda3/bin/activate

mkdir -p logs responses results/model
mkdir -p .cache/huggingface .cache/huggingface/datasets .cache/huggingface/hub

export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$(pwd)/.cache/huggingface/hub"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

for aggregation in sum mean; do
  if [[ "${aggregation}" == "mean" ]]; then
    tag="dynamic_lambda_dpo_llama32_1b_avg"
  else
    tag="dynamic_lambda_dpo_llama32_1b_sum"
  fi

<<train
  conda activate hypo-train

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero1.yaml \
    scripts/run_dpo.py \
    --trainer dynamic_lambda_dpo \
    --config recipes/llama32-1b/dynamic_lambda_dpo/config.yaml \
    --output_dir "results/model/${tag}" \
    --dlambda_logp_aggregation "${aggregation}" \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    2>&1 | tee "logs/train_${tag}.log"
train
  conda activate hypo-test

  python scripts/gen.py \
    --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
    --output_file "responses/${tag}.json" \
    --model_path "results/model/${tag}" \
    --generator_name "${tag}" \
    --tensor_parallel_size "${VLLM_TP_SIZE}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    2>&1 | tee "logs/gen_${tag}.log"

  python eval/score.py \
    --input_json "responses/${tag}.json" \
    --output_json "responses/${tag}-reward.json" \
    2>&1 | tee "logs/score_${tag}.log"

  if [[ -f "responses/calculate_winrate.py" && -f "responses/gpt4-reward.json" ]]; then
    python responses/calculate_winrate.py \
      --data_path "responses/${tag}-reward.json" \
      --baseline_path "responses/gpt4-reward.json" \
      --save_path "responses/${tag}-winrate.json"
  fi
  conda activate hypo-train
done
