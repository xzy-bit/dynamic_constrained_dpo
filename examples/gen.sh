python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file responses/llama3_hypo_dpo.json \
  --model_path results/model/llama3-8b_hypo_dpo \
  --generator_name llama_hypo_dpo
