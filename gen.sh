python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file response/llama3_dpo.json \
  --model_path results/model/llama3-8b_dpo \
  --generator_name llama_dpo
