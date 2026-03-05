ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer sp_dpo \
  --config recipes/llama31-8b/dpo/config.yaml
