TRAINER="sp_dpo"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer $TRAINER \
  --config recipes/llama31-8b/dpo/config.yaml

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file responses/$TRAINER.json \
  --model_path results/model/$TRAINER \
  --generator_name $TRAINER

python eval/score.py --input_json responses/$TRAINER.json

<<dpo
TRAINER="dpo"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer $TRAINER \
  --config recipes/llama31-8b/dpo/config.yaml

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file responses/$TRAINER.json \
  --model_path results/model/$TRAINER \
  --generator_name $TRAINER

python eval/score.py --input_json responses/$TRAINER.json

TRAINER="hypo_dpo"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer $TRAINER \
  --config recipes/llama31-8b/dpo/config.yaml

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file responses/$TRAINER.json \
  --model_path results/model/$TRAINER \
  --generator_name $TRAINER

python eval/score.py --input_json responses/$TRAINER.json
dpo