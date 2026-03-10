TRAINER="sp_dpo"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer $TRAINER \
  --config recipes/llama31-8b/dpo/config.yaml \
  2>&1 | tee train.log

source /home/llm/miniforge3/etc/profile.d/conda.sh
conda activate hypo-test

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo.json" \
  --model_path results/model/sp_dpo \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo.json
#python eval/diversity.py --input_json responses/sp_dpo_margin.json


<<gen
python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo_beta_01_t_06.json" \
  --model_path results/model/sp_dpo_beta01 \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo_beta_01_t_06.json
#python eval/diversity.py --input_json responses/sp_dpo_beta_01_t_06.json

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo_beta_03_t_06.json" \
  --model_path results/model/sp_dpo_beta03 \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo_beta_03_t_06.json
#python eval/diversity.py --input_json responses/sp_dpo_beta_03_t_06.json

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/dpo_t_06.json" \
  --model_path results/model/dpo \
  --generator_name $TRAINER

python eval/score.py --input_json responses/dpo_t_06.json
#python eval/diversity.py --input_json responses/dpo_t_06.json

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo_beta_05_t_06.json" \
  --model_path results/model/sp_dpo_beta05 \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo_beta_05_t_06.json
#python eval/diversity.py --input_json responses/sp_dpo_beta_05_t_06.json

python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo_alpha_2_beta_025.json" \
  --model_path results/model/sp_dpo_alpha_2_beta_025 \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo_alpha_2_beta_025.json
#python eval/diversity.py --input_json responses/sp_dpo_alpha_2_beta_025.json


python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file "responses/sp_dpo_without_ns.json" \
  --model_path results/model/sp_dpo_without_ns/sp_dpo \
  --generator_name $TRAINER

python eval/score.py --input_json responses/sp_dpo_without_ns.json
#python eval/diversity.py --input_json responses/sp_dpo_without_ns.json
gen

<<finish
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
dpo

TRAINER="hypo_dpo"

<<train
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer $TRAINER \
  --config recipes/llama31-8b/dpo/config.yaml
train
python scripts/gen.py \
  --config_yaml eval/alpacaeval/configs/llama3-instruct.yaml \
  --output_file responses/$TRAINER.json \
  --model_path results/model/$TRAINER \
  --generator_name $TRAINER

python eval/score.py --input_json responses/$TRAINER.json
finish
