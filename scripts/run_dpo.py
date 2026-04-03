# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# LoRA:
python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, ScriptArguments, get_dataset, get_model, get_tokenizer
from dynamic_lambda_dpo_config import DynamicLambdaDPOConfig
from dynamic_lambda_dpo_trainer import DynamicLambdaDPOTrainer
from trl import DPOTrainer, ModelConfig, TrlParser, get_peft_config
from dpo_metrics_trainer import DPOMetricsTrainer
from hypo_dpo_trainer import HypoDPOTrainer
from simpo_config import SimPOConfig
from simpo_trainer import SimPOTrainer

try:
    from sp_dpo_config import SPDPOConfig
    from sp_dpo_trainer import SPDPOTrainer
except ImportError:
    SPDPOConfig = None
    SPDPOTrainer = None

try:
    from sp_simpo_config import SPSimPOConfig
    from sp_simpo_trainer import SPSimPOTrainer
except ImportError:
    SPSimPOConfig = None
    SPSimPOTrainer = None

TRAINER_REGISTRY = {
    "dpo": DPOMetricsTrainer,
    "dynamic_lambda_dpo": DynamicLambdaDPOTrainer,
    "hypo_dpo": HypoDPOTrainer,
    "simpo": SimPOTrainer,
}

if SPDPOTrainer is not None:
    TRAINER_REGISTRY["sp_dpo"] = SPDPOTrainer

if SPSimPOTrainer is not None:
    TRAINER_REGISTRY["sp_simpo"] = SPSimPOTrainer

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(example, tokenizer,change_template = None,):
    """
    Build a chat-templated prompt from `messages`, then keep prompt/chosen/rejected as strings.
    This avoids TRL's internal maybe_apply_chat_template schema checks.
    """
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    # 1) build prompt from messages if available
    if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) > 0:
        msgs = example["messages"]

        if msgs[-1].get("role") == "assistant":
            prompt_msgs = msgs[:-1]
        else:
            prompt_msgs = msgs

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = example.get("prompt", "")

    chosen = example["chosen"]
    rejected = example["rejected"]

    if isinstance(chosen, list):
        chosen = tokenizer.apply_chat_template(chosen[-1:], tokenize=False)
    if isinstance(rejected, list):
        rejected = tokenizer.apply_chat_template(rejected[-1:], tokenize=False)

    if tokenizer.bos_token:
        if isinstance(chosen, str) and chosen.startswith(tokenizer.bos_token):
            chosen = chosen[len(tokenizer.bos_token):]
        if isinstance(rejected, str) and rejected.startswith(tokenizer.bos_token):
            rejected = rejected[len(tokenizer.bos_token):]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main(script_args, training_args, model_args,trainer_name: str = "dpo"):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###################
    # Model & Tokenizer
    ###################
    if trainer_name in {"simpo", "sp_simpo"}:
        training_args.truncation_side = "left"
    model = get_model(model_args, training_args)
    use_reference_model = trainer_name not in {"simpo", "sp_simpo"}
    ref_model = get_model(model_args, training_args) if use_reference_model else None
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    #########
    # Dataset
    #########
    raw_dataset = get_dataset(script_args)["train_prefs"]

    column_names = list(raw_dataset.features)

    dataset = raw_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=8,
        remove_columns=column_names,
        desc="Apply chat template (SimPO-style)")

    ##########
    # Training
    ##########
    #def formatting_func(example):
    #    return tokenizer.apply_chat_template(example["messages"], tokenize=False)
    TrainerCls = TRAINER_REGISTRY[trainer_name]

    if trainer_name == "simpo":
        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )
    elif trainer_name == "sp_simpo":
        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
            sp_alpha=training_args.sp_alpha,
            sp_beta=training_args.sp_beta,
            sp_temperature=training_args.sp_temperature,
            sp_neg_support_coef=training_args.sp_neg_support_coef,
        )
    elif trainer_name == "dynamic_lambda_dpo":
        trainer = TrainerCls(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            dlambda_alpha=training_args.dlambda_alpha,
            dlambda_epsilon=training_args.dlambda_epsilon,
            dlambda_lambda_max=training_args.dlambda_lambda_max,
            dlambda_grad_target=training_args.dlambda_grad_target,
            dlambda_reference_free=training_args.dlambda_reference_free,
            dlambda_logp_aggregation=training_args.dlambda_logp_aggregation,
        )
    elif trainer_name == "sp_dpo":
        trainer = TrainerCls(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            sp_alpha=training_args.sp_alpha,
            sp_beta=training_args.sp_beta,
            sp_temperature=training_args.sp_temperature,
            sp_neg_support_coef=training_args.sp_neg_support_coef,
            reference_free=training_args.reference_free,
        )
    else:
        trainer = TrainerCls(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
        )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    save_dir = os.path.join(training_args.output_dir,trainer_name) 
    trainer.save_model(save_dir)
    
    #if training_args.push_to_hub:
    #    trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    import argparse

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--trainer",
        type=str,
        default="dpo",
        choices=sorted(TRAINER_REGISTRY.keys()),
        help="Which Trainer to use.",
    )
    pre_args, remaining_argv = pre.parse_known_args()

    _old_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        if pre_args.trainer == "simpo":
            config_cls = SimPOConfig
        elif pre_args.trainer == "sp_simpo":
            if SPSimPOConfig is None:
                raise ImportError("sp_simpo trainer/config is not available in this workspace.")
            config_cls = SPSimPOConfig
        elif pre_args.trainer == "sp_dpo":
            if SPDPOConfig is None:
                raise ImportError("sp_dpo trainer/config is not available in this workspace.")
            config_cls = SPDPOConfig
        elif pre_args.trainer == "dynamic_lambda_dpo":
            config_cls = DynamicLambdaDPOConfig
        else:
            config_cls = DPOConfig
        parser = TrlParser((ScriptArguments, config_cls, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()
    finally:
        sys.argv = _old_argv

    main(script_args, training_args, model_args, trainer_name=pre_args.trainer)
