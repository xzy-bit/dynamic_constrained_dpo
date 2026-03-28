import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, get_dataset, get_model, get_tokenizer
from simpo_config import SimPOConfig
from simpo_trainer import SimPOTrainer
from trl import ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\\n\\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return
    chat_template = tokenizer.chat_template or tokenizer.default_chat_template or ""
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def is_openai_format(messages):
    return isinstance(messages, list) and all(
        isinstance(message, dict) and "role" in message and "content" in message for message in messages
    )


def apply_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    change_template: str = None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    if not all(k in example.keys() for k in ("chosen", "rejected")):
        raise ValueError(f"Expected chosen/rejected keys, found {list(example.keys())}")
    if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
        raise ValueError("SimPO preprocessing requires OpenAI-format messages for chosen/rejected.")

    if "prompt" in example and is_openai_format(example["prompt"]):
        prompt_messages = example["prompt"]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    elif "messages" in example and is_openai_format(example["messages"]):
        prompt_messages = example["messages"]
        if prompt_messages[-1]["role"] == "assistant":
            prompt_messages = prompt_messages[:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]
    else:
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    if tokenizer.bos_token:
        if chosen.startswith(tokenizer.bos_token):
            chosen = chosen[len(tokenizer.bos_token) :]
        if rejected.startswith(tokenizer.bos_token):
            rejected = rejected[len(tokenizer.bos_token) :]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

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

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    training_args.truncation_side = "left"

    model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    raw_dataset = get_dataset(script_args)
    train_split = getattr(script_args, "dataset_train_split", "train_prefs")
    train_dataset = raw_dataset[train_split]
    eval_dataset = None
    if training_args.do_eval:
        eval_split = getattr(script_args, "dataset_test_split", None)
        if eval_split and eval_split in raw_dataset:
            eval_dataset = raw_dataset[eval_split]

    column_names = list(train_dataset.features)
    change_template = "mistral" if "mistral" in model_args.model_name_or_path.lower() else None

    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": True,
            "change_template": change_template,
        },
        num_proc=training_args.dataset_num_proc,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "auto_insert_empty_system_msg": True,
                "change_template": change_template,
            },
            num_proc=training_args.dataset_num_proc,
            remove_columns=list(eval_dataset.features),
            desc="Formatting eval comparisons with prompt template",
        )

    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    trainer.save_model(training_args.output_dir)

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SimPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
