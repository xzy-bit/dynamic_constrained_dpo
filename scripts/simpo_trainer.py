import inspect
import os
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available

from simpo_config import SimPOConfig
from seeking_utils import append_seeking_log, combine_grads, get_last_layer_weight, grad_cosine, grad_norm
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


class SimPOTrainer(Trainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs but the model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError("PEFT is not installed but `peft_config` was provided.")
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                support_gc_kwargs = hasattr(args, "gradient_checkpointing_kwargs") and (
                    "gradient_checkpointing_kwargs"
                    in list(inspect.signature(prepare_model_for_kbit_training).parameters)
                )
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                if support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True
        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, `is_encoder_decoder` must be passed.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a SimPO dataset.")

        self.max_length = args.max_length if args.max_length is not None else 512
        self.max_prompt_length = args.max_prompt_length if args.max_prompt_length is not None else 128
        self.max_target_length = args.max_target_length if args.max_target_length is not None else 128
        self.enable_grad_metrics = os.environ.get("SIMPO_ENABLE_GRAD_METRICS", "1") != "0"
        self.grad_metrics_interval = max(int(os.environ.get("SIMPO_GRAD_METRICS_INTERVAL", "5")), 1)
        self._last_grad_metrics_step = -1

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "DPODataCollatorWithPadding requires `remove_unused_columns=False`; it has been set for you.",
                    UserWarning,
                )
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.truncation_mode = args.truncation_mode
        self._tokenizer = tokenizer
        self.beta = args.beta
        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.sft_weight = args.sft_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def build_tokenized_answer(self, prompt: str, answer: str) -> Dict[str, List[int]]:
        tokenizer = self._tokenizer
        full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])
        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        return {
            "prompt_input_ids": full_tokenized["input_ids"][:response_token_ids_start_idx],
            "prompt_attention_mask": full_tokenized["attention_mask"][:response_token_ids_start_idx],
            "input_ids": full_tokenized["input_ids"][response_token_ids_start_idx:],
            "attention_mask": full_tokenized["attention_mask"][response_token_ids_start_idx:],
        }

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
                raise ValueError("prompt/chosen/rejected should all be strings after preprocessing.")

            tokenizer = self._tokenizer
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            prompt_len_input_ids = min(
                len(chosen_tokens["prompt_input_ids"]),
                len(rejected_tokens["prompt_input_ids"]),
            )
            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            bos_token_id = tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if len(chosen_tokens["prompt_input_ids"]) == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if len(rejected_tokens["prompt_input_ids"]) == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            eos_token_id = tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for prefix, toks in {"chosen_": chosen_sequence_tokens, "rejected_": rejected_sequence_tokens, "": prompt_tokens}.items():
                for type_key, tokens in toks.items():
                    if type_key != "token_type_ids":
                        batch[f"{prefix}{type_key}"] = tokens
        else:
            tokenizer = self._tokenizer
            chosen_tokens = tokenizer(chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            rejected_tokens = tokenizer(rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True)
            prompt_tokens = tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)
            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                else:
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                else:
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1).to(device=device)

        return concatenated_batch

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        logits = (policy_chosen_logps - policy_rejected_logps).to(self.accelerator.device) - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        return (
            all_logps[:len_chosen],
            all_logps[len_chosen:],
            all_logits[:len_chosen],
            all_logits[len_chosen:],
            concatenated_batch["concatenated_labels"][:len_chosen],
            concatenated_batch["concatenated_labels"][len_chosen:],
        )

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same batch and sequence dimensions.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        return (per_token_logps * loss_mask).sum(-1)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "seeking_eval/" if train_eval == "eval" else "seeking/"
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        loss = losses.mean()
        raw_margin = policy_chosen_logps - policy_rejected_logps
        objective_margin = self.beta * (raw_margin - self.gamma_beta_ratio)
        sigmoid_margin = torch.sigmoid(objective_margin)
        objective_accuracy = (policy_chosen_logps > policy_rejected_logps).float()
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().cpu(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().cpu(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().cpu(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
            f"{prefix}scores/chosen": policy_chosen_logps.detach().mean().cpu(),
            f"{prefix}scores/rejected": policy_rejected_logps.detach().mean().cpu(),
            f"{prefix}scores/accuracies": objective_accuracy.detach().mean().cpu(),
            f"{prefix}scores/raw_margins": raw_margin.detach().mean().cpu(),
            f"{prefix}scores/objective_margins": objective_margin.detach().mean().cpu(),
            f"{prefix}scores/sigmoid_margins": sigmoid_margin.detach().mean().cpu(),
            f"{prefix}logps/rejected": self.get_batch_logps(
                policy_rejected_logits,
                rejected_labels,
                average_log_prob=False,
                label_pad_token_id=self.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            ).detach().mean().cpu(),
            f"{prefix}logps/chosen": self.get_batch_logps(
                policy_chosen_logits,
                chosen_labels,
                average_log_prob=False,
                label_pad_token_id=self.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            ).detach().mean().cpu(),
            f"{prefix}logits/rejected": policy_rejected_logits.detach().mean().cpu(),
            f"{prefix}logits/chosen": policy_chosen_logits.detach().mean().cpu(),
        }

        should_compute_grad_metrics = self.enable_grad_metrics
        if should_compute_grad_metrics and train_eval == "train":
            current_step = self.state.global_step + 1
            should_compute_grad_metrics = (
                current_step % self.grad_metrics_interval == 0 and current_step != self._last_grad_metrics_step
            )
            if should_compute_grad_metrics:
                self._last_grad_metrics_step = current_step

        if should_compute_grad_metrics:
            was_training = model.training
            model.eval()
            try:
                last_layer_weight = get_last_layer_weight(model)

                # Use fresh forward graphs for the grad diagnostics. Reusing the same
                # graph for multiple autograd.grad calls can break on some distributed
                # runs with long-sequence batches.
                def _fresh_policy_outputs():
                    return self.concatenated_forward(model, batch)

                (
                    fresh_chosen_logps,
                    fresh_rejected_logps,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                chosen_only_losses, _, _ = self.simpo_loss(fresh_chosen_logps, fresh_rejected_logps.detach())
                chosen_param_grad = torch.autograd.grad(
                    chosen_only_losses.mean(), last_layer_weight, retain_graph=False, allow_unused=True
                )[0]

                (
                    fresh_chosen_logps,
                    fresh_rejected_logps,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                rejected_only_losses, _, _ = self.simpo_loss(fresh_chosen_logps.detach(), fresh_rejected_logps)
                rejected_param_grad = torch.autograd.grad(
                    rejected_only_losses.mean(), last_layer_weight, retain_graph=False, allow_unused=True
                )[0]
                total_param_grad = combine_grads(chosen_param_grad, rejected_param_grad)

                (
                    fresh_chosen_logps,
                    fresh_rejected_logps,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                full_losses, _, _ = self.simpo_loss(fresh_chosen_logps, fresh_rejected_logps)
                full_param_grad = torch.autograd.grad(
                    full_losses.mean(), last_layer_weight, retain_graph=False, allow_unused=True
                )[0]
                chosen_total_cos = grad_cosine(chosen_param_grad, full_param_grad)
                rejected_total_cos = grad_cosine(rejected_param_grad, full_param_grad)
                chosen_rejected_cos = grad_cosine(chosen_param_grad, rejected_param_grad)
                combined_total_cos = grad_cosine(total_param_grad, full_param_grad)
                chosen_grad_norm = grad_norm(chosen_param_grad)
                rejected_grad_norm = grad_norm(rejected_param_grad)
                chosen_rejected_grad_norm_ratio = chosen_grad_norm / rejected_grad_norm.clamp_min(1e-12)
                residual_grad = None
                if total_param_grad is not None and full_param_grad is not None:
                    residual_grad = total_param_grad - full_param_grad
                elif total_param_grad is not None:
                    residual_grad = total_param_grad
                elif full_param_grad is not None:
                    residual_grad = -full_param_grad
                decomposition_residual = grad_norm(residual_grad)

                metrics.update(
                    {
                        f"{prefix}grads/chosen_grad_norm": chosen_grad_norm.detach().cpu(),
                        f"{prefix}grads/rejected_grad_norm": rejected_grad_norm.detach().cpu(),
                        f"{prefix}grads/chosen_rejected_grad_norm_ratio": chosen_rejected_grad_norm_ratio.detach().cpu(),
                        f"{prefix}grads/chosen_loss_grad_total_cosine": chosen_total_cos.detach().cpu(),
                        f"{prefix}grads/rejected_loss_grad_total_cosine": rejected_total_cos.detach().cpu(),
                        f"{prefix}grads/chosen_rejected_grad_cosine": chosen_rejected_cos.detach().cpu(),
                        f"{prefix}grads/combined_loss_grad_total_cosine": combined_total_cos.detach().cpu(),
                        f"{prefix}grads/total_grad_norm": grad_norm(full_param_grad).detach().cpu(),
                        f"{prefix}grads/decomposition_residual_norm": decomposition_residual.detach().cpu(),
                    }
                )
            finally:
                if was_training:
                    model.train()

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            sft_loss = nn.CrossEntropyLoss()(
                policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]),
                chosen_labels.view(-1),
            )
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")
        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        self.store_metrics(metrics, train_eval="eval")
        if prediction_loss_only:
            return loss.detach(), None, None
        logits = torch.stack(
            [metrics["eval_logits/chosen"].unsqueeze(0), metrics["eval_logits/rejected"].unsqueeze(0)]
        ).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        return loss.detach(), logits, labels

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        append_seeking_log("simpo", logs)
        return super().log(logs, start_time)
