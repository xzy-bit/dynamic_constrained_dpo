from typing import Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import flush_left, selective_log_softmax

from seeking_utils import append_seeking_log, combine_grads, get_last_layer_weight, grad_cosine, grad_norm


class DPOMetricsTrainer(DPOTrainer):
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)

            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                    )

            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)
        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["chosen_labels"] = labels[:num_examples]
        output["rejected_labels"] = labels[num_examples:]
        output["chosen_loss_mask"] = loss_mask[:num_examples]
        output["rejected_loss_mask"] = loss_mask[num_examples:]

        if self.padding_free:
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
            raw_chosen_logits = logits[0, :split_idx].unsqueeze(0)
            raw_rejected_logits = logits[0, split_idx:].unsqueeze(0)
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()
            raw_chosen_logits = logits[:num_examples]
            raw_rejected_logits = logits[num_examples:]

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits
        output["chosen_logits"] = raw_chosen_logits
        output["rejected_logits"] = raw_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval=train_eval)

        model_output = self.concatenated_forward(model, batch)
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        chosen_score = model_output["chosen_logps"] - ref_chosen_logps
        rejected_score = model_output["rejected_logps"] - ref_rejected_logps
        raw_margin = chosen_score - rejected_score
        objective_margin = self.beta * raw_margin
        sigmoid_margin = torch.sigmoid(objective_margin)
        objective_accuracy = (chosen_score > rejected_score).float()
        last_layer_weight = get_last_layer_weight(model)
        chosen_only_losses, _, _ = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"].detach(),
            ref_chosen_logps,
            ref_rejected_logps,
        )
        rejected_only_losses, _, _ = self.dpo_loss(
            model_output["chosen_logps"].detach(),
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )
        chosen_param_grad = torch.autograd.grad(
            chosen_only_losses.mean(), last_layer_weight, retain_graph=True, allow_unused=True
        )[0]
        rejected_param_grad = torch.autograd.grad(
            rejected_only_losses.mean(), last_layer_weight, retain_graph=True, allow_unused=True
        )[0]
        total_param_grad = combine_grads(chosen_param_grad, rejected_param_grad)
        full_param_grad = torch.autograd.grad(loss, last_layer_weight, retain_graph=True, allow_unused=True)[0]
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
        remapped_metrics = {}
        for key, value in metrics.items():
            if key.startswith("eval_"):
                remapped_metrics[f"seeking_eval/{key[len('eval_'):]}"] = value
            else:
                remapped_metrics[f"seeking/{key}"] = value
        metrics = remapped_metrics

        prefix = "seeking_eval/" if train_eval == "eval" else "seeking/"
        metrics[f"{prefix}scores/chosen"] = self.accelerator.gather_for_metrics(chosen_score).detach().mean().item()
        metrics[f"{prefix}scores/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_score).detach().mean().item()
        )
        metrics[f"{prefix}scores/accuracies"] = (
            self.accelerator.gather_for_metrics(objective_accuracy).detach().mean().item()
        )
        metrics[f"{prefix}scores/raw_margins"] = (
            self.accelerator.gather_for_metrics(raw_margin).detach().mean().item()
        )
        metrics[f"{prefix}scores/objective_margins"] = (
            self.accelerator.gather_for_metrics(objective_margin).detach().mean().item()
        )
        metrics[f"{prefix}scores/sigmoid_margins"] = (
            self.accelerator.gather_for_metrics(sigmoid_margin).detach().mean().item()
        )
        metrics[f"{prefix}grads/chosen_grad_norm"] = (
            self.accelerator.gather_for_metrics(chosen_grad_norm).detach().mean().item()
        )
        metrics[f"{prefix}grads/rejected_grad_norm"] = (
            self.accelerator.gather_for_metrics(rejected_grad_norm).detach().mean().item()
        )
        metrics[f"{prefix}grads/chosen_rejected_grad_norm_ratio"] = (
            self.accelerator.gather_for_metrics(chosen_rejected_grad_norm_ratio).detach().mean().item()
        )
        metrics[f"{prefix}grads/chosen_loss_grad_total_cosine"] = (
            self.accelerator.gather_for_metrics(chosen_total_cos).detach().mean().item()
        )
        metrics[f"{prefix}grads/rejected_loss_grad_total_cosine"] = (
            self.accelerator.gather_for_metrics(rejected_total_cos).detach().mean().item()
        )
        metrics[f"{prefix}grads/chosen_rejected_grad_cosine"] = (
            self.accelerator.gather_for_metrics(chosen_rejected_cos).detach().mean().item()
        )
        metrics[f"{prefix}grads/combined_loss_grad_total_cosine"] = (
            self.accelerator.gather_for_metrics(combined_total_cos).detach().mean().item()
        )
        metrics[f"{prefix}grads/total_grad_norm"] = (
            self.accelerator.gather_for_metrics(grad_norm(full_param_grad)).detach().mean().item()
        )
        metrics[f"{prefix}grads/decomposition_residual_norm"] = (
            self.accelerator.gather_for_metrics(decomposition_residual).detach().mean().item()
        )
        return loss, metrics

    def log(self, logs, start_time=None):
        append_seeking_log("dpo", logs)
        return super().log(logs, start_time)
