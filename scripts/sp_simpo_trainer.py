from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from entmax import sparsemax

from seeking_utils import append_seeking_log, combine_grads, get_last_layer_weight, grad_cosine, grad_norm
from simpo_trainer import SimPOTrainer


def _get_batch_sp_score_simpo(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    alpha: float = 1.5,
    beta: float = 0.5,
    temperature: float = 2.0,
    average_score: bool = True,
    neg_support_coef: float = 1.0,
    ispos: bool = False,
):
    B, M, V = logits.shape

    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)

    flat_logits = logits.reshape(-1, V)
    flat_labels = safe_labels.reshape(-1)

    sparse_probs = sparsemax(flat_logits/temperature, -1)
    z_y = flat_logits.gather(dim=1, index=flat_labels.unsqueeze(1)).squeeze(1)

    flat_loss = (
        (sparse_probs * flat_logits).sum(dim=-1)
        - 0.5 * sparse_probs.pow(2).sum(dim=-1)
        - z_y
        + 0.5
    )

    # if not ispos:
    #     target_sparse_probs = sparse_probs.gather(dim=1, index=flat_labels.unsqueeze(1)).squeeze(1)
    #     is_inside_support = target_sparse_probs != 0.0
    #     coef = torch.where(
    #         is_inside_support,
    #         torch.full_like(flat_loss, neg_support_coef),
    #         torch.ones_like(flat_loss),
    #     )
    #     # Keep the forward score unchanged while scaling gradients on the
    #     # selected negative tokens by `neg_support_coef`.
    #     flat_loss = flat_loss * coef + flat_loss.detach() * (1.0 - coef)

    token_loss = flat_loss.view(B, M) * mask.float()
    lengths = mask.sum(-1).clamp_min(1)
    seq_scores = -token_loss.sum(-1)
    if average_score:
        seq_scores = seq_scores / lengths

    return seq_scores


class SPSimPOTrainer(SimPOTrainer):
    def __init__(
        self,
        *args,
        sp_alpha: float = 1.5,
        sp_beta: float = 0.2,
        sp_temperature: float = 2.0,
        sp_neg_support_coef: float = 1.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sp_alpha = sp_alpha
        self.sp_beta = sp_beta
        self.sp_temperature = sp_temperature
        self.sp_neg_support_coef = sp_neg_support_coef

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

        score_logits = all_logits
        score_labels = concatenated_batch["concatenated_labels"]
        if not self.is_encoder_decoder:
            score_labels = score_labels[:, 1:].clone()
            score_logits = score_logits[:, :-1, :]

        chosen_scores = _get_batch_sp_score_simpo(
            score_logits[:len_chosen],
            score_labels[:len_chosen],
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            temperature=self.sp_temperature,
            average_score=True,
            neg_support_coef=self.sp_neg_support_coef,
            ispos=True,
        )
        rejected_scores = _get_batch_sp_score_simpo(
            score_logits[len_chosen:],
            score_labels[len_chosen:],
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            temperature=self.sp_temperature,
            average_score=True,
            neg_support_coef=self.sp_neg_support_coef,
            ispos=False,
        )

        return (
            chosen_scores,
            rejected_scores,
            all_logits[:len_chosen],
            all_logits[len_chosen:],
            concatenated_batch["concatenated_labels"][:len_chosen],
            concatenated_batch["concatenated_labels"][len_chosen:],
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        prefix = "seeking_eval/" if train_eval == "eval" else "seeking/"
        (
            policy_chosen_scores,
            policy_rejected_scores,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_scores, policy_rejected_scores)
        loss = losses.mean()
        real_chosen_logps = self.get_batch_logps(
            policy_chosen_logits,
            chosen_labels,
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        real_rejected_logps = self.get_batch_logps(
            policy_rejected_logits,
            rejected_labels,
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        raw_margin = policy_chosen_scores - policy_rejected_scores
        objective_margin = self.beta * (raw_margin - self.gamma_beta_ratio)
        sigmoid_margin = torch.sigmoid(objective_margin)
        objective_accuracy = (policy_chosen_scores > policy_rejected_scores).float()
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().cpu(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().cpu(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().cpu(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
            f"{prefix}scores/chosen": policy_chosen_scores.detach().mean().cpu(),
            f"{prefix}scores/rejected": policy_rejected_scores.detach().mean().cpu(),
            f"{prefix}scores/accuracies": objective_accuracy.detach().mean().cpu(),
            f"{prefix}scores/raw_margins": raw_margin.detach().mean().cpu(),
            f"{prefix}scores/objective_margins": objective_margin.detach().mean().cpu(),
            f"{prefix}scores/sigmoid_margins": sigmoid_margin.detach().mean().cpu(),
            f"{prefix}logps/chosen": real_chosen_logps.detach().mean().cpu(),
            f"{prefix}logps/rejected": real_rejected_logps.detach().mean().cpu(),
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
                # sparsemax graph for repeated autograd.grad calls can trigger shape
                # mismatches on distributed runs.
                def _fresh_policy_outputs():
                    return self.concatenated_forward(model, batch)

                (
                    fresh_chosen_scores,
                    fresh_rejected_scores,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                chosen_only_losses, _, _ = self.simpo_loss(fresh_chosen_scores, fresh_rejected_scores.detach())
                chosen_param_grad = torch.autograd.grad(
                    chosen_only_losses.mean(), last_layer_weight, retain_graph=False, allow_unused=True
                )[0]

                (
                    fresh_chosen_scores,
                    fresh_rejected_scores,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                rejected_only_losses, _, _ = self.simpo_loss(fresh_chosen_scores.detach(), fresh_rejected_scores)
                rejected_param_grad = torch.autograd.grad(
                    rejected_only_losses.mean(), last_layer_weight, retain_graph=False, allow_unused=True
                )[0]
                total_param_grad = combine_grads(chosen_param_grad, rejected_param_grad)

                (
                    fresh_chosen_scores,
                    fresh_rejected_scores,
                    _fresh_chosen_logits,
                    _fresh_rejected_logits,
                    _fresh_chosen_labels,
                    _fresh_rejected_labels,
                ) = _fresh_policy_outputs()
                full_losses, _, _ = self.simpo_loss(fresh_chosen_scores, fresh_rejected_scores)
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

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        append_seeking_log("sp_simpo", logs)
        return super().log(logs, start_time)
