import torch
import torch.nn.functional as F
from trl import DPOTrainer

from dpo_metrics_trainer import DPOMetricsTrainer
from seeking_utils import (
    append_seeking_log,
    get_last_layer_weight,
    grad_list_inner,
    grad_list_norm,
)


class DynamicLambdaDPOTrainer(DPOMetricsTrainer):
    def __init__(
        self,
        *args,
        dlambda_alpha: float = 1.0,
        dlambda_epsilon: float = 0.0,
        dlambda_lambda_max: float | None = 10.0,
        dlambda_grad_target: str = "all",
        dlambda_reference_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dlambda_alpha = dlambda_alpha
        self.dlambda_epsilon = dlambda_epsilon
        self.dlambda_lambda_max = dlambda_lambda_max
        self.dlambda_grad_target = dlambda_grad_target
        self.dlambda_reference_free = dlambda_reference_free

    def _lambda_params(self, model):
        if self.dlambda_grad_target == "last_layer":
            last_layer_weight = get_last_layer_weight(model)
            return [last_layer_weight] if last_layer_weight is not None else []
        return [param for param in model.parameters() if param.requires_grad]

    def _constraint_value(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        # logp是最容易拿到的，在这里做近似
        # 用policy和ref的logp差值平均值来近似KL divergence surrogate constraint g_hat
        chosen_kl = policy_chosen_logps - ref_chosen_logps
        rejected_kl = policy_rejected_logps - ref_rejected_logps
        surrogate_kl = 0.5 * (chosen_kl.mean() + rejected_kl.mean())
        return surrogate_kl - self.dlambda_epsilon

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        model_output = self.concatenated_forward(model, batch)
        if self.dlambda_reference_free:
            ref_chosen_logps = torch.zeros_like(model_output["chosen_logps"])
            ref_rejected_logps = torch.zeros_like(model_output["rejected_logps"])
        elif "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        policy_margin = model_output["chosen_logps"] - model_output["rejected_logps"]
        preference_losses = -F.logsigmoid(policy_margin)
        preference_loss = preference_losses.mean()

        # KL Divergence
        constraint_value = self._constraint_value(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )

        lambda_params = self._lambda_params(model)
        grad_inner = torch.tensor(0.0, device=preference_loss.device)
        grad_g_norm_sq = torch.tensor(0.0, device=preference_loss.device)
        dynamic_lambda = torch.tensor(0.0, device=preference_loss.device)

        if train_eval == "train" and lambda_params:
            grad_f = torch.autograd.grad(
                preference_loss,
                lambda_params,
                retain_graph=True,
                allow_unused=True,
            )
            grad_g = torch.autograd.grad(
                constraint_value,
                lambda_params,
                retain_graph=True,
                allow_unused=True,
            )
            grad_inner = grad_list_inner(grad_g, grad_f)
            grad_g_norm = grad_list_norm(grad_g)
            grad_g_norm_sq = grad_g_norm.pow(2)

            numerator = self.dlambda_alpha * constraint_value.detach() - grad_inner.detach()
            dynamic_lambda = torch.clamp(numerator / grad_g_norm_sq.clamp_min(1e-12), min=0.0)
            if self.dlambda_lambda_max is not None:
                dynamic_lambda = torch.clamp(dynamic_lambda, max=self.dlambda_lambda_max)

        loss = preference_loss + dynamic_lambda.detach() * constraint_value

        sigmoid_margin = torch.sigmoid(policy_margin)
        objective_accuracy = (policy_margin > 0).float()

        prefix = "eval/" if train_eval == "eval" else "train/"
        metrics = {
            f"{prefix}loss/preference": self.accelerator.gather_for_metrics(preference_loss.detach()).mean().item(),
            f"{prefix}loss/constraint": self.accelerator.gather_for_metrics(constraint_value.detach()).mean().item(),
            f"{prefix}loss/dynamic_lambda": self.accelerator.gather_for_metrics(dynamic_lambda.detach()).mean().item(),
            f"{prefix}loss/total": self.accelerator.gather_for_metrics(loss.detach()).mean().item(),
            # f"{prefix}grads/constraint_pref_inner": self.accelerator.gather_for_metrics(grad_inner.detach()).mean().item(),
            # f"{prefix}grads/constraint_grad_norm_sq": (
            #     self.accelerator.gather_for_metrics(grad_g_norm_sq.detach()).mean().item()
            # ),
            # f"{prefix}scores/chosen": (
            #     self.accelerator.gather_for_metrics(model_output["chosen_logps"].detach()).mean().item()
            # ),
            # f"{prefix}scores/rejected": (
            #     self.accelerator.gather_for_metrics(model_output["rejected_logps"].detach()).mean().item()
            # ),
            f"{prefix}/accuracies": self.accelerator.gather_for_metrics(objective_accuracy.detach()).mean().item(),
            f"{prefix}/margins": self.accelerator.gather_for_metrics(policy_margin.detach()).mean().item(),
            f"{prefix}/sigmoid_margins": (
                self.accelerator.gather_for_metrics(sigmoid_margin.detach()).mean().item()
            ),
        }

        return loss, metrics

    def log(self, logs, start_time=None):
        append_seeking_log("dynamic_lambda_dpo", logs)
        return super().log(logs, start_time)
