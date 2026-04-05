import os
import torch
import torch.nn.functional as F
from deepspeed.utils import safe_get_full_grad, safe_set_full_grad
from accelerate.utils import DistributedType

from dpo_metrics_trainer import DPOMetricsTrainer
from seeking_utils import (
    append_seeking_log,
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
        dlambda_logp_aggregation: str = "sum",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dlambda_alpha = dlambda_alpha
        self.dlambda_epsilon = dlambda_epsilon
        self.dlambda_lambda_max = dlambda_lambda_max
        self.dlambda_grad_target = dlambda_grad_target
        self.dlambda_reference_free = dlambda_reference_free
        self.dlambda_logp_aggregation = dlambda_logp_aggregation
        self.grad_rewrite_debug = False
        self.grad_collection_debug = os.getenv("DYNAMIC_LAMBDA_GRAD_DEBUG", "0") == "1"
        self._last_grad_rewrite_step = -1
        self._last_grad_collection_debug_step = -1
        self._grad_f_buffer = {}
        self._grad_g_buffer = {}
        self._grad_buffer_dtypes = {}
        self._constraint_sum = None
        self._constraint_count = 0

    def _trainable_grad_params(self, model):
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and param.numel() > 0
        ]

    def _lambda_param_names(self, model):
        trainable_params = self._trainable_grad_params(model)
        if self.dlambda_grad_target == "last_three_layers":
            return [name for name, _ in trainable_params[-3:]]
        return [name for name, _ in trainable_params]

    def collect_full_grads(self, engine, target_names=None):
        target_names = set(target_names) if target_names is not None else None
        params = {}
        grads = {}
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue
            if target_names is not None and name not in target_names:
                continue
            grad = safe_get_full_grad(param)
            if grad is None:
                continue
            params[name] = param
            grads[name] = grad
        return params, grads

    def _accumulate_grad_buffer(self, buffer_dict, grads):
        for name, grad in grads.items():
            self._grad_buffer_dtypes[name] = grad.dtype
            grad_buffer = grad.detach().to(dtype=torch.float16)
            if name in buffer_dict:
                buffer_dict[name] = buffer_dict[name] + grad_buffer
            else:
                buffer_dict[name] = grad_buffer.clone()

    def _reset_grad_rewrite_state(self):
        self._grad_f_buffer = {}
        self._grad_g_buffer = {}
        self._grad_buffer_dtypes = {}
        self._constraint_sum = None
        self._constraint_count = 0

    def _maybe_debug_grad_collection(self, step_label, target_names, grads, loss_name):
        if not self.grad_collection_debug:
            return
        step_id = f"{self.state.global_step}:{step_label}:{loss_name}"
        if step_id == self._last_grad_collection_debug_step:
            return
        self._last_grad_collection_debug_step = step_id

        found_names = list(grads.keys())
        missing_names = [name for name in target_names if name not in grads]
        sample_found = found_names[:3]
        sample_missing = missing_names[:3]
        print(
            "[grad_collection_debug]",
            f"global_step={self.state.global_step}",
            f"loss_name={loss_name}",
            f"target_count={len(target_names)}",
            f"found_count={len(found_names)}",
            f"missing_count={len(missing_names)}",
            f"sample_found={sample_found}",
            f"sample_missing={sample_missing}",
        )

    def _compute_batch_state(self, model, batch):
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
        constraint_value = self._constraint_value(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )
        return {
            "model_output": model_output,
            "ref_chosen_logps": ref_chosen_logps,
            "ref_rejected_logps": ref_rejected_logps,
            "policy_margin": policy_margin,
            "preference_loss": preference_loss,
            "constraint_value": constraint_value,
        }

    def _compute_dynamic_lambda_from_buffers(self, device):
        if not self._grad_g_buffer:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        grad_target_names = self._lambda_param_names(self.deepspeed.module)
        grad_f = [self._grad_f_buffer.get(name) for name in grad_target_names]
        grad_g = [self._grad_g_buffer.get(name) for name in grad_target_names]

        grad_inner = self._grad_list_inner(grad_g, grad_f, device=device)
        grad_g_norm = self._grad_list_norm(grad_g, device=device)
        grad_g_norm_sq = grad_g_norm.pow(2)

        if self._constraint_sum is None or self._constraint_count == 0:
            constraint_mean = torch.tensor(0.0, device=device)
        else:
            constraint_mean = self._constraint_sum / self._constraint_count

        numerator = self.dlambda_alpha * constraint_mean - grad_inner.detach()
        dynamic_lambda = torch.clamp(numerator / grad_g_norm_sq.clamp_min(1e-12), min=0.0)
        if self.dlambda_lambda_max is not None:
            dynamic_lambda = torch.clamp(dynamic_lambda, max=self.dlambda_lambda_max)

        return dynamic_lambda, grad_inner, grad_g_norm_sq

    def compute_lambda(self, params):
        lambda_value, _, _ = self._compute_dynamic_lambda_from_buffers(
            device=next(iter(params.values())).device if params else torch.device("cpu")
        )
        return {
            name: lambda_value.to(device=param.device, dtype=param.dtype)
            for name, param in params.items()
        }

    def rewrite_grads(self, params, grad_f, grad_g, lambda_dict):
        debug_rows = []
        for name, param in params.items():
            grad_pref = grad_f.get(name)
            grad_constraint = grad_g.get(name)
            lam = lambda_dict.get(name)
            if grad_pref is None and grad_constraint is None:
                continue
            if grad_pref is None:
                grad_pref = torch.zeros_like(grad_constraint)
            if grad_constraint is None:
                grad_constraint = torch.zeros_like(grad_pref)
            if lam is None:
                lam = torch.tensor(0.0, device=grad_pref.device, dtype=grad_pref.dtype)
            target_dtype = self._grad_buffer_dtypes.get(name, grad_pref.dtype)
            new_grad = (grad_pref + lam * grad_constraint).to(dtype=target_dtype)
            safe_set_full_grad(param, new_grad)

            if self.grad_rewrite_debug and len(debug_rows) < 2:
                orig_grad = grad_pref + grad_constraint
                debug_rows.append(
                    {
                        "name": name,
                        "orig_norm": orig_grad.float().norm().item(),
                        "lambda": lam.float().mean().item(),
                        "new_norm": new_grad.float().norm().item(),
                    }
                )
        return debug_rows

    def _flatten_grad_list(self, grads):
        if grads is None:
            return None
        flat_parts = [grad.reshape(-1) for grad in grads if grad is not None]
        if not flat_parts:
            return None
        return torch.cat(flat_parts)

    def _grad_list_norm(self, grads, device=None):
        flat = self._flatten_grad_list(grads)
        if flat is None:
            if device is None:
                return torch.tensor(0.0)
            return torch.tensor(0.0, device=device)
        return flat.norm()

    def _grad_list_inner(self, grads_a, grads_b, device=None):
        flat_a = self._flatten_grad_list(grads_a)
        flat_b = self._flatten_grad_list(grads_b)

        result_device = device
        for tensor in (flat_a, flat_b):
            if tensor is not None:
                result_device = tensor.device
                break

        if flat_a is None or flat_b is None:
            if result_device is None:
                return torch.tensor(0.0)
            return torch.tensor(0.0, device=result_device)

        return torch.dot(flat_a, flat_b)

    def _lambda_params(self, model):
        trainable_params = self._trainable_grad_params(model)
        if self.dlambda_grad_target == "last_three_layers":
            return [param for _, param in trainable_params[-3:]]
        return [param for _, param in trainable_params]

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

    # def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
    #     batch_state = self._compute_batch_state(model, batch)
    #     model_output = batch_state["model_output"]
    #     policy_margin = batch_state["policy_margin"]
    #     preference_loss = batch_state["preference_loss"]
    #     constraint_value = batch_state["constraint_value"]

    #     lambda_params = self._lambda_params(model)
    #     grad_inner = torch.tensor(0.0, device=preference_loss.device)
    #     grad_g_norm_sq = torch.tensor(0.0, device=preference_loss.device)
    #     dynamic_lambda = torch.tensor(0.0, device=preference_loss.device)

    #     if train_eval == "train" and lambda_params:
    #         grad_f = torch.autograd.grad(
    #             preference_loss,
    #             lambda_params,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )
    #         grad_g = torch.autograd.grad(
    #             constraint_value,
    #             lambda_params,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )
    #         grad_inner = self._grad_list_inner(grad_g, grad_f, device=preference_loss.device)
    #         grad_g_norm = self._grad_list_norm(grad_g, device=preference_loss.device)
    #         grad_g_norm_sq = grad_g_norm.pow(2)

    #         numerator = self.dlambda_alpha * constraint_value.detach() - grad_inner.detach()
    #         dynamic_lambda = torch.clamp(numerator / grad_g_norm_sq.clamp_min(1e-12), min=0.0)
    #         if self.dlambda_lambda_max is not None:
    #             dynamic_lambda = torch.clamp(dynamic_lambda, max=self.dlambda_lambda_max)

    #     loss = preference_loss + dynamic_lambda.detach() * constraint_value

    #     sigmoid_margin = torch.sigmoid(policy_margin)
    #     objective_accuracy = (policy_margin > 0).float()

    #     prefix = "eval/" if train_eval == "eval" else "train/"
    #     metrics = {
    #         f"{prefix}loss/preference": self.accelerator.gather_for_metrics(preference_loss.detach()).mean().item(),
    #         f"{prefix}loss/constraint": self.accelerator.gather_for_metrics(constraint_value.detach()).mean().item(),
    #         f"{prefix}loss/dynamic_lambda": self.accelerator.gather_for_metrics(dynamic_lambda.detach()).mean().item(),
    #         f"{prefix}loss/total": self.accelerator.gather_for_metrics(loss.detach()).mean().item(),
    #         f"{prefix}grads/constraint_pref_inner": self.accelerator.gather_for_metrics(grad_inner.detach()).mean().item(),
    #         f"{prefix}grads/constraint_grad_norm_sq": (
    #             self.accelerator.gather_for_metrics(grad_g_norm_sq.detach()).mean().item()
    #         ),
    #         # f"{prefix}scores/chosen": (
    #         #     self.accelerator.gather_for_metrics(model_output["chosen_logps"].detach()).mean().item()
    #         # ),
    #         # f"{prefix}scores/rejected": (
    #         #     self.accelerator.gather_for_metrics(model_output["rejected_logps"].detach()).mean().item()
    #         # ),
    #         f"{prefix}/accuracies": self.accelerator.gather_for_metrics(objective_accuracy.detach()).mean().item(),
    #         f"{prefix}/margins": self.accelerator.gather_for_metrics(policy_margin.detach()).mean().item(),
    #         f"{prefix}/sigmoid_margins": (
    #             self.accelerator.gather_for_metrics(sigmoid_margin.detach()).mean().item()
    #         ),
    #     }

    #     return loss, metrics

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            batch_state = self._compute_batch_state(model, inputs)

        preference_loss = batch_state["preference_loss"]
        constraint_value = batch_state["constraint_value"]
        policy_margin = batch_state["policy_margin"]
        loss = preference_loss

        sigmoid_margin = torch.sigmoid(policy_margin)
        objective_accuracy = (policy_margin > 0).float()
        prefix = "train/"
        metrics = {
            f"{prefix}loss/preference": self.accelerator.gather_for_metrics(preference_loss.detach()).mean().item(),
            f"{prefix}loss/constraint": self.accelerator.gather_for_metrics(constraint_value.detach()).mean().item(),
            f"{prefix}loss/dynamic_lambda": 0.0,
            f"{prefix}loss/total": self.accelerator.gather_for_metrics(preference_loss.detach()).mean().item(),
            f"{prefix}grads/constraint_pref_inner": 0.0,
            f"{prefix}grads/constraint_grad_norm_sq": 0.0,
            f"{prefix}/accuracies": self.accelerator.gather_for_metrics(objective_accuracy.detach()).mean().item(),
            f"{prefix}/margins": self.accelerator.gather_for_metrics(policy_margin.detach()).mean().item(),
            f"{prefix}/sigmoid_margins": (
                self.accelerator.gather_for_metrics(sigmoid_margin.detach()).mean().item()
            ),
        }

        engine = self.deepspeed
        target_names = self._lambda_param_names(engine.module)
        engine.zero_grad()
        engine.backward(preference_loss, retain_graph=True)
        params, grads_f = self.collect_full_grads(engine, target_names=target_names)
        self._maybe_debug_grad_collection("after_backward", target_names, grads_f, "preference")
        self._accumulate_grad_buffer(self._grad_f_buffer, grads_f)

        engine.zero_grad()
        engine.backward(constraint_value)
        _, grads_g = self.collect_full_grads(engine, target_names=target_names)
        self._maybe_debug_grad_collection("after_backward", target_names, grads_g, "constraint")
        self._accumulate_grad_buffer(self._grad_g_buffer, grads_g)
        engine.zero_grad()

        constraint_detached = constraint_value.detach()
        if self._constraint_sum is None:
            self._constraint_sum = constraint_detached
        else:
            self._constraint_sum = self._constraint_sum + constraint_detached
        self._constraint_count += 1

        if engine.is_gradient_accumulation_boundary():
            all_params = {
                name: param
                for name, param in engine.module.named_parameters()
                if param.requires_grad and param.numel() > 0 and name in target_names
            }
            lambda_value, grad_inner, grad_g_norm_sq = self._compute_dynamic_lambda_from_buffers(
                device=preference_loss.device
            )
            lambda_dict = {
                name: lambda_value.to(device=param.device, dtype=param.dtype)
                for name, param in all_params.items()
            }
            debug_rows = self.rewrite_grads(all_params, self._grad_f_buffer, self._grad_g_buffer, lambda_dict)

            metrics[f"{prefix}loss/dynamic_lambda"] = (
                self.accelerator.gather_for_metrics(lambda_value.detach()).mean().item()
            )
            metrics[f"{prefix}loss/total"] = (
                self.accelerator.gather_for_metrics(
                    (preference_loss + lambda_value.detach() * constraint_value).detach()
                ).mean().item()
            )
            metrics[f"{prefix}grads/constraint_pref_inner"] = (
                self.accelerator.gather_for_metrics(grad_inner.detach()).mean().item()
            )
            metrics[f"{prefix}grads/constraint_grad_norm_sq"] = (
                self.accelerator.gather_for_metrics(grad_g_norm_sq.detach()).mean().item()
            )

            if self.grad_rewrite_debug and self.state.global_step != self._last_grad_rewrite_step:
                self._last_grad_rewrite_step = self.state.global_step
                for row in debug_rows:
                    print(
                        "[grad_rewrite]",
                        f"step={self.state.global_step}",
                        f"name={row['name']}",
                        f"orig_norm={row['orig_norm']:.6f}",
                        f"lambda={row['lambda']:.6f}",
                        f"new_norm={row['new_norm']:.6f}",
                    )

            self._reset_grad_rewrite_state()

        self.store_metrics(metrics, train_eval="train")
        return loss.detach()

    def log(self, logs, start_time=None):
        append_seeking_log("dynamic_lambda_dpo", logs)
        return super().log(logs, start_time)
