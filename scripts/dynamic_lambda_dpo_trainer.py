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
        self._final_grad_buffer = {}
        self._grad_buffer_dtypes = {}
        self._manual_micro_steps = 0

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
        sampled = 0
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue
            if target_names is not None and name not in target_names:
                continue
            if sampled < 2:
                self._maybe_debug_memory_probe("before_safe_get_full_grad", {"name": name})
            grad = safe_get_full_grad(param)
            if sampled < 2:
                extra = {"name": name, "grad_is_none": grad is None}
                if grad is not None:
                    extra["grad_dtype"] = str(grad.dtype)
                    extra["grad_numel"] = grad.numel()
                    extra["grad_elem_bytes"] = grad.element_size()
                self._maybe_debug_memory_probe("after_safe_get_full_grad", extra)
                sampled += 1
            if grad is None:
                continue
            params[name] = param
            grads[name] = grad
        return params, grads

    def _accumulate_grad_buffer(self, buffer_dict, grads):
        for idx, (name, grad) in enumerate(grads.items()):
            self._grad_buffer_dtypes[name] = grad.dtype
            if idx < 2:
                self._maybe_debug_memory_probe("before_grad_to_fp16", {"name": name, "grad_dtype": str(grad.dtype)})
            grad_buffer = grad.detach().to(dtype=torch.float16)
            if idx < 2:
                self._maybe_debug_memory_probe(
                    "after_grad_to_fp16",
                    {"name": name, "buffer_dtype": str(grad_buffer.dtype), "buffer_numel": grad_buffer.numel()},
                )
            if name in buffer_dict:
                buffer_dict[name] = buffer_dict[name] + grad_buffer
            else:
                buffer_dict[name] = grad_buffer.clone()

    def _accumulate_scaled_grad_buffer(self, buffer_dict, grads, scale):
        scale_value = float(scale.detach().float().item()) if isinstance(scale, torch.Tensor) else float(scale)
        for idx, (name, grad) in enumerate(grads.items()):
            self._grad_buffer_dtypes[name] = grad.dtype
            if idx < 2:
                self._maybe_debug_memory_probe(
                    "before_scaled_grad_to_fp16",
                    {"name": name, "grad_dtype": str(grad.dtype), "scale": f"{scale_value:.6e}"},
                )
            grad_buffer = (grad.detach().float() * scale_value).to(dtype=torch.float16)
            if idx < 2:
                self._maybe_debug_memory_probe(
                    "after_scaled_grad_to_fp16",
                    {"name": name, "buffer_dtype": str(grad_buffer.dtype), "buffer_numel": grad_buffer.numel()},
                )
            if name in buffer_dict:
                buffer_dict[name] = buffer_dict[name] + grad_buffer
            else:
                buffer_dict[name] = grad_buffer.clone()

    def _reset_grad_rewrite_state(self):
        self._final_grad_buffer = {}
        self._grad_buffer_dtypes = {}

    def _debug_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _buffer_debug_summary(self, buffer_dict):
        if not buffer_dict:
            return "count=0"

        total_norm = 0.0
        nonzero_count = 0
        for grad in buffer_dict.values():
            grad_norm = grad.float().norm().item()
            total_norm += grad_norm
            if grad_norm > 0:
                nonzero_count += 1
        return (
            f"count={len(buffer_dict)} "
            f"nonzero_count={nonzero_count} "
            f"total_norm={total_norm:.6f}"
        )

    def _maybe_debug_step_state(
        self,
        step_label,
        engine,
        boundary=None,
    ):
        if not self.grad_collection_debug:
            return
        ds_grad_acc_steps = (
            engine.gradient_accumulation_steps() if hasattr(engine, "gradient_accumulation_steps") else "n/a"
        )
        micro_steps = getattr(engine, "micro_steps", "n/a")
        manual_micro_steps = self._manual_micro_steps

        print(
            "[dlambda_step]",
            f"rank={self._debug_rank()}",
            f"step={self.state.global_step}",
            f"phase={step_label}",
            f"micro_steps={micro_steps}",
            f"manual_micro_steps={manual_micro_steps}",
            f"trainer_gas={self.args.gradient_accumulation_steps}",
            f"engine_gas={ds_grad_acc_steps}",
            f"boundary={boundary}",
        )

    def _maybe_debug_grad_collection(self, step_label, target_names, grads, loss_name):
        if not self.grad_collection_debug:
            return
        return

    def _cuda_mem_stats(self):
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        return {
            "device": device,
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        }

    def _maybe_debug_memory_probe(self, label, extra=None):
        if not self.grad_collection_debug:
            return
        stats = self._cuda_mem_stats()
        if stats is None:
            return
        parts = [
            "[dlambda_mem]",
            f"rank={self._debug_rank()}",
            f"step={self.state.global_step}",
            f"label={label}",
            f"allocated_mb={stats['allocated_mb']:.2f}",
            f"reserved_mb={stats['reserved_mb']:.2f}",
            f"max_allocated_mb={stats['max_allocated_mb']:.2f}",
        ]
        if extra:
            for key, value in extra.items():
                parts.append(f"{key}={value}")
        print(*parts)

    def _maybe_debug_constraint_components(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        constraint_value: torch.Tensor,
    ):
        if not self.grad_collection_debug:
            return

        policy_chosen_mean = policy_chosen_logps.float().mean()
        policy_rejected_mean = policy_rejected_logps.float().mean()
        ref_chosen_mean = ref_chosen_logps.float().mean()
        ref_rejected_mean = ref_rejected_logps.float().mean()
        chosen_kl = (policy_chosen_logps.float() - ref_chosen_logps.float()).mean()
        rejected_kl = (policy_rejected_logps.float() - ref_rejected_logps.float()).mean()

        print(
            "[dlambda_constraint_raw]",
            f"rank={self._debug_rank()}",
            f"step={self.state.global_step}",
            f"policy_chosen_mean={policy_chosen_mean.detach().item():.12e}",
            f"ref_chosen_mean={ref_chosen_mean.detach().item():.12e}",
            f"chosen_kl_mean={chosen_kl.detach().item():.12e}",
            f"policy_rejected_mean={policy_rejected_mean.detach().item():.12e}",
            f"ref_rejected_mean={ref_rejected_mean.detach().item():.12e}",
            f"rejected_kl_mean={rejected_kl.detach().item():.12e}",
            f"constraint_value={constraint_value.detach().float().item():.12e}",
        )

    def _compute_batch_state(self, model, batch):
        model_output = self.concatenated_forward(model, batch)
        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        del model_output

        if self.dlambda_reference_free:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        elif "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        policy_margin = chosen_logps - rejected_logps
        preference_losses = -F.logsigmoid(policy_margin)
        preference_loss = preference_losses.mean()
        constraint_value = self._constraint_value(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        self._maybe_debug_constraint_components(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            constraint_value,
        )
        return {
            "policy_margin": policy_margin,
            "preference_loss": preference_loss,
            "constraint_value": constraint_value,
        }

    def _compute_dynamic_lambda_from_grads(self, grads_f, grads_g, constraint_value, device):
        if not grads_g:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        grad_target_names = self._lambda_param_names(self.deepspeed.module)
        grad_f = [grads_f.get(name) for name in grad_target_names]
        grad_g = [grads_g.get(name) for name in grad_target_names]

        grad_inner = self._grad_list_inner(grad_g, grad_f, device=device)
        grad_g_norm = self._grad_list_norm(grad_g, device=device)
        grad_g_norm_sq = grad_g_norm.pow(2)
        constraint_mean = constraint_value.detach().float()

        numerator = self.dlambda_alpha * constraint_mean - grad_inner.detach()
        dynamic_lambda = torch.clamp(numerator / grad_g_norm_sq.clamp_min(1e-12), min=0.0)
        if self.dlambda_lambda_max is not None:
            dynamic_lambda = torch.clamp(dynamic_lambda, max=self.dlambda_lambda_max)

        if self.grad_collection_debug:
            print(
                "[dlambda_value]",
                f"rank={self._debug_rank()}",
                f"step={self.state.global_step}",
                f"constraint_mean={constraint_mean.detach().float().item():.8f}",
                f"grad_inner={grad_inner.detach().float().item():.8f}",
                f"grad_g_norm_sq={grad_g_norm_sq.detach().float().item():.8f}",
                f"lambda={dynamic_lambda.detach().float().item():.8f}",
            )

        return dynamic_lambda, grad_inner, grad_g_norm_sq

    def _compose_final_grads(self, grad_f, grad_g, lambda_value):
        final_grads = {}
        all_names = set(grad_f) | set(grad_g)
        for name in all_names:
            grad_pref = grad_f.get(name)
            grad_constraint = grad_g.get(name)
            if grad_pref is None and grad_constraint is None:
                continue
            if grad_pref is None:
                grad_pref = torch.zeros_like(grad_constraint)
            if grad_constraint is None:
                grad_constraint = torch.zeros_like(grad_pref)
            final_grads[name] = grad_pref + lambda_value.to(device=grad_pref.device, dtype=grad_pref.dtype) * grad_constraint
        return final_grads

    def rewrite_grads(self, params, final_grads):
        debug_rows = []
        for name, param in params.items():
            new_grad = final_grads.get(name)
            if new_grad is None:
                continue
            target_dtype = self._grad_buffer_dtypes.get(name, new_grad.dtype)
            new_grad = new_grad.to(dtype=target_dtype)
            safe_set_full_grad(param, new_grad)

            if self.grad_rewrite_debug and len(debug_rows) < 2:
                debug_rows.append(
                    {
                        "name": name,
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
        # Compute the KL surrogate in fp32. With bf16 + sum aggregation, policy/ref
        # log-probs can be large-magnitude negatives, and their small differences are
        # easily rounded to exactly zero before logging.
        policy_chosen_logps = policy_chosen_logps.float()
        policy_rejected_logps = policy_rejected_logps.float()
        ref_chosen_logps = ref_chosen_logps.float()
        ref_rejected_logps = ref_rejected_logps.float()

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
        del batch_state
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
        self._manual_micro_steps += 1
        boundary = self._manual_micro_steps % self.args.gradient_accumulation_steps == 0
        engine.set_gradient_accumulation_boundary(boundary)
        self._maybe_debug_step_state(
            "start",
            engine,
            boundary=boundary,
        )
        engine.zero_grad()
        engine.backward(preference_loss, retain_graph=True)
        _, grads_f_full = self.collect_full_grads(engine)
        _, grads_f_target = self.collect_full_grads(engine, target_names=target_names)
        self._maybe_debug_grad_collection("after_backward", target_names, grads_f_target, "preference")
        self._accumulate_grad_buffer(self._final_grad_buffer, grads_f_full)

        engine.zero_grad()
        engine.backward(constraint_value)
        _, grads_g_full = self.collect_full_grads(engine)
        _, grads_g_target = self.collect_full_grads(engine, target_names=target_names)
        self._maybe_debug_grad_collection("after_backward", target_names, grads_g_target, "constraint")

        lambda_value, grad_inner, grad_g_norm_sq = self._compute_dynamic_lambda_from_grads(
            grads_f=grads_f_target,
            grads_g=grads_g_target,
            constraint_value=constraint_value,
            device=preference_loss.device,
        )
        self._accumulate_scaled_grad_buffer(self._final_grad_buffer, grads_g_full, lambda_value)
        engine.zero_grad()

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

        if boundary:
            all_params = {
                name: param
                for name, param in engine.module.named_parameters()
                if param.requires_grad and param.numel() > 0
            }
            debug_rows = self.rewrite_grads(all_params, self._final_grad_buffer)

            if self.grad_rewrite_debug and self.state.global_step != self._last_grad_rewrite_step:
                self._last_grad_rewrite_step = self.state.global_step
                for row in debug_rows:
                    print(
                        "[grad_rewrite]",
                        f"step={self.state.global_step}",
                        f"name={row['name']}",
                        f"new_norm={row['new_norm']:.6f}",
                    )

            self._reset_grad_rewrite_state()

        self.store_metrics(metrics, train_eval="train")
        return loss.detach()

    def log(self, logs, start_time=None):
        append_seeking_log("dynamic_lambda_dpo", logs)
        return super().log(logs, start_time)
