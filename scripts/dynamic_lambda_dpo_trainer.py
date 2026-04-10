import os
from collections.abc import Mapping
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from accelerate.utils import DistributedType
from deepspeed.utils import safe_get_full_grad

from dpo_metrics_trainer import DPOMetricsTrainer
from seeking_utils import append_seeking_log


class DynamicLambdaDPOTrainer(DPOMetricsTrainer):
    def __init__(
        self,
        *args,
        dlambda_alpha: float = 1.0,
        dlambda_epsilon: float = 0.0,
        dlambda_lambda_max: float | None = 20.0,
        dlambda_grad_target: str = "last_two_layers",
        dlambda_reference_free: bool = False,
        dlambda_logp_aggregation: str = "sum",
        dlambda_kl_normalize_by_length: bool = True,
        dlambda_barrier_mu: float = 0.0,
        dlambda_apply_beta_to_preference: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dlambda_alpha = dlambda_alpha
        self.dlambda_epsilon = dlambda_epsilon
        self.dlambda_lambda_max = dlambda_lambda_max
        self.dlambda_grad_target = dlambda_grad_target
        self.dlambda_reference_free = dlambda_reference_free
        self.dlambda_logp_aggregation = dlambda_logp_aggregation
        self.dlambda_kl_normalize_by_length = dlambda_kl_normalize_by_length
        self.dlambda_barrier_mu = dlambda_barrier_mu
        self.dlambda_apply_beta_to_preference = dlambda_apply_beta_to_preference

        self.grad_collection_debug = os.getenv("DYNAMIC_LAMBDA_GRAD_DEBUG", "0") == "1"
        self._manual_micro_steps = 0
        self._cached_micro_batches = []
        self._tracked_param_name = None

    def _trainable_grad_params(self, model):
        return [(name, param) for name, param in model.named_parameters() if param.requires_grad and param.numel() > 0]

    def _resolve_transformer_layers(self, model):
        candidate_paths = [
            ("model", "layers"),
            ("model", "model", "layers"),
            ("base_model", "model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
        ]
        for path in candidate_paths:
            current = model
            ok = True
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    ok = False
                    break
            if ok:
                return current
        return None

    def _lambda_param_names(self, model):
        if self.dlambda_grad_target == "all":
            return [name for name, _ in self._trainable_grad_params(model)]

        layers = self._resolve_transformer_layers(model)
        if layers is None:
            return [name for name, _ in self._trainable_grad_params(model)]

        # Number of trailing layers used as the proxy subset for the gradient projection.
        # Mirrors dpo_opt's `proxy_projection_layer_mode` (which defaults to "last_layer").
        if self.dlambda_grad_target == "last_layer":
            n_tail = 1
        elif self.dlambda_grad_target == "last_two_layers":
            n_tail = 2
        elif self.dlambda_grad_target == "last_three_layers":
            n_tail = 3
        elif self.dlambda_grad_target == "last_layer_down_proj":
            # Mirrors dpo_opt: only the MLP down_proj of the final transformer block.
            last_layer = layers[-1]
            down_proj = getattr(getattr(last_layer, "mlp", None), "down_proj", None)
            if down_proj is None:
                raise ValueError(
                    "dlambda_grad_target=last_layer_down_proj requires the final transformer "
                    "block to expose `.mlp.down_proj` (Llama/Qwen-style)."
                )
            target_param_ids = {id(p) for p in down_proj.parameters() if p.requires_grad}
            names = [name for name, p in model.named_parameters() if id(p) in target_param_ids]
            if not names:
                return [name for name, _ in self._trainable_grad_params(model)]
            return names
        else:
            raise ValueError(
                f"Unknown dlambda_grad_target={self.dlambda_grad_target!r}. "
                "Expected one of: all, last_layer, last_two_layers, last_three_layers, "
                "last_layer_down_proj."
            )

        tail_layers = list(layers[-n_tail:])
        target_names = []
        for layer in tail_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    target_names.append(name)

        if not target_names:
            return [name for name, _ in self._trainable_grad_params(model)]

        layer_param_ids = {id(param) for layer in tail_layers for param in layer.parameters() if param.requires_grad}
        return [name for name, param in model.named_parameters() if id(param) in layer_param_ids]

    def _lambda_params(self, model):
        target_names = set(self._lambda_param_names(model))
        return [param for name, param in model.named_parameters() if name in target_names]

    def _get_tracked_param_name(self, model):
        if self._tracked_param_name is not None:
            return self._tracked_param_name
        target_names = self._lambda_param_names(model)
        self._tracked_param_name = target_names[0] if target_names else None
        return self._tracked_param_name

    def _debug_step_param_update(self, model, before_step_snapshot):
        if not self.grad_collection_debug or before_step_snapshot is None:
            return
        param_name = self._get_tracked_param_name(model)
        if param_name is None:
            return
        for name, param in model.named_parameters():
            if name != param_name:
                continue
            after_step = param.detach().float().cpu()
            mean_abs = (after_step - before_step_snapshot).abs().mean().item()
            max_abs = (after_step - before_step_snapshot).abs().max().item()
            rank = getattr(self.accelerator, "process_index", 0)
            print(
                f"[dlambda_param_update] rank={rank} param={param_name} "
                f"mean_abs={mean_abs:.12e} max_abs={max_abs:.12e}",
                flush=True,
            )
            return

    @contextmanager
    def _freeze_except(self, model, target_names):
        target_names = set(target_names)
        original_flags = {name: param.requires_grad for name, param in model.named_parameters()}
        try:
            for name, param in model.named_parameters():
                param.requires_grad = name in target_names
            yield
        finally:
            for name, param in model.named_parameters():
                param.requires_grad = original_flags[name]

    def _clone_to_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().clone()
        if isinstance(obj, Mapping):
            return {key: self._clone_to_cpu(value) for key, value in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._clone_to_cpu(value) for value in obj)
        if isinstance(obj, list):
            return [self._clone_to_cpu(value) for value in obj]
        return obj

    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, Mapping):
            return {key: self._move_to_device(value, device) for key, value in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(value, device) for value in obj)
        if isinstance(obj, list):
            return [self._move_to_device(value, device) for value in obj]
        return obj

    def _collect_full_grads(self, engine, target_names=None):
        target_names = set(target_names) if target_names is not None else None
        grads = {}
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue
            if target_names is not None and name not in target_names:
                continue
            grad = safe_get_full_grad(param)
            if grad is None:
                continue
            grads[name] = grad.detach()
        return grads

    def _add_grad_dicts(self, acc, new):
        if acc is None:
            return {name: grad.clone() for name, grad in new.items()}
        for name, grad in new.items():
            if name in acc:
                acc[name] = acc[name] + grad
            else:
                acc[name] = grad.clone()
        return acc

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
            return torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
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
            return torch.tensor(0.0, device=result_device) if result_device is not None else torch.tensor(0.0)
        return torch.dot(flat_a, flat_b)

    def _maybe_mean_normalize(self, logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        # Used only for offline-precomputed ref logps which are stored as sum-over-tokens
        # ([B] shape). When the trainer is in "mean" aggregation mode we need to divide
        # by token count here so that ref and policy logps live on the same per-token scale.
        if self.dlambda_logp_aggregation != "mean":
            return logps
        token_count = loss_mask.sum(dim=-1).clamp_min(1).to(logps.dtype)
        return logps / token_count

    def _constraint_value(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        chosen_loss_mask: torch.Tensor | None = None,
        rejected_loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_chosen_logps = policy_chosen_logps.float()
        policy_rejected_logps = policy_rejected_logps.float()
        ref_chosen_logps = ref_chosen_logps.float()
        ref_rejected_logps = ref_rejected_logps.float()

        chosen_kl = policy_chosen_logps - ref_chosen_logps
        rejected_kl = policy_rejected_logps - ref_rejected_logps

        # Mirror dpo_opt's _compute_kl_constraint per-token normalization (chosen_kl / chosen_lengths).
        # When dlambda_logp_aggregation == "sum" the logps are sum-over-tokens; dividing by length
        # converts the surrogate to per-token-mean log-ratio. When aggregation is already "mean"
        # this normalization is a no-op (so we skip it).
        if (
            self.dlambda_kl_normalize_by_length
            and self.dlambda_logp_aggregation == "sum"
            and chosen_loss_mask is not None
            and rejected_loss_mask is not None
        ):
            chosen_lengths = chosen_loss_mask.sum(dim=-1).clamp_min(1).to(chosen_kl.dtype)
            rejected_lengths = rejected_loss_mask.sum(dim=-1).clamp_min(1).to(rejected_kl.dtype)
            chosen_kl = chosen_kl / chosen_lengths
            rejected_kl = rejected_kl / rejected_lengths

        surrogate_kl = 0.5 * (chosen_kl.mean() + rejected_kl.mean())
        return surrogate_kl - self.dlambda_epsilon

    def _compute_batch_state(self, model, batch):
        model_output = self.concatenated_forward(model, batch)
        chosen_loss_mask = model_output["chosen_loss_mask"]
        rejected_loss_mask = model_output["rejected_loss_mask"]
        # concatenated_forward already aggregated per-token logps to [B] using
        # the base-class _aggregate_logps, which reads self.dlambda_logp_aggregation.
        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]

        if self.dlambda_reference_free:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        elif "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            # Offline-precomputed ref logps are stored as sum-over-tokens; convert
            # to per-token mean if the trainer is in mean mode.
            ref_chosen_logps = self._maybe_mean_normalize(batch["ref_chosen_logps"], chosen_loss_mask)
            ref_rejected_logps = self._maybe_mean_normalize(batch["ref_rejected_logps"], rejected_loss_mask)
        else:
            # compute_ref_log_probs goes through concatenated_forward, so it already
            # respects dlambda_logp_aggregation — no further normalization needed.
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        # Mirror dpo_opt: scale the sigmoid argument by beta. When dlambda_apply_beta_to_preference
        # is False the trainer keeps the original beta-free behavior.
        policy_margin = chosen_logps - rejected_logps
        if self.dlambda_apply_beta_to_preference:
            preference_loss = (-F.logsigmoid(self.beta * policy_margin)).mean()
        else:
            preference_loss = (-F.logsigmoid(policy_margin)).mean()
        constraint_value = self._constraint_value(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_loss_mask=chosen_loss_mask,
            rejected_loss_mask=rejected_loss_mask,
        )

        del model_output
        batch_examples = int(chosen_logps.shape[0])
        return {
            "preference_loss": preference_loss,
            "constraint_value": constraint_value,
            "policy_margin": policy_margin,
            "batch_examples": batch_examples,
        }

    def _compute_minibatch_lambda(self, engine, device):
        target_names = self._lambda_param_names(engine.module)
        total_examples = sum(int(batch["prompt_input_ids"].shape[0]) for batch in self._cached_micro_batches)
        total_examples = max(total_examples, 1)

        grad_f_acc = None
        grad_g_acc = None
        constraint_acc = torch.tensor(0.0, device=device)

        with self._freeze_except(engine.module, target_names):
            for cached_batch in self._cached_micro_batches:
                batch = self._move_to_device(cached_batch, device)
                batch_state = self._compute_batch_state(engine.module, batch)
                batch_weight = batch_state["batch_examples"] / total_examples

                engine.zero_grad()
                engine.set_gradient_accumulation_boundary(False)
                engine.backward(batch_state["preference_loss"] * batch_weight, retain_graph=True)
                grads_f = self._collect_full_grads(engine, target_names=target_names)

                engine.zero_grad()
                engine.set_gradient_accumulation_boundary(False)
                engine.backward(batch_state["constraint_value"] * batch_weight)
                grads_g = self._collect_full_grads(engine, target_names=target_names)
                engine.zero_grad()

                grad_f_acc = self._add_grad_dicts(grad_f_acc, grads_f)
                grad_g_acc = self._add_grad_dicts(grad_g_acc, grads_g)
                constraint_acc = constraint_acc + batch_state["constraint_value"].detach() * batch_weight

        grad_target_names = self._lambda_param_names(engine.module)
        grad_f_list = [grad_f_acc.get(name) if grad_f_acc is not None else None for name in grad_target_names]
        grad_g_list = [grad_g_acc.get(name) if grad_g_acc is not None else None for name in grad_target_names]

        grad_inner = self._grad_list_inner(grad_g_list, grad_f_list, device=device)
        grad_g_norm_sq = self._grad_list_norm(grad_g_list, device=device).pow(2)
        grad_f_norm_sq = self._grad_list_norm(grad_f_list, device=device).pow(2)

        numerator = self.dlambda_alpha * constraint_acc.detach().float() - grad_inner.detach()
        lambda_value = torch.clamp(numerator / grad_g_norm_sq.clamp_min(1e-12), min=0.0)
        if self.dlambda_lambda_max is not None:
            lambda_value = torch.clamp(lambda_value, max=self.dlambda_lambda_max)

        return (
            lambda_value.detach(),
            constraint_acc.detach(),
            grad_inner.detach(),
            grad_g_norm_sq.detach(),
            grad_f_norm_sq.detach(),
        )

    def _second_stage_backward(self, engine, lambda_value, device):
        total_examples = sum(int(batch["prompt_input_ids"].shape[0]) for batch in self._cached_micro_batches)
        total_examples = max(total_examples, 1)

        total_pref = torch.tensor(0.0, device=device)
        total_constraint = torch.tensor(0.0, device=device)
        total_margin = torch.tensor(0.0, device=device)
        total_accuracy = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)

        engine.zero_grad()
        for idx, cached_batch in enumerate(self._cached_micro_batches):
            batch = self._move_to_device(cached_batch, device)
            batch_state = self._compute_batch_state(engine.module, batch)
            batch_weight = batch_state["batch_examples"] / total_examples
            # Mirror dpo_opt: loss = preference + lambda * violation + 0.5 * mu * relu(violation)^2
            constraint_value = batch_state["constraint_value"]
            barrier_term = 0.5 * self.dlambda_barrier_mu * F.relu(constraint_value).pow(2)
            batch_loss = batch_state["preference_loss"] + lambda_value * constraint_value + barrier_term

            is_last = idx == len(self._cached_micro_batches) - 1
            engine.set_gradient_accumulation_boundary(is_last)
            engine.backward(batch_loss * batch_weight)

            total_pref = total_pref + batch_state["preference_loss"].detach() * batch_weight
            total_constraint = total_constraint + batch_state["constraint_value"].detach() * batch_weight
            total_margin = total_margin + batch_state["policy_margin"].detach().mean() * batch_weight
            total_accuracy = total_accuracy + (batch_state["policy_margin"] > 0).float().detach().mean() * batch_weight
            total_loss = total_loss + batch_loss.detach() * batch_weight

        return {
            "loss": total_loss.detach(),
            "preference_loss": total_pref.detach(),
            "constraint_value": total_constraint.detach(),
            "policy_margin": total_margin.detach(),
            "objective_accuracy": total_accuracy.detach(),
        }

    def _reset_cached_batches(self):
        self._cached_micro_batches = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        prepared_inputs = self._prepare_inputs(inputs)
        self._cached_micro_batches.append(self._clone_to_cpu(prepared_inputs))

        engine = self.deepspeed
        self._manual_micro_steps += 1
        boundary = self._manual_micro_steps % self.args.gradient_accumulation_steps == 0

        if not boundary:
            return torch.tensor(0.0, device=self.accelerator.device)

        (
            lambda_value,
            mini_constraint,
            grad_inner,
            grad_g_norm_sq,
            grad_f_norm_sq,
        ) = self._compute_minibatch_lambda(engine, self.accelerator.device)
        # Cosine between ∇F (preference) and ∇g (constraint) on the proxy subset.
        # Recorded BEFORE the optimizer step (engine.step is called below) so it
        # reflects the geometry that lambda is actually computed from. Cosine ≈ 0
        # means the two objectives are locally orthogonal on this proxy.
        denom = (grad_f_norm_sq.clamp_min(1e-12) * grad_g_norm_sq.clamp_min(1e-12)).sqrt()
        grad_cosine = grad_inner / denom
        second_stage = self._second_stage_backward(engine, lambda_value, self.accelerator.device)
        before_step_snapshot = None
        if self.grad_collection_debug:
            tracked_name = self._get_tracked_param_name(model)
            if tracked_name is not None:
                for name, param in model.named_parameters():
                    if name == tracked_name:
                        before_step_snapshot = param.detach().float().cpu().clone()
                        break
        engine.step()
        self._debug_step_param_update(model, before_step_snapshot)
        self._reset_cached_batches()

        metrics = {
            "train/loss/preference": self.accelerator.gather_for_metrics(second_stage["preference_loss"]).mean().item(),
            "train/loss/constraint": self.accelerator.gather_for_metrics(second_stage["constraint_value"]).mean().item(),
            "train/loss/dynamic_lambda": self.accelerator.gather_for_metrics(lambda_value).mean().item(),
            "train/loss/total": self.accelerator.gather_for_metrics(second_stage["loss"]).mean().item(),
            "train/grads/constraint_pref_inner": self.accelerator.gather_for_metrics(grad_inner).mean().item(),
            "train/grads/constraint_grad_norm_sq": self.accelerator.gather_for_metrics(grad_g_norm_sq).mean().item(),
            "train/grads/preference_grad_norm_sq": self.accelerator.gather_for_metrics(grad_f_norm_sq).mean().item(),
            "train/grads/cosine_F_g": self.accelerator.gather_for_metrics(grad_cosine).mean().item(),
            "train/constraint/minibatch": self.accelerator.gather_for_metrics(mini_constraint).mean().item(),
            "train/accuracies": self.accelerator.gather_for_metrics(second_stage["objective_accuracy"]).mean().item(),
            "train/margins": self.accelerator.gather_for_metrics(second_stage["policy_margin"]).mean().item(),
            "train/sigmoid_margins": (
                self.accelerator.gather_for_metrics(torch.sigmoid(second_stage["policy_margin"])).mean().item()
            ),
        }
        self.store_metrics(metrics, train_eval="train")
        return second_stage["loss"]

    def log(self, logs, start_time=None):
        append_seeking_log("dynamic_lambda_dpo", logs)
        return super().log(logs, start_time)
