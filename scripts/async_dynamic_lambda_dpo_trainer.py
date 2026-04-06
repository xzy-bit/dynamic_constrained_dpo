import torch
from accelerate.utils import DistributedType

from dynamic_lambda_dpo_trainer import DynamicLambdaDPOTrainer
from seeking_utils import append_seeking_log


class AsyncDynamicLambdaDPOTrainer(DynamicLambdaDPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_lambda = 0.0
        self._next_async_lambda = 0.0
        self._final_grad_f_buffer = {}
        self._final_grad_g_buffer = {}
        self._target_grad_f_buffer = {}
        self._target_grad_g_buffer = {}
        self._constraint_value_sum = None

    def _reset_async_state(self):
        self._final_grad_f_buffer = {}
        self._final_grad_g_buffer = {}
        self._target_grad_f_buffer = {}
        self._target_grad_g_buffer = {}
        self._constraint_value_sum = None

    def _accumulate_scalar(self, value: torch.Tensor):
        value = value.detach().float()
        if self._constraint_value_sum is None:
            self._constraint_value_sum = value.clone()
        else:
            self._constraint_value_sum = self._constraint_value_sum + value

    def _mean_constraint_value(self, device: torch.device) -> torch.Tensor:
        if self._constraint_value_sum is None:
            return torch.tensor(0.0, device=device)
        denom = max(int(self.args.gradient_accumulation_steps), 1)
        return self._constraint_value_sum / float(denom)

    def _compose_final_grads_async(self, grad_f, grad_g, lambda_value):
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
            final_grads[name] = grad_pref + lambda_value.to(
                device=grad_pref.device, dtype=grad_pref.dtype
            ) * grad_constraint
        return final_grads

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

        sigmoid_margin = torch.sigmoid(policy_margin)
        objective_accuracy = (policy_margin > 0).float()
        prefix = "train/"

        engine = self.deepspeed
        target_names = self._lambda_param_names(engine.module)
        self._manual_micro_steps += 1
        boundary = self._manual_micro_steps % self.args.gradient_accumulation_steps == 0
        engine.set_gradient_accumulation_boundary(boundary)
        self._maybe_debug_step_state("start_async", engine, boundary=boundary)

        engine.zero_grad()
        engine.backward(preference_loss, retain_graph=True)
        _, grads_f_full = self.collect_full_grads(engine)
        _, grads_f_target = self.collect_full_grads(engine, target_names=target_names)
        self._accumulate_grad_buffer(self._final_grad_f_buffer, grads_f_full)
        self._accumulate_grad_buffer(self._target_grad_f_buffer, grads_f_target)

        engine.zero_grad()
        engine.backward(constraint_value)
        _, grads_g_full = self.collect_full_grads(engine)
        _, grads_g_target = self.collect_full_grads(engine, target_names=target_names)
        self._accumulate_grad_buffer(self._final_grad_g_buffer, grads_g_full)
        self._accumulate_grad_buffer(self._target_grad_g_buffer, grads_g_target)
        self._accumulate_scalar(constraint_value)
        engine.zero_grad()

        applied_lambda = torch.tensor(
            self._async_lambda,
            device=preference_loss.device,
            dtype=torch.float32,
        )
        total_loss_for_log = preference_loss + applied_lambda.detach() * constraint_value

        grad_inner = torch.tensor(0.0, device=preference_loss.device)
        grad_g_norm_sq = torch.tensor(0.0, device=preference_loss.device)
        next_lambda = torch.tensor(self._async_lambda, device=preference_loss.device, dtype=torch.float32)

        if boundary:
            mean_constraint_value = self._mean_constraint_value(preference_loss.device)
            next_lambda, grad_inner, grad_g_norm_sq = self._compute_dynamic_lambda_from_grads(
                grads_f=self._target_grad_f_buffer,
                grads_g=self._target_grad_g_buffer,
                constraint_value=mean_constraint_value,
                device=preference_loss.device,
            )

            final_grads = self._compose_final_grads_async(
                self._final_grad_f_buffer,
                self._final_grad_g_buffer,
                applied_lambda.detach(),
            )
            all_params = {
                name: param
                for name, param in engine.module.named_parameters()
                if param.requires_grad and param.numel() > 0
            }
            debug_rows = self.rewrite_grads(all_params, final_grads)

            if self.grad_rewrite_debug and self.state.global_step != self._last_grad_rewrite_step:
                self._last_grad_rewrite_step = self.state.global_step
                for row in debug_rows:
                    print(
                        "[grad_rewrite_async]",
                        f"step={self.state.global_step}",
                        f"name={row['name']}",
                        f"new_norm={row['new_norm']:.6f}",
                    )

            self._next_async_lambda = float(next_lambda.detach().item())
            self._async_lambda = self._next_async_lambda
            self._reset_async_state()

        metrics = {
            f"{prefix}loss/preference": self.accelerator.gather_for_metrics(preference_loss.detach()).mean().item(),
            f"{prefix}loss/constraint": self.accelerator.gather_for_metrics(constraint_value.detach()).mean().item(),
            f"{prefix}loss/dynamic_lambda": self.accelerator.gather_for_metrics(applied_lambda.detach()).mean().item(),
            f"{prefix}loss/next_dynamic_lambda": self.accelerator.gather_for_metrics(next_lambda.detach()).mean().item(),
            f"{prefix}loss/total": self.accelerator.gather_for_metrics(total_loss_for_log.detach()).mean().item(),
            f"{prefix}grads/constraint_pref_inner": self.accelerator.gather_for_metrics(grad_inner.detach()).mean().item(),
            f"{prefix}grads/constraint_grad_norm_sq": self.accelerator.gather_for_metrics(grad_g_norm_sq.detach()).mean().item(),
            f"{prefix}/accuracies": self.accelerator.gather_for_metrics(objective_accuracy.detach()).mean().item(),
            f"{prefix}/margins": self.accelerator.gather_for_metrics(policy_margin.detach()).mean().item(),
            f"{prefix}/sigmoid_margins": self.accelerator.gather_for_metrics(sigmoid_margin.detach()).mean().item(),
        }

        self.store_metrics(metrics, train_eval="train")
        return preference_loss.detach()

    def log(self, logs, start_time=None):
        append_seeking_log("async_dynamic_lambda_dpo", logs)
        return super().log(logs, start_time)
