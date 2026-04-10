"""
HyPO DPO Trainer — thin subclass of DPOMetricsTrainer.

Implements the HyPO (Hypothetical Preference Optimization) loss from
https://github.com/tmllab/2026_ICLR_HyPO.  The only modification to standard
DPO is the treatment of the reference margin:

    ref_prime = gamma + tau * softplus((ref_logratios - gamma) / tau)   [soft]
    ref_prime = max(ref_logratios, gamma)                                [hard]
    logits    = pi_logratios - ref_prime

Config knobs (passed through DPOConfig / yaml):
    im_enable (bool, default True):  Enable HyPO; False = standard DPO.
    im_gamma  (float, default 0.0):  Clipping threshold for the reference margin.
    im_tau    (float, default 0.0):  Softplus temperature; <= 0 uses hard clamp.
"""

import torch
import torch.nn.functional as F

from dpo_metrics_trainer import DPOMetricsTrainer
from seeking_utils import append_seeking_log


class HypoDPOTrainer(DPOMetricsTrainer):

    def __init__(
        self,
        *args,
        im_enable: bool = True,
        im_gamma: float = 0.0,
        im_tau: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.im_enable = im_enable
        self.im_gamma = im_gamma
        self.im_tau = im_tau

    # ------------------------------------------------------------------
    # Override the DPO loss to inject the HyPO reference-margin clamp.
    # Signature must match the base TRL DPOTrainer.dpo_loss.
    # ------------------------------------------------------------------
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        if self.ref_model is None and not hasattr(self, "null_ref_context"):
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)

        if self.im_enable:
            gamma = torch.as_tensor(self.im_gamma, dtype=ref_logratios.dtype, device=ref_logratios.device)
            if self.im_tau > 0:
                tau = float(self.im_tau)
                ref_prime = gamma + tau * F.softplus((ref_logratios - gamma) / tau)
            else:
                ref_prime = torch.maximum(ref_logratios, gamma)
            logits = pi_logratios - ref_prime
        else:
            logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def log(self, logs, start_time=None):
        append_seeking_log("hypo_dpo", logs)
        return super().log(logs, start_time)
