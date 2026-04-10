from dataclasses import dataclass, field
from typing import Literal, Optional

from alignment import DPOConfig


@dataclass
class DynamicLambdaDPOConfig(DPOConfig):
    offline_ref_logps_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset directory containing precomputed ref_chosen_logps/ref_rejected_logps."},
    )
    dlambda_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha coefficient in lambda = max((alpha * g - <grad_g, grad_f>) / ||grad_g||^2, 0)."},
    )
    dlambda_epsilon: float = field(
        default=0.0,
        metadata={"help": "Target KL budget epsilon used in the surrogate constraint g_hat - epsilon."},
    )
    dlambda_lambda_max: Optional[float] = field(
        default=20.0,
        metadata={"help": "Optional upper clip for the dynamic lambda. Set to null to disable clipping."},
    )
    dlambda_grad_target: Literal[
        "all",
        "last_three_layers",
        "last_two_layers",
        "last_layer",
        "last_layer_down_proj",
    ] = field(
        default="last_two_layers",
        metadata={"help": "Which trainable parameters to use when estimating grad_F and grad_g for lambda."},
    )
    dlambda_reference_free: bool = field(
        default=False,
        metadata={"help": "If true, treat the reference log-probs as zeros when forming the KL surrogate."},
    )
    dlambda_logp_aggregation: Literal["sum", "mean"] = field(
        default="sum",
        metadata={"help": "How to aggregate response log-probs: sum over tokens or mean over response length."},
    )
    dlambda_kl_normalize_by_length: bool = field(
        default=True,
        metadata={
            "help": "If true, divide each chosen/rejected logp difference by its sequence length when "
            "forming the KL surrogate. Mirrors dpo_opt's _compute_kl_constraint per-token normalization."
        },
    )
    dlambda_barrier_mu: float = field(
        default=0.0,
        metadata={
            "help": "Optional dynamic-barrier coefficient mu for 0.5 * mu * relu(g)^2 added to the "
            "stage-2 loss. Mirrors dpo_opt's barrier_mu."
        },
    )
    dlambda_apply_beta_to_preference: bool = field(
        default=True,
        metadata={
            "help": "If true, scale the pairwise sigmoid argument by self.beta to match dpo_opt's "
            "_compute_pairwise_sigmoid_loss. Set false to keep the original (beta-free) behavior."
        },
    )
