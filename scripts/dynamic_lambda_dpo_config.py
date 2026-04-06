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
        default=10.0,
        metadata={"help": "Optional upper clip for the dynamic lambda. Set to null to disable clipping."},
    )
    dlambda_grad_target: Literal["all", "last_three_layers"] = field(
        default="last_three_layers",
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
