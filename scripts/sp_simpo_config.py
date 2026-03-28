from dataclasses import dataclass, field

from simpo_config import SimPOConfig


@dataclass
class SPSimPOConfig(SimPOConfig):
    sp_alpha: float = field(
        default=1.5,
        metadata={"help": "Alpha-like hyperparameter used by the sparse preference score."},
    )
    sp_beta: float = field(
        default=0.2,
        metadata={"help": "Auxiliary beta-like hyperparameter used by the sparse preference score."},
    )
    sp_temperature: float = field(
        default=2.0,
        metadata={"help": "Divisor applied to logits before sparsemax in SP-SimPO."},
    )
    sp_neg_support_coef: float = field(
        default=1.1,
        metadata={"help": "Coefficient applied to negative tokens whose gold token remains inside the sparsemax support set."},
    )
