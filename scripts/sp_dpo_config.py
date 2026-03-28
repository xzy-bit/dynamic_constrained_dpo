from dataclasses import dataclass, field
from typing import Literal

from alignment import DPOConfig


@dataclass
class SPDPOConfig(DPOConfig):
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
        metadata={"help": "Divisor applied to logits before sparsemax in SP-DPO."},
    )
    sp_neg_support_coef: float = field(
        default=1.1,
        metadata={"help": "Coefficient applied on negative tokens whose gold token remains in the sparsemax support."},
    )
    sp_loss_type: Literal["dpo", "simpo", "hypo"] = field(
        default="dpo",
        metadata={"help": "Which FY-score preference loss to use: DPO-like, SimPO-like, or HyPO-like."},
    )
    sp_margin: float = field(
        default=0.0,
        metadata={"help": "Constant margin subtracted inside the FY-score loss."},
    )
    sp_average_score: bool = field(
        default=False,
        metadata={"help": "Whether to length-normalize the FY sequence score like SimPO."},
    )
