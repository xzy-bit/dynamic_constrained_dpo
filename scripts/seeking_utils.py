import json
import os
from datetime import datetime

import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def append_seeking_log(name: str, logs: dict) -> None:
    seeking_dir = os.path.join(get_repo_root(), "seeking")
    os.makedirs(seeking_dir, exist_ok=True)
    path = os.path.join(seeking_dir, f"{name}.jsonl")

    record = dict(logs)
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if wandb is not None and wandb.run is not None:
        step = record.get("step")
        payload = {}
        for key, value in record.items():
            if key in {"timestamp", "step"}:
                continue
            if isinstance(value, (int, float, bool)):
                payload[f"seeking/{name}/{key}"] = value
        if payload:
            if isinstance(step, int):
                wandb.log(payload, step=step)
            else:
                wandb.log(payload)


def flatten_grad(grad: torch.Tensor | None) -> torch.Tensor | None:
    if grad is None:
        return None
    return grad.reshape(-1)


def _zero_on_device(device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(0.0, device=device)


def grad_norm(grad: torch.Tensor | None) -> torch.Tensor:
    if grad is None:
        return _zero_on_device()
    return grad.norm()


def combine_grads(grad_a: torch.Tensor | None, grad_b: torch.Tensor | None) -> torch.Tensor | None:
    if grad_a is None and grad_b is None:
        return None
    if grad_a is None:
        return grad_b
    if grad_b is None:
        return grad_a
    return grad_a + grad_b


def grad_cosine(grad_a: torch.Tensor | None, grad_b: torch.Tensor | None) -> torch.Tensor:
    vec_a = flatten_grad(grad_a)
    vec_b = flatten_grad(grad_b)

    device = None
    for tensor in (vec_a, vec_b):
        if tensor is not None:
            device = tensor.device
            break

    if vec_a is None or vec_b is None:
        return _zero_on_device(device)

    denom = vec_a.norm() * vec_b.norm()
    if denom.item() == 0:
        return _zero_on_device(vec_a.device)
    return torch.dot(vec_a, vec_b) / denom


def flatten_grad_list(grads) -> torch.Tensor | None:
    if grads is None:
        return None
    flat_parts = [grad.reshape(-1) for grad in grads if grad is not None]
    if not flat_parts:
        return None
    return torch.cat(flat_parts)


def grad_list_norm(grads) -> torch.Tensor:
    flat = flatten_grad_list(grads)
    if flat is None:
        return _zero_on_device()
    return flat.norm()


def grad_list_inner(grads_a, grads_b) -> torch.Tensor:
    flat_a = flatten_grad_list(grads_a)
    flat_b = flatten_grad_list(grads_b)

    device = None
    for tensor in (flat_a, flat_b):
        if tensor is not None:
            device = tensor.device
            break

    if flat_a is None or flat_b is None:
        return _zero_on_device(device)

    return torch.dot(flat_a, flat_b)


def get_last_layer_weight(model):
    output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(output_embeddings):
        layer = output_embeddings()
        if layer is not None and hasattr(layer, "weight"):
            return layer.weight

    for _, param in reversed(list(model.named_parameters())):
        if param.requires_grad:
            return param
    return None


def grad_mean(grad: torch.Tensor | None) -> torch.Tensor:
    if grad is None:
        return _zero_on_device()
    return grad.mean()


def grad_abs_mean(grad: torch.Tensor | None) -> torch.Tensor:
    if grad is None:
        return _zero_on_device()
    return grad.abs().mean()


def cosine_with_total(
    part_grad: torch.Tensor | None,
    other_grad: torch.Tensor | None,
    total_grad_part: torch.Tensor | None,
    total_grad_other: torch.Tensor | None,
) -> torch.Tensor:
    part_vec = flatten_grad(part_grad)
    other_vec = flatten_grad(other_grad)
    total_part_vec = flatten_grad(total_grad_part)
    total_other_vec = flatten_grad(total_grad_other)

    device = None
    for tensor in (part_vec, other_vec, total_part_vec, total_other_vec):
        if tensor is not None:
            device = tensor.device
            break

    if part_vec is None or total_part_vec is None:
        return _zero_on_device(device)

    if other_vec is None:
        other_vec = torch.zeros(0, device=part_vec.device)
    if total_other_vec is None:
        total_other_vec = torch.zeros(0, device=part_vec.device)

    numerator = torch.dot(part_vec, total_part_vec)
    part_norm = part_vec.norm()
    total_norm_sq = total_part_vec.pow(2).sum()
    if total_other_vec.numel() > 0:
        total_norm_sq = total_norm_sq + total_other_vec.pow(2).sum()
    total_norm = torch.sqrt(total_norm_sq)
    denom = part_norm * total_norm
    if denom.item() == 0:
        return _zero_on_device(part_vec.device)
    return numerator / denom
