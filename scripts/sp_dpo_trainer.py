import torch
import torch.nn.functional as F
from entmax import sparsemax_loss, sparsemax, entmax_bisect_loss,entmax15,entmax15_loss,entmax_bisect
from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn as nn
from trl.trainer.utils import pad_to_length

IGNORE_INDEX = -100

def flush_left(attention_mask: torch.Tensor,
               input_ids: torch.Tensor,
               loss_mask: torch.Tensor):
    B, T = attention_mask.shape
    new_attention_mask = torch.zeros_like(attention_mask)
    new_input_ids = torch.zeros_like(input_ids)
    new_loss_mask = torch.zeros_like(loss_mask)

    for i in range(B):
        valid_idx = attention_mask[i].bool()
        n = valid_idx.sum().item()
        if n > 0:
            new_attention_mask[i, :n] = attention_mask[i, valid_idx]
            new_input_ids[i, :n] = input_ids[i, valid_idx]
            new_loss_mask[i, :n] = loss_mask[i, valid_idx]

    return new_attention_mask, new_input_ids, new_loss_mask

def _get_batch_logps(logits, index):
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)   # [B, T, 1]

    loss_mask = index != -100
    safe_labels = index.masked_fill(~loss_mask, 0)

    per_token_logps = []
    for row_logits, row_labels in zip(logits, safe_labels, strict=True):
        row_logps = F.log_softmax(row_logits, dim=-1)   # [T, V]
        row_per_token_logps = row_logps.gather(
            dim=-1, index=row_labels
        )                                               # [T, 1]
        per_token_logps.append(row_per_token_logps)

    per_token_logps = torch.stack(per_token_logps, dim=0)   # [B, T, 1]

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)       # [B, T]

    per_token_logps = per_token_logps.masked_fill(~loss_mask.squeeze(-1) if squeeze else ~loss_mask, 0.0)

    return per_token_logps.sum(-1)

def _get_rejection_penalty(logits, labels, eps=1e-5):
    B, M, V = logits.shape
    mask = (labels != -100)

    active_logits = logits[mask]   # [num_active, V]
    active_labels = labels[mask]   # [num_active]

    if active_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    probs = F.softmax(active_logits,dim=-1)
    p_target = probs.gather(-1, active_labels.unsqueeze(-1)).squeeze(-1)  # [num_active]

    # 计算 -log(1-p)
    penalty_per_token = -torch.log(1 - p_target + eps)  # [num_active]

    token_penalty = torch.zeros(B, M, device=logits.device, dtype=penalty_per_token.dtype)
    token_penalty[mask] = penalty_per_token
    return token_penalty.sum(-1)

def _get_batch_ent_score(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    alpha: float = 1.5,
    beta: float = 0.5,
    ispos: bool = False,
):
    """
    Compute sequence-level Fenchel–Young (entmax/sparsemax-family) scores.
    Returns: scores of shape (B,)
    """
    B, M, V = logits.shape

    # shift like NLL
    mask = (labels != -100)
    safe_labels = labels.masked_fill(~mask, 0)

    flat_logits = logits.view(-1, V).float()
    flat_labels = safe_labels.view(-1)

    # token-level entmax loss
    flat_loss = entmax15_loss(flat_logits,flat_labels)
    #flat_loss = entmax_bisect_loss(flat_logits, flat_labels, alpha, n_iter=50)  # [B*(M-1)]
    token_loss = flat_loss.view(B, M)
    #print("=====using entmax ========")
    '''
    if ispos:
       if alpha == 1.5:
           entmax_probs = entmax15(flat_logits, dim=-1)
       else:
           entmax_probs = entmax_bisect(flat_logits, alpha=alpha, dim=-1, n_iter=50)

       softmax_probs = F.softmax(flat_logits, dim=-1)

       one_hot = F.one_hot(flat_labels, num_classes=softmax_probs.size(-1)).bool()
       tail_mask = (entmax_probs == 0.0) & (~one_hot)

       suppressed_mass = (softmax_probs * tail_mask.float()).sum(dim=-1)
       suppressed_mass = torch.clamp(suppressed_mass, max=0.99)

       ns_loss = -torch.log(1.0 - suppressed_mass)              # [B*(M-1)]
       ns_loss = ns_loss.view(B, M - 1)

       ns_loss = beta * ns_loss * mask
       token_loss = token_loss + ns_loss - ns_loss.detach()
    '''

    token_loss = token_loss * mask.float()

    #per_token_logps = _get_batch_logps(logits,safe_labels)
    #per_token_logps = per_token_logps * mask
    
    per_token_logps = torch.zeros_like(token_loss)

    return -token_loss.sum(-1), per_token_logps.sum(-1)

def _get_batch_sp_score(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    alpha: float = 1.5,
    beta: float = 0.5,
    ispos: bool = False,
):
    """
    Compute sequence-level Fenchel–Young (entmax/sparsemax-family) scores.
    Returns: scores of shape (B,)
    """
    B, M, V = logits.shape

    mask = (labels != -100)
    safe_labels = labels.masked_fill(~mask, 0)

    flat_logits = logits.view(-1, V)
    flat_labels = safe_labels.view(-1)

    # token-level entmax loss
    flat_loss = sparsemax_loss(flat_logits, flat_labels)  # [B*(M-1)]
    token_loss = flat_loss.view(B, M)

    '''
    if ispos:
       if alpha == 1.5:
           entmax_probs = entmax15(flat_logits, dim=-1)
       else:
           entmax_probs = entmax_bisect(flat_logits, alpha=alpha, dim=-1, n_iter=50)

       softmax_probs = F.softmax(flat_logits, dim=-1)

       one_hot = F.one_hot(flat_labels, num_classes=softmax_probs.size(-1)).bool()
       tail_mask = (entmax_probs == 0.0) & (~one_hot)

       suppressed_mass = (softmax_probs * tail_mask.float()).sum(dim=-1)
       suppressed_mass = torch.clamp(suppressed_mass, max=0.99)

       ns_loss = -torch.log(1.0 - suppressed_mass)              # [B*(M-1)]
       ns_loss = ns_loss.view(B, M - 1)

       ns_loss = beta * ns_loss * mask
       token_loss = token_loss + ns_loss - ns_loss.detach()
    '''

    token_loss = token_loss * mask.float()
    
    #per_token_logps = _get_batch_logps(logits,safe_labels)
    #per_token_logps = per_token_logps * mask

    per_token_logps = torch.zeros_like(token_loss)

    return -token_loss.sum(-1), per_token_logps.sum(-1)


class SPDPOTrainer(DPOTrainer):
    def __init__(
        self,
        *args,
        sp_alpha: float = 1.5,
        sp_beta: float = 0.2,
        reference_free: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.sp_alpha = sp_alpha
        self.sp_beta = sp_beta
        self.reference_free = reference_free

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        if self.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        #ref_logratios = torch.clamp(ref_logratios, min=0.0)

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        #loss = -F.logsigmoid(self.beta * logits) + self.beta * self.rejected_tail

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
   
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def concatenated_forward(self, model, batch):
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch,
            padding_value=self.padding_value,
        )

        prompt_input_ids = concatenated_batch["prompt_input_ids"]                  # [2B, Tp]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]        # [2B, Tp]
        completion_input_ids = concatenated_batch["completion_input_ids"]          # [2B, Tc]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]# [2B, Tc]
    
        # 1) 拼接 prompt + completion
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)        # [2B, T]
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)

        # 2) loss mask: prompt 不参与，completion 参与
        loss_mask = torch.cat(
            [torch.zeros_like(prompt_attention_mask), completion_attention_mask],
            dim=1,
        )  # [2B, T]

        # 3) 左对齐
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
    
        # 4) 截断
        if getattr(self, "max_length", None) is not None:
            if self.truncation_mode == "keep_end":
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                loss_mask = loss_mask[:, -self.max_length:]
            elif self.truncation_mode == "keep_start":
                input_ids = input_ids[:, :self.max_length]
                attention_mask = attention_mask[:, :self.max_length]
                loss_mask = loss_mask[:, :self.max_length]
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # 5) 前向
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits   # [2B, T, V]

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        loss_mask[:, -1] = False

        labels = labels.clone()
        labels[~loss_mask] = -100

        chosen_logits = logits[:num_examples]
        rejected_logits = logits[num_examples:]

        chosen_labels = labels[:num_examples]
        rejected_labels = labels[num_examples:]
        
        '''
        chosen_scores, chosen_logps = _get_batch_sp_score(
            chosen_logits,
            chosen_labels,
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            ispos=True,
        )
        rejected_scores, rejected_logps = _get_batch_sp_score(
            rejected_logits,
            rejected_labels,
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            ispos=False,
        )
        '''
        
        
        chosen_scores, chosen_logps = _get_batch_ent_score(
            chosen_logits,
            chosen_labels,
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            ispos=True,
        )
        rejected_scores, rejected_logps = _get_batch_ent_score(
            rejected_logits,
            rejected_labels,
            alpha=self.sp_alpha,
            beta=self.sp_beta,
            ispos=False,
        )
        


        chosen_valid = chosen_labels != -100
        rejected_valid = rejected_labels != -100

        mean_chosen_logits = chosen_logits[chosen_valid].mean()
        mean_rejected_logits = rejected_logits[rejected_valid].mean()
        
        #rejected_logps = _get_batch_logps(rejected_logits,rejected_labels)
        #rejected_penalty = _get_rejection_penalty(rejected_logits,rejected_labels)

        return {
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
         #   "rejected_penalty": rejected_penalty,
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
        ):
        policy_out = self.concatenated_forward(model, inputs)
        policy_chosen_logps = policy_out["chosen_logps"]
        policy_rejected_logps = policy_out["rejected_logps"]
        policy_chosen_scores = policy_out["chosen_scores"]
        policy_rejected_scores = policy_out["rejected_scores"]

        if self.reference_free or self.ref_model is None:
            ref_chosen_logps = torch.zeros_like(policy_chosen_scores)
            ref_rejected_logps = torch.zeros_like(policy_rejected_scores)
        else:
            with torch.no_grad():
                ref_out = self.concatenated_forward(self.ref_model, inputs)
            ref_chosen_logps = ref_out["chosen_logps"]
            ref_rejected_logps = ref_out["rejected_logps"]
            ref_chosen_scores = ref_out["chosen_scores"]
            ref_rejected_scores = ref_out["rejected_scores"]
        
        # change gradient
        # policy_rejected_scores = 1.1 * policy_rejected_scores - 0.1 * policy_rejected_scores.detach()
        
        dpo_loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_scores,
            policy_rejected_logps=policy_rejected_scores,
            reference_chosen_logps=ref_chosen_scores,
            reference_rejected_logps=ref_rejected_scores,
        )   
        
        #rejected_penalty = policy_out['rejected_penalty']
        
        loss = dpo_loss
        #loss = dpo_loss + self.beta * 0.1 * rejected_penalty 

        metrics = {
            "rewards/chosen": chosen_rewards.mean().detach().cpu(),
            "rewards/rejected": rejected_rewards.mean().detach().cpu(),
            "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().detach().cpu(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().detach().cpu(),
            "logps/chosen": policy_chosen_logps.mean().detach().cpu(),
            "logps/rejected": policy_rejected_logps.mean().detach().cpu(),
            "logits/chosen": policy_out["chosen_logits"].mean().detach().cpu(),
            "logits/rejected": policy_out["rejected_logits"].mean().detach().cpu(),
            "scores/chosen": policy_chosen_scores.mean().detach().cpu(),
            "scores/rejected": policy_rejected_scores.mean().detach().cpu(),
        }
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            out = dict(policy_out)
            out.update({
                "loss_dpo": dpo_loss.detach(),
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            })
            return loss.mean(), out

        return loss.mean()
