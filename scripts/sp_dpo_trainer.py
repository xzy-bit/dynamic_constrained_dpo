import torch
import torch.nn.functional as F
from entmax import sparsemax_loss, sparsemax, entmax_bisect_loss,entmax15,entmax_bisect
from trl import DPOTrainer
from typing import Dict, Union, List
import torch.nn as nn
from trl.trainer.utils import pad_to_length

IGNORE_INDEX = -100

def _get_rejection_penalty(logits, labels, eps=1e-8):
    B, M, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != IGNORE_INDEX)

    active_logits = shift_logits[mask]   # [num_active, V]
    active_labels = shift_labels[mask]   # [num_active]

    if active_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    
    probs = F.softmax(active_logits,dim=-1)
    #probs = entmax_bisect(active_logits, alpha=alpha, dim=-1, n_iter=50)  # [num_active, V]
    p_target = probs.gather(-1, active_labels.unsqueeze(-1)).squeeze(-1)  # [num_active]

    # 计算 -log(1-p)
    penalty_per_token = -torch.log(1 - p_target + eps)  # [num_active]

    token_penalty = torch.zeros(B, M-1, device=logits.device, dtype=penalty_per_token.dtype)
    token_penalty[mask] = penalty_per_token
    seq_penalty = token_penalty.sum(dim=-1)  # [B]
    return seq_penalty

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
    shift_logits = logits[:, :-1, :].contiguous()   # [B, M-1, V]
    shift_labels = labels[:, 1:].contiguous()       # [B, M-1]
    mask = (shift_labels != -100)
    shift_labels = shift_labels.masked_fill(~mask, 0)

    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)

    # token-level entmax loss
    flat_loss = entmax_bisect_loss(flat_logits, flat_labels, alpha, n_iter=50)  # [B*(M-1)]
    token_loss = flat_loss.view(B, M - 1)
    
    per_token_logps = []
    
    '''
    for row_logits, row_labels in zip(shift_logits, shift_labels, strict=True):
        row_logps = F.log_softmax(row_logits, dim=-1)   # [M-1, V]
        row_per_token_logps = row_logps.gather(
            dim=-1,
            index=row_labels.unsqueeze(-1)              # [M-1, 1]
        ).squeeze(-1)                                   # [M-1]
        per_token_logps.append(row_per_token_logps)

    per_token_logps = torch.stack(per_token_logps)      # [B, M-1]
    per_token_logps = per_token_logps.masked_fill(~mask, 0.0)
    '''
    
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

    token_loss = token_loss * mask
    scores = -token_loss.sum(-1)   # [B]
    
    #per_token_logps  = per_token_logps.sum(-1)

    return scores, per_token_logps

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
        self.sp_alpha = sp_alpha
        self.sp_beta = sp_beta
        self.reference_free = reference_free
        self.rejected_tail = 0.0

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


        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        #loss = -F.logsigmoid(self.beta * logits) + self.beta * self.rejected_tail

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

    def concatenated_inputs(self,batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        
        return concatenated_batch

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_labels = concatenated_batch['concatenated_input_ids'].clone()
        all_labels[concatenated_batch['concatenated_attention_mask'] == 0] = -100

        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits
        bsz = batch['chosen_input_ids'].shape[0]

        chosen_labels = all_labels[:bsz]
        rejected_labels = all_labels[bsz:]

        chosen_logits, rejected_logits = all_logits.split(bsz, dim=0)

        chosen_scores, chosen_logps = _get_batch_ent_score(
            chosen_logits, chosen_labels,
            alpha=self.sp_alpha, beta=self.sp_beta, ispos=True
        )
        rejected_scores, rejected_logps = _get_batch_ent_score(
            rejected_logits, rejected_labels,
            alpha=self.sp_alpha, beta=self.sp_beta, ispos=False
        )
        #self.rejected_tail = _get_rejection_penalty(rejected_logits,rejected_labels)
        


        return {
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_logps": chosen_scores,
            "rejected_logps": rejected_scores,
            "mean_chosen_logits": chosen_logits.mean(),
            "mean_rejected_logits": rejected_logits.mean(),

            # "chosen_scores": chosen_scores,
            # "rejected_scores": rejected_scores,

            }

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #
    #     policy_out = self.concatenated_forward(model, inputs)
    #     policy_chosen_logps = policy_out["chosen_logps"]
    #     policy_rejected_logps = policy_out["rejected_logps"]
    #     policy_chosen_scores = policy_out["chosen_scores"]
    #     policy_rejected_scores = policy_out["rejected_scores"]
    #
    #
    #     # reference forward
    #     if self.reference_free or self.ref_model is None:
    #         ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
    #         ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
    #     else:
    #         with torch.no_grad():
    #             ref_out = self.concatenated_forward(self.ref_model, inputs)
    #         ref_chosen_scores = ref_out["chosen_scores"]
    #         ref_rejected_scores = ref_out["rejected_scores"]
    #
    #     dpo_loss, chosen_rewards, rejected_rewards = self.dpo_loss(
    #         policy_chosen_logps=policy_chosen_scores,
    #         policy_rejected_logps=policy_rejected_scores,
    #         reference_chosen_logps=ref_chosen_scores,
    #         reference_rejected_logps=ref_rejected_scores,
    #     )
    #
    #     loss = dpo_loss.mean()
    #
    #     if self.state.global_step % self.args.logging_steps == 0:
    #         reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
    #
    #         metrics = {
    #             "loss_dpo": dpo_loss.mean().detach().float().item(),
    #             "rewards/chosen": chosen_rewards.mean().detach().float().item(),
    #             "rewards/rejected": rejected_rewards.mean().detach().float().item(),
    #             "scores/chosen": policy_chosen_scores.mean().detach().float().item(),
    #             "scores/rejected": policy_rejected_scores.mean().detach().float().item(),
    #             "rewards/margins": (chosen_rewards - rejected_rewards).mean().detach().float().item(),
    #             "rewards/accuracies": reward_accuracy.detach().float().item(),
    #
    #             "logits/chosen": policy_out["chosen_logits"].mean().detach().float().item(),
    #             "logits/rejected": policy_out["rejected_logits"].mean().detach().float().item(),
    #
    #             "logps/chosen": -policy_chosen_logps.mean().detach().float().item(),
    #             "logps/rejected": -policy_rejected_logps.mean().detach().float().item(),
    #             "logps/margins": (policy_rejected_logps - policy_chosen_logps).mean().detach().float().item(),
    #             "logps/accuracies": (policy_chosen_logps < policy_rejected_logps).float().mean().item()
    #         }
    #
    #         self.log(metrics)
    #
    #
    #     if return_outputs:
    #         policy_out = dict(policy_out)
    #         policy_out.update({
    #             "loss_dpo": dpo_loss.detach(),
    #             "chosen_rewards": chosen_rewards,
    #             "rejected_rewards": rejected_rewards,
    #         })
    #         return loss.mean(), policy_out
    #
    #     return loss.mean()
    
