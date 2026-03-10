import torch
import torch.nn.functional as F
from entmax import sparsemax_loss, sparsemax, entmax_bisect_loss,entmax15,entmax_bisect
from trl import DPOTrainer

IGNORE_INDEX = -100

def _right_pad(x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    # x: [B, L]
    pad_len = target_len - x.size(1)
    if pad_len <= 0:
        return x
    return F.pad(x, (0, pad_len), value=pad_value)

def _build_labels_from_prompt(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    prompt_attention_mask: torch.LongTensor,
    ignore_index: int = -100,
) -> torch.LongTensor:
    """
    labels = input_ids, but mask:
      - prompt tokens (per-example prompt length) -> ignore_index
      - padding tokens (attention_mask==0) -> ignore_index
    """
    labels = input_ids.clone()

    # 每条样本的 prompt 长度（含 special tokens 取决于你 template/collator 的定义）
    prompt_lens = prompt_attention_mask.sum(dim=1)  # [B]

    B, L = labels.shape
    # mask prompt 部分
    arange = torch.arange(L, device=labels.device).unsqueeze(0).expand(B, L)  # [B, L]
    prompt_mask = arange < prompt_lens.unsqueeze(1)  # [B, L]
    labels[prompt_mask] = ignore_index

    # mask padding
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    return labels

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
    
    ce_loss = -per_token_logps.sum(-1)

    return scores, ce_loss

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

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch):
        prompt_attention_mask = batch["prompt_attention_mask"]

        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        # 1) 统一长度（否则 torch.cat 会炸）
        max_len = max(chosen_input_ids.size(1), rejected_input_ids.size(1))

        # pad_token_id 的获取：tokenizer -> model.config.pad_token_id -> eos_token_id -> 0
        pad_id = None
        if getattr(self, "processing_class", None) is not None and self.processing_class.pad_token_id is not None:
            pad_id = self.processing_class.pad_token_id
        else:
            pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(getattr(model, "config", None), "eos_token_id", None)
        if pad_id is None:
            pad_id = 0

        chosen_input_ids = _right_pad(chosen_input_ids, max_len, pad_id)
        rejected_input_ids = _right_pad(rejected_input_ids, max_len, pad_id)
        chosen_attention_mask = _right_pad(chosen_attention_mask, max_len, 0)
        rejected_attention_mask = _right_pad(rejected_attention_mask, max_len, 0)

        # 2) 在 pad 后的序列上构造 labels（保证和 logits 对齐）
        chosen_labels = _build_labels_from_prompt(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            ignore_index=IGNORE_INDEX,
        )
        rejected_labels = _build_labels_from_prompt(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            ignore_index=IGNORE_INDEX,
        )

        # 3) 拼接前向
        concatenated_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        concatenated_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        outputs = model(
            input_ids=concatenated_input_ids,
            attention_mask=concatenated_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [2B, L, V]

        bsz = chosen_input_ids.size(0)
        chosen_logits, rejected_logits = logits.split(bsz, dim=0)

        # 4) 用 entmax/sparsemax-family 的“序列级分数”作为 logps（否则你返回 0 学不到）
        #    通常把 chosen 当作 ispos=True，rejected 当作 ispos=False
        chosen_scores, chosen_logps = _get_batch_ent_score(
            chosen_logits, chosen_labels,
            alpha=self.sp_alpha, beta=self.sp_beta, ispos=True
        )
        rejected_scores, rejected_logps = _get_batch_ent_score(
            rejected_logits, rejected_labels,
            alpha=self.sp_alpha, beta=self.sp_beta, ispos=False
        )

        #print("chosen_scores:", chosen_scores[:5])
        #print("rejected_scores:", rejected_scores[:5])
        #print("margin:", (chosen_scores - rejected_scores)[:5])

        return {
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
            "mean_chosen_logits": chosen_logits.mean(),
            "mean_rejected_logits": rejected_logits.mean(),

            "mean_chosen_logps": chosen_logps.mean(),
            "mean_rejected_logps": rejected_logps.mean(),
            }
    
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        policy_out = self.concatenated_forward(model, inputs)
        policy_chosen_logps = policy_out["chosen_logps"]
        policy_rejected_logps = policy_out["rejected_logps"]
        policy_chosen_scores = policy_out["chosen_scores"]
        policy_rejected_scores = policy_out["rejected_scores"]


        # reference forward
        if self.reference_free or self.ref_model is None:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            with torch.no_grad():
                ref_out = self.concatenated_forward(self.ref_model, inputs)
            ref_chosen_scores = ref_out["chosen_scores"]
            ref_rejected_scores = ref_out["rejected_scores"]

        dpo_loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_scores,
            policy_rejected_logps=policy_rejected_scores,
            reference_chosen_logps=ref_chosen_scores,
            reference_rejected_logps=ref_rejected_scores,
        )

        loss = dpo_loss.mean()
        
        if self.state.global_step % self.args.logging_steps == 0:
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

            metrics = {
                "loss_dpo": dpo_loss.mean().detach().float().item(),
                "rewards/chosen": chosen_rewards.mean().detach().float().item(),
                "rewards/rejected": rejected_rewards.mean().detach().float().item(),
                "scores/chosen": policy_chosen_scores.mean().detach().float().item(),
                "scores/rejected": policy_rejected_scores.mean().detach().float().item(),
                "rewards/margins": (chosen_rewards - rejected_rewards).mean().detach().float().item(),
                "rewards/accuracies": reward_accuracy.detach().float().item(),

                "logits/chosen": policy_out["chosen_logits"].mean().detach().float().item(),
                "logits/rejected": policy_out["rejected_logits"].mean().detach().float().item(),

                "logps/chosen": -policy_chosen_logps.mean().detach().float().item(),
                "logps/rejected": -policy_rejected_logps.mean().detach().float().item(),
                "logps/margins": (policy_rejected_logps - policy_chosen_logps).mean().detach().float().item(),
                "logps/accuracies": (policy_chosen_logps < policy_rejected_logps).float().mean().item()
            }

            self.log(metrics)


        if return_outputs:
            policy_out = dict(policy_out)
            policy_out.update({
                "loss_dpo": dpo_loss.detach(),
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            })
            return loss.mean(), policy_out

        return loss.mean()
    
