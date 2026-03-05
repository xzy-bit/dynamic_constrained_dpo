#################
# Diversity evaluation: Distinct-n, 100-SelfBLEU, Sent-BERT diversity
# Compatible with:
# 1) raw json: list[{"prompt": str, "answer": list[str]}]
# 2) cleaned json: list[list[str]]  (shape: [k][N])
#################

import os
import json
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sacrebleu
from transformers import HfArgumentParser, AutoTokenizer
from nltk.util import ngrams
from nltk import word_tokenize
import sentence_transformers


@dataclass
class AllArguments:
    response_path: str = field(
        default="./results/responses.json",
        metadata={"help": "Path to response json/jsonl. Supports raw or cleaned format."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer for apply_chat_template. Required if building cleaned set from raw."},
    )
    detokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional detokenizer to decode prompt; usually same as tokenizer_path."},
    )
    # If True, evaluate only answers (assistant content) rather than prompt+template
    eval_answer_only: bool = field(
        default=True,
        metadata={"help": "If True, evaluate only answer strings (recommended)."},
    )
    # Sent-BERT model
    sentbert_model: str = field(
        default="sentence-transformers/all-mpnet-base-v2",
        metadata={"help": "SentenceTransformer model name."},
    )
    sentbert_batch_size: int = field(default=256)
    # For speed/memory control
    max_pairs_per_prompt: int = field(
        default=0,
        metadata={
            "help": "If >0, sample at most this many pairs per prompt for Sent-BERT similarity (0 means use all pairs)."
        },
    )


# -----------------------
# IO helpers
# -----------------------
def load_json_utf8(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_utf8(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_cleaned_format(obj: Any) -> bool:
    # cleaned: list[list[str]]
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], list)
        and (len(obj[0]) == 0 or isinstance(obj[0][0], str))
    )


def is_raw_format(obj: Any) -> bool:
    # raw: list[dict(prompt, answer)]
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], dict)
        and ("answer" in obj[0])
    )


def ensure_response_set(
    args: AllArguments,
) -> Tuple[List[List[str]], str]:
    """
    Returns:
      response_set: List[List[str]] of shape [k][N]
      cleaned_path: path of cleaned json (may equal args.response_path if already cleaned)
    """
    cleaned_path = args.response_path.replace(".json", "-cleaned.json")
    if os.path.exists(cleaned_path):
        obj = load_json_utf8(cleaned_path)
        if not is_cleaned_format(obj):
            raise ValueError(f"{cleaned_path} exists but is not cleaned format [k][N].")
        return obj, cleaned_path

    obj = load_json_utf8(args.response_path)

    # If already cleaned
    if is_cleaned_format(obj):
        dump_json_utf8(obj, cleaned_path)
        return obj, cleaned_path

    # If raw format: build cleaned
    if not is_raw_format(obj):
        raise ValueError(
            "Unrecognized json format. Expected either cleaned [k][N] or raw list[{prompt, answer:list[str]}]."
        )

    if args.tokenizer_path is None:
        raise ValueError("tokenizer_path is required to build cleaned set from raw json.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    detokenizer = None
    if args.detokenizer_path is not None:
        detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)

    data: List[Dict[str, Any]] = obj
    response_set: List[List[str]] = []

    for i in tqdm(range(len(data)), desc="Building cleaned response_set"):
        x = data[i]
        answers = x["answer"]
        if not isinstance(answers, list) or len(answers) == 0:
            continue

        k = len(answers)
        if len(response_set) == 0:
            response_set = [[] for _ in range(k)]
        else:
            if len(response_set) != k:
                raise ValueError(f"Inconsistent number of answers at i={i}: {len(response_set)} vs {k}")

        # prompt normalization
        prompt_str = x.get("prompt", "")
        if detokenizer is not None:
            prompt_str = detokenizer.decode(detokenizer.encode(prompt_str), skip_special_tokens=True)
            prompt_str = prompt_str.replace("user\n\n", "").replace("assistant\n\n", "")

        for j in range(k):
            ans_str = answers[j]
            if not isinstance(ans_str, str):
                ans_str = str(ans_str)

            # common cleanup
            ans_str = ans_str.replace("<|eot_id|>", "").strip()

            if args.eval_answer_only:
                # evaluate answer only (recommended)
                response_set[j].append(ans_str)
            else:
                # evaluate full chat template text (prompt + assistant)
                chat = [{"role": "user", "content": prompt_str}, {"role": "assistant", "content": ans_str}]
                res = tokenizer.apply_chat_template(chat, tokenize=False)
                response_set[j].append(res)

    dump_json_utf8(response_set, cleaned_path)
    return response_set, cleaned_path


# -----------------------
# Metrics
# -----------------------
class AveragedDistinctNgrams:
    """
    Averaged Distinct-n across prompts and across n in [n_min, n_max].
    response_set: [k][N]
      - for each prompt t: we take texts = [response_set[j][t] for j in 0..k-1]
      - compute distinct-n over the k texts (pooled ngrams)
    """
    def __init__(self, n_min: int = 1, n_max: int = 3):
        assert n_min >= 1 and n_max >= n_min
        self.n_min = n_min
        self.n_max = n_max

    def __call__(self, response_set: List[List[str]]) -> float:
        k = len(response_set)
        N = len(response_set[0]) if k > 0 else 0
        scores = []
        for t in tqdm(range(N), desc="Distinct-n"):
            texts = [response_set[j][t] for j in range(k)]
            for n in range(self.n_min, self.n_max + 1):
                scores.append(self._distinct_n(texts, n))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _distinct_n(texts: List[str], n: int) -> float:
        all_ngrams = []
        for s in texts:
            toks = word_tokenize(s)
            all_ngrams.extend(list(ngrams(toks, n)))
        total = len(all_ngrams)
        if total == 0:
            return 0.0
        uniq = len(set(all_ngrams))
        return uniq / total


class SelfBLEU:
    """
    Self-BLEU per prompt:
      for each hypothesis among k texts, refs are the remaining (k-1) texts as multi-reference,
      compute BLEU; then average across hypotheses; then average across prompts.
    Return range about [0,100]. Higher means more similar (less diverse).
    """
    def __call__(self, response_set: List[List[str]]) -> float:
        k = len(response_set)
        N = len(response_set[0]) if k > 0 else 0
        per_prompt = []
        for t in tqdm(range(N), desc="Self-BLEU"):
            texts = [response_set[j][t] for j in range(k)]
            per_prompt.append(self._self_bleu_one_prompt(texts))
        return float(np.mean(per_prompt)) if per_prompt else 0.0

    @staticmethod
    def _self_bleu_one_prompt(texts: List[str]) -> float:
        k = len(texts)
        if k <= 1:
            return 0.0
        scores = []
        for i in range(k):
            hyp = [texts[i]]
            refs = [[texts[r]] for r in range(k) if r != i]  # multi-ref, each is a corpus of length 1
            bleu = sacrebleu.corpus_bleu(hyp, refs).score
            scores.append(bleu)
        return float(np.mean(scores)) if scores else 0.0


class SentBertDiversity:
    """
    Sent-BERT diversity:
      For each prompt t:
        - embed the k responses -> E [k, d] (normalized)
        - compute cosine sim matrix S = E @ E^T
        - mean_sim = mean upper-triangle (excluding diagonal)
        - diversity_t = 1 - mean_sim
      overall diversity = mean_t diversity_t
    Return in [0,1] typically. Larger means more diverse.
    """
    def __init__(self, model_name: str, batch_size: int = 256, max_pairs_per_prompt: int = 0):
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.max_pairs_per_prompt = max_pairs_per_prompt
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))

    @torch.no_grad()
    def __call__(self, response_set: List[List[str]]) -> float:
        k = len(response_set)
        N = len(response_set[0]) if k > 0 else 0
        if k <= 1 or N == 0:
            return 0.0

        # flatten encode once
        flat: List[str] = []
        for j in range(k):
            flat.extend(response_set[j])  # N each

        emb = self.model.encode(
            flat,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # normalized => dot=cos sim
        )
        if torch.cuda.is_available():
            emb = emb.to(torch.device("cuda"))

        emb = emb.view(k, N, -1)  # [k, N, d]

        rng = np.random.default_rng(42)
        diversities = []

        for t in tqdm(range(N), desc="Sent-BERT diversity"):
            E = emb[:, t, :]  # [k, d]
            # full sim matrix
            S = E @ E.T  # [k, k]

            if self.max_pairs_per_prompt and self.max_pairs_per_prompt > 0:
                # sample pairs (i<j)
                pairs = []
                for i in range(k):
                    for j in range(i + 1, k):
                        pairs.append((i, j))
                if len(pairs) > self.max_pairs_per_prompt:
                    idx = rng.choice(len(pairs), size=self.max_pairs_per_prompt, replace=False)
                    pairs = [pairs[p] for p in idx]
                sims = torch.stack([S[i, j] for (i, j) in pairs]).mean()
                mean_sim = sims
            else:
                triu = torch.triu(S, diagonal=1)
                denom = k * (k - 1) / 2
                mean_sim = triu.sum() / denom

            diversities.append(1.0 - float(mean_sim.item()))

        return float(np.mean(diversities)) if diversities else 0.0


# -----------------------
# Main
# -----------------------
def main():
    parser = HfArgumentParser((AllArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    response_set, cleaned_path = ensure_response_set(args)
    print(f"Loaded response_set shape: k={len(response_set)}, N={len(response_set[0]) if response_set else 0}")
    print(f"Cleaned file: {cleaned_path}")

    results = {}

    # 1) N-gram (Distinct-1..3 averaged)
    print("\nCalculating N-gram diversity (Averaged Distinct-1..3)...")
    ng_metric = AveragedDistinctNgrams(n_min=1, n_max=3)
    ng = ng_metric(response_set)
    results["Ngram_diversity_1to3"] = round(ng * 100, 2)  # ↑

    # 2) 100 - SelfBLEU
    print("\nCalculating Self-BLEU similarity...")
    bleu_metric = SelfBLEU()
    self_bleu = bleu_metric(response_set)  # 0..100 (↑ similar)
    results["100_minus_SelfBLEU"] = round(100.0 - self_bleu, 2)  # ↑ diverse

    # 3) Sent-BERT diversity
    print("\nCalculating Sent-BERT diversity...")
    sb_metric = SentBertDiversity(
        model_name=args.sentbert_model,
        batch_size=args.sentbert_batch_size,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
    )
    sb = sb_metric(response_set)
    results["SentBERT_diversity"] = round(sb * 100, 2)  # ↑

    print("\n=== Diversity Results (higher is more diverse) ===")
    pprint(results)

    # Optionally save results next to cleaned file
    out_path = cleaned_path.replace(".json", "-diversity.json")
    dump_json_utf8(results, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()