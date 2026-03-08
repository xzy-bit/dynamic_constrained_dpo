import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def default_summary_path(in_path: str) -> str:
    if in_path.endswith(".json"):
        return in_path[:-5] + "-diversity-summary.json"
    return in_path + "-diversity-summary.json"


def default_output_path(in_path: str) -> str:
    if in_path.endswith(".json"):
        return in_path[:-5] + "-diversity.json"
    return in_path + "-diversity.json"


def tokenize_simple(text: str) -> List[str]:
    return text.strip().split()


def distinct_n(texts: List[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        toks = tokenize_simple(text)
        if len(toks) < n:
            continue
        ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def sentbert_diversity(texts: List[str], model: SentenceTransformer) -> float:
    if len(texts) <= 1:
        return 0.0

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sim = cosine_similarity(emb, emb)

    vals = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(sim[i, j])

    if len(vals) == 0:
        return 0.0

    mean_sim = float(np.mean(vals))
    return 1.0 - mean_sim


def one_minus_self_bleu(texts: List[str]) -> float:
    """
    Compute 1 - self-BLEU.
    For each text, use all other texts as references.
    Then average BLEU over all texts, and return 1 - avg_self_bleu.
    """
    if len(texts) <= 1:
        return 0.0

    smoothie = SmoothingFunction().method1
    bleu_scores = []

    tokenized = [tokenize_simple(t) for t in texts]

    for i in range(len(tokenized)):
        hyp = tokenized[i]
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]

        if len(hyp) == 0 or len(refs) == 0:
            bleu_scores.append(0.0)
            continue

        # standard BLEU-4 style weights
        score = sentence_bleu(
            refs,
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        bleu_scores.append(score)

    avg_self_bleu = float(np.mean(bleu_scores))
    return 1.0 - avg_self_bleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True, help="Path to response json with outputs list")
    ap.add_argument(
        "--sentbert_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-BERT model",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Per-prompt diversity output path",
    )
    ap.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Global summary output path",
    )
    args = ap.parse_args()

    output_json = args.output_json or default_output_path(args.input_json)
    summary_json = args.summary_json or default_summary_path(args.input_json)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    print(f"Loading SBERT model: {args.sentbert_model}")
    sbert = SentenceTransformer(args.sentbert_model)

    dist1_list = []
    dist2_list = []
    dist3_list = []
    ngram_avg_list = []
    self_bleu_div_list = []
    sentbert_div_list = []

    for ex in tqdm(data, desc="Computing diversity"):
        outputs = ex.get("outputs", [])
        if not isinstance(outputs, list):
            raise ValueError("Each example must contain a list field named 'outputs'.")

        d1 = distinct_n(outputs, 1)
        d2 = distinct_n(outputs, 2)
        d3 = distinct_n(outputs, 3)
        ngram_avg = (d1 + d2 + d3) / 3.0
        self_bleu_div = one_minus_self_bleu(outputs)
        sdiv = sentbert_diversity(outputs, sbert)

        ex["diversity"] = {
            "dist_1": d1,
            "dist_2": d2,
            "dist_3": d3,
            "ngram_avg": ngram_avg,
            "one_minus_self_bleu": self_bleu_div,
            "sentbert_diversity": sdiv,
            "num_outputs": len(outputs),
        }

        dist1_list.append(d1)
        dist2_list.append(d2)
        dist3_list.append(d3)
        ngram_avg_list.append(ngram_avg)
        self_bleu_div_list.append(self_bleu_div)
        sentbert_div_list.append(sdiv)

    summary = {
        "num_prompts": len(data),
        "avg_dist_1": float(np.mean(dist1_list)) if dist1_list else 0.0,
        "avg_dist_2": float(np.mean(dist2_list)) if dist2_list else 0.0,
        "avg_dist_3": float(np.mean(dist3_list)) if dist3_list else 0.0,
        "avg_ngram_avg": float(np.mean(ngram_avg_list)) if ngram_avg_list else 0.0,
        "avg_one_minus_self_bleu": float(np.mean(self_bleu_div_list)) if self_bleu_div_list else 0.0,
        "avg_sentbert_diversity": float(np.mean(sentbert_div_list)) if sentbert_div_list else 0.0,
    }
    print(summary)
    #os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    #with open(output_json, "w", encoding="utf-8") as f:
    #    json.dump(data, f, ensure_ascii=False, indent=2)

    #os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    #with open(summary_json, "w", encoding="utf-8") as f:
    #    json.dump(summary, f, ensure_ascii=False, indent=2)

    #print(f"Saved per-prompt diversity to: {output_json}")
    #print(f"Saved summary to: {summary_json}")
    #print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
