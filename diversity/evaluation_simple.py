import json
import numpy as np
from nltk.util import ngrams
from nltk import word_tokenize
import sacrebleu
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


def load_responses(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = []

    for x in data:

        if "output" in x:
            responses.append(x["output"])

        elif "response" in x:
            responses.append(x["response"])

        elif "text" in x:
            responses.append(x["text"])

    return responses


# -------------------------
# Distinct n-gram
# -------------------------

def distinct_n(responses, n):

    all_ngrams = []

    for r in responses:
        tokens = word_tokenize(r)
        all_ngrams += list(ngrams(tokens, n))

    if len(all_ngrams) == 0:
        return 0

    return len(set(all_ngrams)) / len(all_ngrams)


# -------------------------
# Self BLEU
# -------------------------

def self_bleu(responses):

    scores = []

    for i in range(len(responses)):

        hyp = [responses[i]]
        refs = [[responses[j]] for j in range(len(responses)) if j != i]

        score = sacrebleu.corpus_bleu(hyp, refs).score
        scores.append(score)

    return np.mean(scores)


# -------------------------
# SentBERT diversity
# -------------------------

def sentbert_diversity(responses):

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    emb = model.encode(responses, convert_to_tensor=True, normalize_embeddings=True)

    if torch.cuda.is_available():
        emb = emb.cuda()

    sims = []

    for i in range(len(emb)):
        for j in range(i):
            sims.append(torch.dot(emb[i], emb[j]).item())

    mean_sim = np.mean(sims)

    return 1 - mean_sim


# -------------------------
# Main
# -------------------------

def main():

    path = "sp_dpo.json"

    responses = load_responses(path)

    print("num responses:", len(responses))

    d1 = distinct_n(responses, 1)
    d2 = distinct_n(responses, 2)
    d3 = distinct_n(responses, 3)

    self_bleu_score = self_bleu(responses)

    sb = sentbert_diversity(responses)

    results = {

        "distinct-1": round(d1 * 100, 2),
        "distinct-2": round(d2 * 100, 2),
        "distinct-3": round(d3 * 100, 2),

        "100-selfbleu": round(100 - self_bleu_score, 2),

        "sentbert_diversity": round(sb * 100, 2)

    }

    print(results)


if __name__ == "__main__":
    main()