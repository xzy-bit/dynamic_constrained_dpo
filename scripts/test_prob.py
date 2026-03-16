import argparse
import json
import math
import os
from typing import Any, Dict

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pick(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def extract_mean_prob(completion):
    """
    计算一个 response 的平均 selected token prob
    """
    token_ids = completion.token_ids
    logprobs = completion.logprobs

    if not token_ids or not logprobs:
        return None

    probs = []

    for pos, tok_id in enumerate(token_ids):
        pos_dict = logprobs[pos]

        # vLLM 返回 dict[token_id] -> logprob object
        obj = pos_dict.get(tok_id) or pos_dict.get(str(tok_id))

        if hasattr(obj, "logprob"):
            lp = obj.logprob
        elif isinstance(obj, dict):
            lp = obj["logprob"]
        else:
            lp = float(obj)

        probs.append(math.exp(lp))

    return sum(probs) / len(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", type=str, required=True)
    ap.add_argument("--output_file", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    args = ap.parse_args()

    with open(args.config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    completions_kwargs = pick(cfg, "completions_kwargs", {})

    model_name = args.model_path or pick(completions_kwargs, "model_name")
    tokenizer_name = pick(completions_kwargs, "tokenizer_name_or_path", model_name)

    prompt_template = read_text(cfg["prompt_template"])

    print("Loading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

    prompts = [
        prompt_template.format(instruction=item["instruction"])
        for item in dataset
    ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print("Initializing vLLM...")
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=float(pick(completions_kwargs, "temperature", 0.6)),
        top_p=float(pick(completions_kwargs, "top_p", 1.0)),
        max_tokens=int(pick(completions_kwargs, "max_new_tokens", 2048)),
        n=16,
        logprobs=1,   # 关键
    )

    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    all_points = []

    for output in outputs:
        for completion in output.outputs:
            mean_prob = extract_mean_prob(completion)
            if mean_prob is not None:
                all_points.append(mean_prob)

    print("Total points:", len(all_points))  # 应该是 805*16

    with open(args.output_file, "w") as f:
        json.dump({"points": all_points}, f)

    print("Saved to", args.output_file)


if __name__ == "__main__":
    main()
