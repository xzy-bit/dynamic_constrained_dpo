import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def pick(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--output_file", type=str, default=None, help="Override output path.")
    ap.add_argument("--model_path", type=str, default=None, help="Override model checkpoint/repo.")
    ap.add_argument("--generator_name", type=str, default=None, help="Override generator name.")
    args = ap.parse_args()

    # -------------------------
    # Load YAML
    # -------------------------
    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    # simpo-style fields
    completions_kwargs = pick(cfg_all, "completions_kwargs", {})
    model_kwargs = pick(completions_kwargs, "model_kwargs", {})

    # -------------------------
    # Resolve required settings
    # -------------------------
    # Model name/path (YAML: completions_kwargs.model_name)
    model_name = args.model_path or pick(completions_kwargs, "model_name", None)
    if model_name is None:
        raise ValueError("Missing model path/name. Provide --model_path or set completions_kwargs.model_name in YAML.")

    # Tokenizer: 如果 YAML 不提供，就默认等于 model_name；也允许 CLI 覆盖
    tokenizer_name = pick(completions_kwargs, "tokenizer_name_or_path", None)

    # Prompt template path (YAML: prompt_template)
    prompt_template_path = pick(cfg_all, "prompt_template", None)
    if prompt_template_path is None:
        raise ValueError('Missing prompt_template in YAML (e.g., templates/llama3.txt).')

    prompt_template = read_text(prompt_template_path)

    # -------------------------
    # Dataset
    # -------------------------
    print("Loading AlpacaEval dataset...")
    eval_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

    def format_prompt(instruction):
        return prompt_template.format(instruction=instruction)

    prompts = [format_prompt(item["instruction"]) for item in eval_dataset]

    # -------------------------
    # Tokenizer
    # -------------------------
    # 这里主要是确保 pad_token
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # vLLM init config (consume subset from YAML)
    # -------------------------

    llm_kwargs: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype="bfloat16",
    )

    print("Initializing vLLM...")
    llm = LLM(**llm_kwargs)

    # -------------------------
    # Sampling params (consume subset from YAML)
    # -------------------------
    temperature = float(pick(completions_kwargs, "temperature", 0.6))
    top_p = float(pick(completions_kwargs, "top_p", 1.0))
    max_tokens = int(pick(completions_kwargs, "max_new_tokens", 2048))
    stop_token_ids = pick(completions_kwargs, "stop_token_ids", None)
    if stop_token_ids is not None and not isinstance(stop_token_ids, list):
        raise ValueError("stop_token_ids must be a list in YAML.")

    sampling_kwargs: Dict[str, Any] = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=16,
    )
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = [int(x) for x in stop_token_ids]

    sampling_params = SamplingParams(**sampling_kwargs)

    # -------------------------
    # Generate
    # -------------------------
    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        # output.outputs: List[CompletionOutput]，长度应为 n
        cand_texts = [o.text.strip() for o in output.outputs]

        results.append({
            "instruction": eval_dataset[i]["instruction"],
            "outputs": cand_texts,  # <- list[str], length=n
            "generator": args.generator_name,
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
