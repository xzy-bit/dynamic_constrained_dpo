import argparse
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MISTRAL_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""


def apply_chat_template(example, tokenizer):

    if "messages" in example:
        msgs = example["messages"]

        if msgs[-1]["role"] == "assistant":
            prompt_msgs = msgs[:-1]
        else:
            prompt_msgs = msgs

        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = example["prompt"]

    chosen = example["chosen"]
    rejected = example["rejected"]

    if isinstance(chosen, list):
        chosen = tokenizer.apply_chat_template(chosen[-1:], tokenize=False)

    if isinstance(rejected, list):
        rejected = tokenizer.apply_chat_template(rejected[-1:], tokenize=False)

    if tokenizer.bos_token:
        if chosen.startswith(tokenizer.bos_token):
            chosen = chosen[len(tokenizer.bos_token):]

        if rejected.startswith(tokenizer.bos_token):
            rejected = rejected[len(tokenizer.bos_token):]

    return prompt, chosen, rejected


@torch.no_grad()
def compute_mean_prob(model, tokenizer, prompt, response, device):

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids

    ids = prompt_ids + resp_ids
    input_ids = torch.tensor([ids]).to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logps = F.log_softmax(shift_logits,dim=-1) 
    probs = F.softmax(shift_logits, dim=-1)

    token_probs = probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_logps = logps.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    start = len(prompt_ids) - 1
    end = len(prompt_ids) + len(resp_ids) - 1

    response_probs = token_probs[:, start:end]
    response_logps = token_logps[:, start:end]

    return response_probs.mean().item(),response_logps.sum().item()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs"
    )
    dataset = dataset.select(range(10000))
    chosen_probs = []
    chosen_logps = []
    rejected_probs = []
    rejected_logps = []

    for example in tqdm(dataset):

        prompt, chosen, rejected = apply_chat_template(example, tokenizer)

        c_probs, c_logps = compute_mean_prob(model, tokenizer, prompt, chosen, device)
        
        r_probs, r_logps = compute_mean_prob(model, tokenizer, prompt, rejected, device)

        chosen_probs.append(c_probs)
        rejected_probs.append(r_probs)

        chosen_logps.append(c_logps)
        rejected_logps.append(r_logps)


    print("Saving...")

    with open(args.output_file, "w") as f:
        json.dump(
            {
                "chosen": chosen_probs,
                "chosen_logps": chosen_logps,
                "rejected": rejected_probs,
                "rejected_logps": rejected_logps
            },
            f
        )

    print("Saved to", args.output_file)
    print("Total samples:", len(chosen_probs))


if __name__ == "__main__":
    main()
