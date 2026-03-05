#!/bin/sh

set -e
set -x

TOKENIZER_PATH="meta-llama/Llama-3.1-8B-Instruct"

python evaluation_diversity.py \
    --tokenizer_path "meta-llama/Llama-3.1-8B-Instruct" \
    --detokenizer_path "meta-llama/Llama-3.1-8B-Instruct"\
    --response_path "dpo.json" \
    2>&1 | tee cw.log
