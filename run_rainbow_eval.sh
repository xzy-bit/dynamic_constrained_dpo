#!/bin/bash
set -euo pipefail
# ---------------------------------------------------------------------------
# Local RainbowPO 70B-judge pairwise evaluation.
#
# Usage:
#   ./run_rainbow_eval.sh model_a=responses/a.json model_b=responses/b.json
#
# Requires all 4 GPUs (TP=4) for the 70B judge. Run AFTER training finishes.
# ---------------------------------------------------------------------------

cd "$(dirname "$0")"

source /home/ziyang/anaconda3/bin/activate

export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

JUDGE_MODEL="/data/huggingface_cache/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f"
RAINBOW_ROOT="/home/ziyang/RainbowPO/alpaca_eval"
EVAL_SCRIPT="/home/ziyang/dpo_opt/tools/eval_base_dpo_hypo_rainbow_local70b.py"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
NUM_PROCS="${NUM_PROCS:-4}"
MAX_INSTANCES="${MAX_INSTANCES:-}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/rainbow_70b}"

mkdir -p "$OUTPUT_DIR"
SERVER_LOG="${OUTPUT_DIR}/local70b_server.log"

# ── Step 0: Convert gen.py outputs → AlpacaEval format ────────────────────
# Expects model_name=path pairs, e.g.  dpo=responses/dpo.json
MODEL_ARGS=()
for mapping in "$@"; do
  name="${mapping%%=*}"
  path="${mapping#*=}"
  converted="responses/${name}_alpaca.json"
  python scripts/convert_gen_to_alpaca.py --input "$path" --output "$converted"
  MODEL_ARGS+=(--model "${name}=${converted}")
done

if [[ ${#MODEL_ARGS[@]} -lt 2 ]]; then
  echo "Need at least 2 models for pairwise eval."
  exit 2
fi

# ── Step 1: Start local 70B vLLM judge ────────────────────────────────────
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] Stopping judge server (PID $SERVER_PID)..."
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

conda activate hypo-test

echo "[eval] Starting 70B judge server (TP=$TP_SIZE, port=$PORT)..."
python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL" \
  --served-model-name Meta-Llama-3-70B-Instruct \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[eval] Waiting for judge server (PID $SERVER_PID)..."
for attempt in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[eval] Judge server ready."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[eval] Judge server died. Last log lines:"
    tail -n 50 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || {
  echo "[eval] Timed out waiting for judge server."; exit 1
}

# ── Step 2: Run pairwise evaluation ──────────────────────────────────────
CMD=(
  python "$EVAL_SCRIPT"
  --rainbow-alpaca-eval-root "$RAINBOW_ROOT"
  --output-dir "$OUTPUT_DIR"
  --num-procs "$NUM_PROCS"
  --seed "$SEED"
)
if [[ -n "$MAX_INSTANCES" ]]; then
  CMD+=(--max-instances "$MAX_INSTANCES")
fi
CMD+=("${MODEL_ARGS[@]}")

printf '[eval] %q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

echo "[eval] Done. Results in $OUTPUT_DIR/"
echo "[eval]   wr_matrix.csv   — raw win-rate"
echo "[eval]   lc_wr_matrix.csv — length-controlled win-rate"
