#!/usr/bin/env bash
# sbatch submit_validation_eval.sh PROMPT DATASET REPETITION [OUTPUT_LABEL] [MAX_WORKERS]
#SBATCH --job-name=gepa-val-eval
#SBATCH --partition=pi_ashia07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=gepa-val-eval-%j.out

set -euo pipefail

PROMPT=$1
DATASET=$2
REPETITION=$3
OUTPUT_LABEL=${4:-job21836315}
MAX_WORKERS=${5:-2}
if [[ ! "$OUTPUT_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "output label may contain only letters, numbers, underscores, and hyphens" >&2
    exit 2
fi
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYTHON=/usr/bin/python3.12
VLLM=/home/parmida/orcd/scratch/conda_envs/vllm-server/bin/vllm
TASK_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
API_BASE="http://127.0.0.1:${PORT}/v1"
SEED=$((41 + REPETITION))

export HF_HOME=/home/parmida/orcd/scratch/
export HUGGINGFACE_HUB_CACHE=/home/parmida/orcd/scratch/
export HF_HUB_CACHE=/home/parmida/orcd/scratch/
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/outputs"

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "$VLLM" serve "$TASK_MODEL" \
    --host 127.0.0.1 --port "$PORT" --max-model-len 32768 \
    > "$SCRIPT_DIR/logs/vllm-eval-${SLURM_JOB_ID}.out" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited during startup" >&2; exit 1
    fi
    sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

"$PYTHON" "$SCRIPT_DIR/evaluate_validation.py" "$PROMPT" \
    --dataset "$DATASET" --task-model "$TASK_MODEL" --api-base "$API_BASE" \
    --samples 8 --temperature 0.6 --top-p 0.95 --max-tokens 16384 \
    --max-workers "$MAX_WORKERS" --seed "$SEED" --cache-dir "$HF_HOME" \
    --output "$SCRIPT_DIR/outputs/${OUTPUT_LABEL}_on_${DATASET}_validation_rep${REPETITION}.json"
