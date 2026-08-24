#!/usr/bin/env bash
# Submit with: sbatch submit.sh [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size]
# Example:     sbatch submit.sh 4000 true 16384 32
#SBATCH --job-name=gepa-math
#SBATCH --partition=pi_ashia07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=gepa-math-%j.out

set -euo pipefail

# These four values can be supplied positionally to sbatch; otherwise the
# defaults below are used.
MAX_BUDGET_CALLS=${1:-4000}
DO_MERGE=${2:-true}
TASK_MAX_TOKENS=${3:-16384}
REFLECTION_MINIBATCH_SIZE=${4:-32}

TASK_MODEL=${TASK_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REFLECTOR_MODEL=${REFLECTOR_MODEL:-openai/o4-mini}
REFLECTOR_MAX_TOKENS=${REFLECTOR_MAX_TOKENS:-16384}

if [[ "$DO_MERGE" != "true" && "$DO_MERGE" != "false" ]]; then
    echo "do_merge must be either true or false" >&2
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required by the default reflection model" >&2
    exit 2
fi

# Slurm runs a copied script from its spool directory. SLURM_SUBMIT_DIR points
# back to the directory where the user invoked `sbatch submit.sh`.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYTHON=/usr/bin/python3.12
VLLM=/home/parmida/orcd/scratch/conda_envs/vllm-server/bin/vllm
PORT=$($PYTHON -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
API_BASE="http://127.0.0.1:${PORT}/v1"

export HF_HOME=/home/parmida/orcd/scratch/
export HUGGINGFACE_HUB_CACHE=/home/parmida/orcd/scratch/
export HF_HUB_CACHE=/home/parmida/orcd/scratch/

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/outputs"

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "$VLLM" serve "$TASK_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 32768 \
    > "$SCRIPT_DIR/logs/vllm-${SLURM_JOB_ID}.out" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait up to ten minutes for the model to load.
VLLM_READY=false
for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
        VLLM_READY=true
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
    fi
    sleep 5
done
if [[ "$VLLM_READY" != "true" ]]; then
    echo "vLLM failed to start; see logs/vllm-${SLURM_JOB_ID}.out" >&2
    exit 1
fi

MERGE_FLAG=--do-merge
if [[ "$DO_MERGE" == "false" ]]; then
    MERGE_FLAG=--no-do-merge
fi

# Slashes are unsuitable inside a single output filename.
TASK_LABEL=${TASK_MODEL//\//_}
REFLECTOR_LABEL=${REFLECTOR_MODEL//\//_}
OUTPUT_FILE="$SCRIPT_DIR/outputs/optimized_prompt_${TASK_LABEL}_${REFLECTOR_LABEL}_maxbudget${MAX_BUDGET_CALLS}_merge${DO_MERGE}_minibatch_${REFLECTION_MINIBATCH_SIZE}.txt"

"$PYTHON" "$SCRIPT_DIR/run_GEPA.py" \
    --task-model "$TASK_MODEL" \
    --reflection-model "$REFLECTOR_MODEL" \
    --api-base "$API_BASE" \
    --budget "$MAX_BUDGET_CALLS" \
    "$MERGE_FLAG" \
    --task-max-tokens "$TASK_MAX_TOKENS" \
    --reflector-max-tokens "$REFLECTOR_MAX_TOKENS" \
    --reflection-minibatch-size "$REFLECTION_MINIBATCH_SIZE" \
    --cache-dir "$HF_HOME" \
    --output "$OUTPUT_FILE"
