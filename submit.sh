#!/usr/bin/env bash
# Run locally: ./submit.sh local [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size] [verbose] [run_tag] [dataset] [seed_prompt_file]
# Run on Slurm: ./submit.sh slurm [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size] [verbose] [run_tag] [dataset] [seed_prompt_file]
# Direct Slurm submission with `sbatch submit.sh slurm ...` also works.
# The budget is used only if run_GEPA.py is switched to
# --no-stop-after-coverage; coverage stopping is the default.
#SBATCH --job-name=gepa-math
#SBATCH --partition=pi_ashia07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=gepa-math-%j.out

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  ./submit.sh local [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size] [verbose] [run_tag] [dataset] [seed_prompt_file]
  ./submit.sh slurm [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size] [verbose] [run_tag] [dataset] [seed_prompt_file]

`local` runs immediately on the current machine. `slurm` submits this script
with sbatch, unless it is already running inside a Slurm allocation.
EOF
}

RUN_ENVIRONMENT=${1:-}
if [[ "$RUN_ENVIRONMENT" != "local" && "$RUN_ENVIRONMENT" != "slurm" ]]; then
    usage
    exit 2
fi
shift
if (( $# > 8 )); then
    usage
    exit 2
fi

MAX_BUDGET_CALLS=${1:-4000}
DO_MERGE=${2:-true}
TASK_MAX_TOKENS=${3:-16384}
REFLECTION_MINIBATCH_SIZE=${4:-8}
VERBOSE=${5:-false}
RUN_TAG=${6:-}
DATASET=${7:-mixed}
SEED_PROMPT_FILE=${8:-}

TASK_MODEL=${TASK_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REFLECTOR_MODEL=${REFLECTOR_MODEL:-openai/o4-mini}
REFLECTOR_MAX_TOKENS=${REFLECTOR_MAX_TOKENS:-32000}

if [[ "$DO_MERGE" != "true" && "$DO_MERGE" != "false" ]]; then
    echo "do_merge must be either true or false" >&2
    exit 2
fi
if [[ "$VERBOSE" != "true" && "$VERBOSE" != "false" ]]; then
    echo "verbose must be either true or false" >&2
    exit 2
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "run_tag may contain only letters, numbers, underscores, and hyphens" >&2
    exit 2
fi
if [[ "$DATASET" != "mixed" && "$DATASET" != "aime" && "$DATASET" != "amc" ]]; then
    echo "dataset must be mixed, aime, or amc" >&2
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required by the default reflection model" >&2
    exit 2
fi

# Resolve the project directory before Slurm potentially runs a copied script
# from its spool directory.
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Make the Slurm choice behave like the local choice: the user invokes this
# file directly and the script performs the scheduler submission. Calling it
# through sbatch explicitly is supported too.
if [[ "$RUN_ENVIRONMENT" == "slurm" && -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "slurm mode requires sbatch, but it was not found in PATH" >&2
        exit 127
    fi
    echo "Submitting GEPA pipeline to Slurm from $SOURCE_DIR"
    cd "$SOURCE_DIR"
    exec sbatch "$SOURCE_DIR/submit.sh" slurm "$@"
fi

# For a direct sbatch call, SLURM_SUBMIT_DIR points back to the directory from
# which the job was submitted. Local mode always uses this script's directory.
if [[ "$RUN_ENVIRONMENT" == "slurm" ]]; then
    SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$SOURCE_DIR}
    DEFAULT_PYTHON=/usr/bin/python3.12
    DEFAULT_VLLM=/home/parmida/orcd/scratch/conda_envs/vllm-server/bin/vllm
    CACHE_DIR=${CACHE_DIR:-/home/parmida/orcd/scratch/}
    HUB_CACHE_DIR=${HUB_CACHE_DIR:-$CACHE_DIR}
else
    SCRIPT_DIR=$SOURCE_DIR
    # This instance keeps the project environment next to the repository.
    # The PATH fallbacks make the launcher portable to another local machine.
    DEFAULT_PYTHON="$SOURCE_DIR/../Anaconda/envs/gepa/bin/python"
    DEFAULT_VLLM="$SOURCE_DIR/../Anaconda/envs/gepa/bin/vllm"
    CACHE_DIR=${CACHE_DIR:-${HF_HOME:-$SOURCE_DIR/../hf_cache}}
    HUB_CACHE_DIR=${HUB_CACHE_DIR:-$CACHE_DIR/hub}
fi

find_executable() {
    local candidate
    for candidate in "$@"; do
        if [[ "$candidate" == */* ]]; then
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        elif command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CANDIDATES=("$PYTHON_BIN")
else
    PYTHON_CANDIDATES=("$DEFAULT_PYTHON" python3.12 python3)
fi
if ! PYTHON_BIN=$(find_executable "${PYTHON_CANDIDATES[@]}"); then
    echo "Could not find Python. Set PYTHON_BIN to a Python 3 executable." >&2
    exit 127
fi

if [[ -n "${VLLM_BIN:-}" ]]; then
    VLLM_CANDIDATES=("$VLLM_BIN")
else
    VLLM_CANDIDATES=("$DEFAULT_VLLM" vllm)
fi
if ! VLLM_BIN=$(find_executable "${VLLM_CANDIDATES[@]}"); then
    echo "Could not find vLLM. Set VLLM_BIN to its executable path." >&2
    exit 127
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to check when vLLM is ready" >&2
    exit 127
fi

PORT=$("$PYTHON_BIN" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
API_BASE="http://127.0.0.1:${PORT}/v1"

CACHE_ARGS=()
if [[ -n "$CACHE_DIR" ]]; then
    export HF_HOME=${HF_HOME:-$CACHE_DIR}
    export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HUB_CACHE_DIR}
    export HF_HUB_CACHE=${HF_HUB_CACHE:-$HUB_CACHE_DIR}
    CACHE_ARGS=(--cache-dir "$CACHE_DIR")
fi

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/outputs"

RUN_ID=${SLURM_JOB_ID:-local-$$}
VLLM_LOG="$SCRIPT_DIR/logs/vllm-${RUN_ID}.out"
VLLM_ENV=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}")
if [[ "$RUN_ENVIRONMENT" == "slurm" ]]; then
    VLLM_ENV+=(PYTHONNOUSERSITE=1)
fi

echo "Running GEPA in $RUN_ENVIRONMENT mode"
echo "Python: $PYTHON_BIN"
echo "vLLM: $VLLM_BIN"
echo "vLLM log: $VLLM_LOG"

env "${VLLM_ENV[@]}" "$VLLM_BIN" serve "$TASK_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 32768 \
    > "$VLLM_LOG" 2>&1 &
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
    echo "vLLM failed to start; see $VLLM_LOG" >&2
    exit 1
fi

MERGE_FLAG=--do-merge
if [[ "$DO_MERGE" == "false" ]]; then
    MERGE_FLAG=--no-do-merge
fi
VERBOSE_FLAG=()
if [[ "$VERBOSE" == "true" ]]; then
    VERBOSE_FLAG=(--verbose)
fi
SEED_PROMPT_FLAG=()
if [[ -n "$SEED_PROMPT_FILE" ]]; then
    if [[ ! -f "$SEED_PROMPT_FILE" ]]; then
        echo "seed prompt file not found: $SEED_PROMPT_FILE" >&2
        exit 2
    fi
    SEED_PROMPT_FLAG=(--seed-prompt-file "$SEED_PROMPT_FILE")
fi

# Slashes are unsuitable inside a single output filename.
TASK_LABEL=${TASK_MODEL//\//_}
REFLECTOR_LABEL=${REFLECTOR_MODEL//\//_}
RUN_SUFFIX=${RUN_TAG:+_run${RUN_TAG}}
OUTPUT_FILE="$SCRIPT_DIR/outputs/optimized_prompt_${TASK_LABEL}_${REFLECTOR_LABEL}_dataset${DATASET}_coverage_merge${DO_MERGE}_taskmax${TASK_MAX_TOKENS}_tasktemp0p6_refltemp1p0_objavg4_minibatch_${REFLECTION_MINIBATCH_SIZE}${RUN_SUFFIX}.txt"

"$PYTHON_BIN" "$SCRIPT_DIR/run_GEPA.py" \
    --task-model "$TASK_MODEL" \
    --reflection-model "$REFLECTOR_MODEL" \
    --api-base "$API_BASE" \
    --budget "$MAX_BUDGET_CALLS" \
    --dataset "$DATASET" \
    "$MERGE_FLAG" \
    --task-max-tokens "$TASK_MAX_TOKENS" \
    --reflector-max-tokens "$REFLECTOR_MAX_TOKENS" \
    --reflection-minibatch-size "$REFLECTION_MINIBATCH_SIZE" \
    "${VERBOSE_FLAG[@]}" \
    "${SEED_PROMPT_FLAG[@]}" \
    "${CACHE_ARGS[@]}" \
    --output "$OUTPUT_FILE"
