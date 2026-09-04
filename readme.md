# GEPA for Competition Math

This directory contains a small, standalone setup for optimizing a system
prompt on historical AIME and AMC 12 problems with GEPA.

## Files

- `run_GEPA.py`: runs prompt optimization.
- `load_data.py`: loads, filters, and splits the math datasets.
- `utils.py`: provides strict boxed-answer scoring, vLLM health checking, and
  optional reflector tracing.
- `evaluate_test.py`: evaluates an optimized prompt on the latest held-out
  AIME and AMC 12 sets.
- `submit.sh`: starts vLLM and runs GEPA as a Slurm job.
- `requirements.txt`: Python dependencies.

## Installation

```bash
python3.12 -m pip install -r requirements.txt
```

The default reflection model is `openai/o4-mini`, so set its API key before a
run:

```bash
export OPENAI_API_KEY="..."
```

## Run locally

First start the task model with vLLM:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
```

Then run GEPA in another terminal:

```bash
python3.12 run_GEPA.py \
  --output outputs/optimized_prompt.txt
```

The task-model endpoint defaults to `http://127.0.0.1:8000/v1`. The script
checks that a vLLM server is available there before starting.

Useful options:

```text
--task-model                 Task model served by vLLM
--reflection-model           LiteLLM name of the reflector model
--budget                     Optimization calls after initial validation
--do-merge / --no-do-merge   Enable or disable GEPA merge proposals
--task-max-tokens            Task-model output limit (default: 16384)
--reflector-max-tokens       Reflector output limit (default: 32000)
--reflection-minibatch-size  Problems used for each reflection (default: 32)
--seed-prompt                Initial prompt optimized by GEPA
--verbose                    Print reflector inputs and raw outputs
```

Task-model scoring uses temperature `0.6`, while reflector proposals use
temperature `1.0`. GEPA optimizes `avg@4`: four task-model responses are
generated per problem and their mean correctness is used as the score.
Reflection minibatches default to 32 problems and are balanced evenly between
AIME and AMC (16 of each).

After optimization, the selected prompt is freshly evaluated with eight
responses per validation problem. Overall and per-dataset `pass@1` and `avg@8`
are written to `*.validation_metrics.json`. This final evaluation is outside
the optimization-call budget.

The default seed prompt is generated from the task-model token limit:

```text
Solve the following math problem in less then 16384 tokens
```

`--budget N` reserves `N` calls for optimization. The mandatory initial
validation pass is counted separately and added to GEPA's internal limit.

## Submit with Slurm

Run this command from this directory:

```bash
sbatch submit.sh [max_budget_calls] [do_merge] [task_max_tokens] [reflection_minibatch_size] [verbose] [run_tag]
```

For example:

```bash
sbatch submit.sh 4000 true 16384 32 false rep1
```

The first five arguments have the defaults shown above; the run tag is optional.
`submit.sh` selects a free port,
starts vLLM, waits for it to become healthy, and stops it when the job exits.
Optimized prompts and resumable GEPA state are written under `outputs/`.
Use a distinct `run_tag` for independent replicas with otherwise identical
settings so their outputs and checkpoints do not collide.

The task and reflection models can be overridden through environment
variables:

```bash
TASK_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" \
REFLECTOR_MODEL="openai/o4-mini" \
sbatch submit.sh 4000 true 16384 32 true
```

## Evaluate an optimized prompt

`evaluate_test.py` evaluates AIME 2025, AIME 2026, AMC 12A 2025, and AMC 12B
2025. It generates eight responses per problem by default.

```bash
python3.12 evaluate_test.py \
  outputs/optimized_prompt_TASK_REFLECTOR_maxbudget4000_mergetrue_minibatch_32.txt
```

The full generations, correctness values, and `pass@1`, `avg@8`, and `pass@8`
metrics are saved beside the prompt as `*.test_results.json`. Use `--output`
to select another destination or `--samples` to change the number of responses
per problem.

## Data and scoring

Training uses historical AIME and AMC 12 problems before 2025. Problems that
contain or refer to figures are removed, and the remaining examples are
shuffled deterministically before the train/validation split.

A response is correct only when it contains exactly one `\boxed{...}` whose
contents match the expected integer or multiple-choice letter.
