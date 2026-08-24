"""Run GEPA prompt optimization on historical competition-math problems."""

from __future__ import annotations

import argparse
from pathlib import Path

from load_data import load_train_validation


SEED = 42
DEFAULT_TASK_MAX_TOKENS = 16_384
DEFAULT_REFLECTOR_MAX_TOKENS = 32_000
DEFAULT_SEED_PROMPT = "Solve the following math problem in less than {task_max_tokens} tokens"
TEMPERATURE = 0.9
TOP_P = 0.95
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name accepted by the OpenAI-compatible task-model server",
    )
    parser.add_argument(
        "--reflection-model",
        default='openai/o4-mini',
        help="LiteLLM model name used to improve the prompt (for example, openai/o4-mini)",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Task-model API URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument("--output", type=Path, default=Path("optimized_prompt.txt"))
    parser.add_argument(
        "--budget",
        type=int,
        default=4_000,
        help="Optimization calls, excluding the initial validation pass (default: 4000)",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--reflection-minibatch-size", type=int, default=32)
    parser.add_argument(
        "--do-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GEPA merge proposals (default: enabled)",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--task-max-tokens",
        type=int,
        default=DEFAULT_TASK_MAX_TOKENS,
        help=f"Maximum task-model response length (default: {DEFAULT_TASK_MAX_TOKENS})",
    )
    parser.add_argument(
        "--reflector-max-tokens",
        type=int,
        default=DEFAULT_REFLECTOR_MAX_TOKENS,
        help=(
            "Maximum reflection-model response length "
            f"(default: {DEFAULT_REFLECTOR_MAX_TOKENS})"
        ),
    )
    parser.add_argument(
        "--seed-prompt",
        default=None,
        help=(
            "Initial prompt given to GEPA (default: 'Solve the following math "
            "problem in less then {task_max_tokens} tokens')"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every reflector prompt and raw reflector response",
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    if args.seed_prompt is None:
        args.seed_prompt = DEFAULT_SEED_PROMPT.format(
            task_max_tokens=args.task_max_tokens
        )
    return args


def main() -> None:
    args = parse_args()
    if args.task_max_tokens < 1 or args.reflector_max_tokens < 1:
        raise ValueError("Token limits must be positive integers")

    # Keep the command-line help fast; these heavier imports are only needed
    # when an optimization run actually starts.
    import gepa
    from utils import (
        ExactBoxedMathAdapter,
        ReflectorTraceCallback,
        assert_local_vllm_is_running,
    )

    assert_local_vllm_is_running(args.api_base, DEFAULT_API_BASE)

    train, validation = load_train_validation(
        validation_fraction=args.validation_fraction,
        seed=SEED,
        cache_dir=args.cache_dir,
    )
    if args.budget < 1:
        raise ValueError("--budget must be a positive integer")

    print(f"Loaded {len(train)} training and {len(validation)} validation problems.")
    # GEPA includes its mandatory seed-candidate validation in max_metric_calls.
    # Add that fixed cost so the user-facing budget applies only to subsequent
    # optimization work.
    gepa_metric_call_limit = args.budget + len(validation)
    print(
        f"Optimization budget: {args.budget} calls, plus "
        f"{len(validation)} initial-validation calls.",
        flush=True,
    )
    print(f"First task-model input:\n{train[0]['input']}", flush=True)

    adapter = ExactBoxedMathAdapter(
        model=args.task_model,
        api_base=args.api_base,
        max_workers=args.max_workers,
        max_tokens=args.task_max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )
    seed_candidate = {"system_prompt": args.seed_prompt}

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=validation,
        adapter=adapter,
        reflection_lm=args.reflection_model,
        reflection_lm_kwargs={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": args.reflector_max_tokens,
            "seed": SEED,
        },
        reflection_minibatch_size=args.reflection_minibatch_size,
        batch_sampler="epoch_shuffled",
        use_merge=args.do_merge,
        max_metric_calls=gepa_metric_call_limit,
        cache_evaluation=True,
        val_evaluation_policy="full_eval",
        run_dir=str(args.output.with_suffix(".gepa_run")),
        use_cloudpickle=True,
        seed=SEED,
        display_progress_bar=True,
        callbacks=[ReflectorTraceCallback()] if args.verbose else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.best_candidate["system_prompt"] + "\n")
    print(f"Wrote the optimized prompt to {args.output}")


if __name__ == "__main__":
    main()
