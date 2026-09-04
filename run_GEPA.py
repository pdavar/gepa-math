"""Run GEPA prompt optimization on historical competition-math problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from load_data import load_train_validation


SEED = 42
DEFAULT_TASK_MAX_TOKENS = 16_384
DEFAULT_REFLECTOR_MAX_TOKENS = 32_000
DEFAULT_SEED_PROMPT = "Solve the following math problem in less than {task_max_tokens} tokens"
TASK_TEMPERATURE = 0.6
REFLECTOR_TEMPERATURE = 1.0
OPTIMIZATION_ROLLOUTS = 4
FINAL_VALIDATION_ROLLOUTS = 8
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
    parser.add_argument(
        "--dataset",
        choices=("mixed", "aime", "amc"),
        default="mixed",
        help="Problem domain used for both training and validation (default: mixed)",
    )
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
        BalancedMathBatchSampler,
        ReflectorTraceCallback,
        assert_local_vllm_is_running,
        exact_boxed_match,
    )

    assert_local_vllm_is_running(args.api_base, DEFAULT_API_BASE)

    train, validation = load_train_validation(
        validation_fraction=args.validation_fraction,
        seed=SEED,
        cache_dir=args.cache_dir,
    )
    if args.dataset != "mixed":
        prefix = (
            "Put the final integer answer"
            if args.dataset == "aime"
            else "Put only the letter"
        )
        train = [row for row in train if row["input"].startswith(prefix)]
        validation = [row for row in validation if row["input"].startswith(prefix)]
        if not train or not validation:
            raise RuntimeError(f"No {args.dataset.upper()} data remained after splitting")
    if args.budget < 1:
        raise ValueError("--budget must be a positive integer")

    print(f"Loaded {len(train)} training and {len(validation)} validation problems.")
    # Each problem now costs four metric calls because the optimization score
    # is the mean correctness across four independently sampled responses.
    initial_validation_calls = len(validation) * OPTIMIZATION_ROLLOUTS
    gepa_metric_call_limit = args.budget + initial_validation_calls
    print(
        f"Optimization budget: {args.budget} calls, plus "
        f"{initial_validation_calls} initial-validation calls; "
        f"objective=avg@{OPTIMIZATION_ROLLOUTS}.",
        flush=True,
    )
    print(f"First task-model input:\n{train[0]['input']}", flush=True)

    adapter = ExactBoxedMathAdapter(
        model=args.task_model,
        api_base=args.api_base,
        max_workers=args.max_workers,
        max_tokens=args.task_max_tokens,
        temperature=TASK_TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
        optimization_rollouts=OPTIMIZATION_ROLLOUTS,
    )
    seed_candidate = {"system_prompt": args.seed_prompt}
    batch_sampler = "epoch_shuffled"
    reflection_minibatch_size = args.reflection_minibatch_size
    if args.dataset == "mixed":
        batch_sampler = BalancedMathBatchSampler(
            minibatch_size=args.reflection_minibatch_size,
            seed=SEED,
        )
        reflection_minibatch_size = None

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=validation,
        adapter=adapter,
        reflection_lm=args.reflection_model,
        reflection_lm_kwargs={
            "temperature": REFLECTOR_TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": args.reflector_max_tokens,
            "seed": SEED,
        },
        # The custom sampler already owns the minibatch size. GEPA requires
        # this separate argument to be None when batch_sampler is an object.
        reflection_minibatch_size=reflection_minibatch_size,
        batch_sampler=batch_sampler,
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

    # This is a fresh post-training evaluation and is intentionally outside
    # the GEPA optimization budget.
    final_responses = adapter.generate_rollouts(
        validation,
        result.best_candidate,
        FINAL_VALIDATION_ROLLOUTS,
    )
    per_problem = [
        [exact_boxed_match(response, example["answer"]) for response in samples]
        for example, samples in zip(validation, final_responses, strict=True)
    ]

    def metrics(indices: list[int]) -> dict[str, float | int]:
        return {
            "problems": len(indices),
            "pass@1": sum(per_problem[index][0] for index in indices) / len(indices),
            "avg@8": (
                sum(sum(per_problem[index]) / FINAL_VALIDATION_ROLLOUTS for index in indices)
                / len(indices)
            ),
        }

    aime_indices = [
        index for index, row in enumerate(validation)
        if row["input"].startswith("Put the final integer answer")
    ]
    amc_indices = [
        index for index, row in enumerate(validation)
        if row["input"].startswith("Put only the letter")
    ]
    validation_report = {
        "dataset": args.dataset,
        "optimization_objective": "avg@4",
        "task_temperature": TASK_TEMPERATURE,
        "reflector_temperature": REFLECTOR_TEMPERATURE,
        "overall": metrics(list(range(len(validation)))),
    }
    if aime_indices:
        validation_report["aime"] = metrics(aime_indices)
    if amc_indices:
        validation_report["amc"] = metrics(amc_indices)
    metrics_path = args.output.with_suffix(".validation_metrics.json")
    metrics_path.write_text(json.dumps(validation_report, indent=2) + "\n")
    print("FINAL VALIDATION METRICS:\n" + json.dumps(validation_report, indent=2))
    print(f"Wrote validation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
