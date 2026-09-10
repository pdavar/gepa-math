"""Evaluate an optimized GEPA prompt on the latest AIME and AMC test sets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from load_data import TEST_BENCHMARKS, load_test_benchmark
from utils import assert_local_vllm_is_running, exact_boxed_match


DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_TASK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEED = 42
MAX_TOKENS = 16_384
TEMPERATURE = 0.9
TOP_P = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "optimized_prompt",
        type=Path,
        help="Path to optimized_prompt_...txt produced by run_GEPA.py",
    )
    parser.add_argument("--task-model", default=DEFAULT_TASK_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON destination (default: beside the prompt file)",
    )
    return parser.parse_args()


async def generate_responses(
    rows: list[dict[str, str]],
    system_prompt: str,
    task_model: str,
    api_base: str,
    samples: int,
    max_workers: int,
) -> list[list[str]]:
    """Generate all samples through an OpenAI-compatible vLLM server."""
    from openai import AsyncOpenAI

    semaphore = asyncio.Semaphore(max_workers)
    async with AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=api_base,
        timeout=1_800.0,
        max_retries=2,
    ) as client:

        async def generate_one(row: dict[str, str]) -> list[str]:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=task_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["input"]},
                    ],
                    n=samples,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=SEED,
                )
                return [choice.message.content or "" for choice in response.choices]

        return await asyncio.gather(*(generate_one(row) for row in rows))


def summarize(scores: list[list[bool]]) -> dict[str, float]:
    """Compute the same pass@1, average, and pass@k metrics as the baseline."""
    number_of_problems = len(scores)
    samples = len(scores[0])
    return {
        "pass@1": sum(problem[0] for problem in scores) / number_of_problems,
        f"avg@{samples}": sum(sum(problem) / samples for problem in scores)
        / number_of_problems,
        f"pass@{samples}": sum(any(problem) for problem in scores)
        / number_of_problems,
    }


def main() -> None:
    args = parse_args()
    if not args.optimized_prompt.is_file():
        raise FileNotFoundError(f"Optimized prompt not found: {args.optimized_prompt}")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")

    assert_local_vllm_is_running(args.api_base, DEFAULT_API_BASE)
    system_prompt = args.optimized_prompt.read_text().strip()
    if not system_prompt:
        raise ValueError(f"Optimized prompt is empty: {args.optimized_prompt}")

    report = {
        "task_model": args.task_model,
        "optimized_prompt": str(args.optimized_prompt.resolve()),
        "samples_per_problem": args.samples,
        "benchmarks": {},
    }
    for benchmark in TEST_BENCHMARKS:
        rows = load_test_benchmark(benchmark, cache_dir=args.cache_dir)
        generations = asyncio.run(
            generate_responses(
                rows,
                system_prompt,
                args.task_model,
                args.api_base,
                args.samples,
                args.max_workers,
            )
        )
        scores = [
            [exact_boxed_match(response, row["answer"]) for response in responses]
            for row, responses in zip(rows, generations, strict=True)
        ]
        report["benchmarks"][benchmark] = {
            "metrics": summarize(scores),
            "problems": [
                {
                    **row,
                    "generations": responses,
                    "correct": problem_scores,
                }
                for row, responses, problem_scores in zip(
                    rows, generations, scores, strict=True
                )
            ],
        }

    output = args.output or args.optimized_prompt.with_suffix(".test_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    summary = {
        benchmark: result["metrics"]
        for benchmark, result in report["benchmarks"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote full results to {output}")


if __name__ == "__main__":
    main()
