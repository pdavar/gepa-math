"""Evaluate one fixed prompt on the deterministic AIME or AMC validation split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from load_data import AIME_INSTRUCTION, load_train_validation
from utils import assert_local_vllm_is_running, exact_boxed_match


DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_TASK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimized_prompt", type=Path)
    parser.add_argument("--dataset", choices=("aime", "amc"), required=True)
    parser.add_argument("--task-model", default=DEFAULT_TASK_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def generate(args: argparse.Namespace, rows: list[dict], prompt: str) -> list[list[str]]:
    from openai import AsyncOpenAI

    semaphore = asyncio.Semaphore(args.max_workers)
    async with AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=args.api_base,
        timeout=1_800.0,
        max_retries=2,
    ) as client:
        async def one(row: dict) -> list[str]:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=args.task_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": row["input"]},
                    ],
                    n=args.samples,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                return [choice.message.content or "" for choice in response.choices]

        return await asyncio.gather(*(one(row) for row in rows))


def main() -> None:
    args = parse_args()
    if not args.optimized_prompt.is_file():
        raise FileNotFoundError(args.optimized_prompt)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    assert_local_vllm_is_running(args.api_base, DEFAULT_API_BASE)

    _, validation = load_train_validation(seed=42, cache_dir=args.cache_dir)
    want_aime = args.dataset == "aime"
    rows = [
        row for row in validation
        if row["input"].startswith(AIME_INSTRUCTION) == want_aime
    ]
    prompt = args.optimized_prompt.read_text().strip()
    responses = asyncio.run(generate(args, rows, prompt))
    scores = [
        [exact_boxed_match(text, row["answer"]) for text in samples]
        for row, samples in zip(rows, responses, strict=True)
    ]
    metrics = {
        "problems": len(rows),
        "pass@1": sum(x[0] for x in scores) / len(scores),
        f"avg@{args.samples}": sum(sum(x) / args.samples for x in scores) / len(scores),
    }
    report = {
        "dataset": args.dataset,
        "optimized_prompt": str(args.optimized_prompt.resolve()),
        "task_model": args.task_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "samples_per_problem": args.samples,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
