"""Load and prepare historical AIME and AMC 12 problems for GEPA."""

from __future__ import annotations

import re

from datasets import load_dataset


AIME_INSTRUCTION = "Put the final integer answer in exactly one \\boxed{...}."
AMC_INSTRUCTION = (
    "Put only the letter of the correct choice (A, B, C, D, or E) "
    "in exactly one \\boxed{...}."
)

# Problems that contain or refer to a visual are unsuitable for a text-only run.
VISUAL_REFERENCE = re.compile(
    r"(?is)(\[asy\]|```\s*asy|\\begin\s*\{asy\}|<img|"
    r"\bfigure\b|\bdiagram\b|\bpicture\b|\billustration\b|"
    r"(?:shown|depicted|pictured|illustrated)\s+(?:above|below))"
)

TEST_BENCHMARKS = ("aime2026", "aime2025", "amc12a2025", "amc12b2025")


def _format_problem(problem: str, is_amc: bool) -> str:
    instruction = AMC_INSTRUCTION if is_amc else AIME_INSTRUCTION
    return f"{instruction}\n\nProblem: {problem}"


def load_math_data(cache_dir: str | None = None) -> list[dict[str, str]]:
    """Return text-only historical problems, excluding the 2025 test year."""
    aime_dataset = load_dataset(
        "Pandores/aime-1983-2025", split="train", cache_dir=cache_dir
    )
    amc_dataset = load_dataset(
        "edev2000/amc12-full", split="train", cache_dir=cache_dir
    )

    examples: list[dict[str, str]] = []
    for row in aime_dataset:
        problem = str(row["problem"])
        if int(row["year"]) < 2025 and not VISUAL_REFERENCE.search(problem):
            examples.append(
                {
                    "input": _format_problem(problem, is_amc=False),
                    "answer": str(row["answer"]).strip(),
                    "additional_context": {},
                }
            )

    for row in amc_dataset:
        problem = str(row["question"])
        if not str(row["problem_id"]).startswith("2025") and not VISUAL_REFERENCE.search(problem):
            examples.append(
                {
                    "input": _format_problem(problem, is_amc=True),
                    "answer": str(row["answer"]).strip(),
                    "additional_context": {},
                }
            )

    if not examples:
        raise RuntimeError("No math problems remained after filtering")
    return examples


def load_train_validation(
    validation_fraction: float = 0.10,
    seed: int = 42,
    cache_dir: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Make a reproducible shuffled train/validation split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    # Hugging Face Dataset supplies a simple deterministic shuffle.
    from datasets import Dataset

    shuffled = Dataset.from_list(load_math_data(cache_dir)).shuffle(seed=seed)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    split_at = len(shuffled) - validation_size
    return list(shuffled.select(range(split_at))), list(shuffled.select(range(split_at, len(shuffled))))


def load_test_benchmark(
    benchmark: str, cache_dir: str | None = None
) -> list[dict[str, str]]:
    """Load one of the latest held-out AIME or AMC 12 test sets."""
    if benchmark not in TEST_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    rows: list[dict[str, str]] = []
    if benchmark == "aime2026":
        try:
            raw = load_dataset("math-ai/aime26", split="train", cache_dir=cache_dir)
        except (ValueError, KeyError):
            raw = load_dataset("math-ai/aime26", split="test", cache_dir=cache_dir)
        for index, row in enumerate(raw):
            problem = str(row.get("problem", row.get("question")))
            if not VISUAL_REFERENCE.search(problem):
                rows.append(
                    {
                        "id": str(row.get("id") or f"aime2026-{index + 1}"),
                        "benchmark": benchmark,
                        "input": _format_problem(problem, is_amc=False),
                        "answer": str(row["answer"]).strip(),
                    }
                )
    elif benchmark == "aime2025":
        raw = load_dataset("Pandores/aime-1983-2025", split="train", cache_dir=cache_dir)
        for index, row in enumerate(raw):
            problem = str(row["problem"])
            if int(row["year"]) == 2025 and not VISUAL_REFERENCE.search(problem):
                rows.append(
                    {
                        "id": str(row.get("id") or f"aime2025-{index + 1}"),
                        "benchmark": benchmark,
                        "input": _format_problem(problem, is_amc=False),
                        "answer": str(row["answer"]).strip(),
                    }
                )
    else:
        wanted_exam = "A" if benchmark == "amc12a2025" else "B"
        raw = load_dataset("edev2000/amc12-full", split="train", cache_dir=cache_dir)
        for row in raw:
            problem_id = str(row.get("problem_id", row.get("id")))
            compact_id = re.sub(r"[^A-Z0-9]", "", problem_id.upper())
            matches_exam = re.match(
                rf"^2025(?:AMC12)?{wanted_exam}(?:P)?\d+$", compact_id
            )
            problem = str(row.get("question", row.get("problem")))
            if matches_exam and not VISUAL_REFERENCE.search(problem):
                rows.append(
                    {
                        "id": problem_id,
                        "benchmark": benchmark,
                        "input": _format_problem(problem, is_amc=True),
                        "answer": str(row["answer"]).strip().upper(),
                    }
                )

    if not rows:
        raise RuntimeError(f"No text-only problems found for {benchmark}")
    print(f"{benchmark}: loaded {len(rows)} text-only problems", flush=True)
    return rows
