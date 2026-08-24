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
