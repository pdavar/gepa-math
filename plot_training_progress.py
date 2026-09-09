"""Plot best-so-far GEPA avg@4 validation accuracy versus metric calls."""

from __future__ import annotations

import glob
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt


RUNS = {
    "AIME-only": "*datasetaime*single_aime_rep2.gepa_run/gepa_state.bin",
    "AMC-only": "*datasetamc*single_amc_rep6.gepa_run/gepa_state.bin",
    "both": "*objavg4_minibatch_3_runavg4_b3_mfalse_rep6.gepa_run/gepa_state.bin",
    "AMC with AIME-seed": "*datasetamc*amc_customseed_rep6.gepa_run/gepa_state.bin",
    "AIME with AMC-seed": "*datasetaime*aime_customseed_rep3.gepa_run/gepa_state.bin",
}


def load_curve(pattern: str) -> tuple[list[int], list[float]]:
    matches = glob.glob(str(Path("outputs") / pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one GEPA state for {pattern!r}, found {matches}")
    with open(matches[0], "rb") as file:
        state = cloudpickle.load(file)

    calls = state["num_metric_calls_by_discovery"]
    candidate_scores = [
        sum(subscores.values()) / len(subscores)
        for subscores in state["prog_candidate_val_subscores"]
    ]
    if len(calls) != len(candidate_scores):
        raise RuntimeError(f"Mismatched calls and scores in {matches[0]}")

    best_so_far: list[float] = []
    best = float("-inf")
    for score in candidate_scores:
        best = max(best, score)
        best_so_far.append(100 * best)
    return calls, best_so_far


def style_axes(ax) -> None:
    ax.set_xlabel("Metric calls at candidate discovery")
    ax.set_ylabel("Best validation avg@4 accuracy (%)")
    ax.grid(True, alpha=0.25)


def main() -> None:
    output_dir = Path("plots/training_progress")
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = {name: load_curve(pattern) for name, pattern in RUNS.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (calls, scores) in curves.items():
        ax.step(calls, scores, where="post", linewidth=2, label=name)
        ax.scatter(calls, scores, s=18)
    style_axes(ax)
    ax.set_title("GEPA training progress by experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=200)
    plt.close(fig)

    for name, (calls, scores) in curves.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(calls, scores, where="post", linewidth=2)
        ax.scatter(calls, scores, s=28)
        style_axes(ax)
        ax.set_title(name)
        fig.tight_layout()
        filename = name.lower().replace(" ", "_").replace("-", "_") + ".png"
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    print(f"Wrote comparison and individual plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
