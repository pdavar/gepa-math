"""Small helpers used by the GEPA math experiment."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsAdapter
from gepa.core.adapter import EvaluationBatch
from openai import AsyncOpenAI


_BOXED_ANSWER = re.compile(r"\\boxed\s*\{([^{}]*)\}")


class BalancedMathBatchSampler:
    """Sample balanced groups without replacement until all training IDs are seen."""

    def __init__(self, minibatch_size: int, seed: int = 42) -> None:
        if minibatch_size < 1:
            raise ValueError("Minibatch size must be positive")
        self.minibatch_size = minibatch_size
        self.seed = seed
        self._cached_loader_size = None
        self._aime_ids = []
        self._amc_ids = []
        self._aime_queue = []
        self._amc_queue = []
        self._rng = random.Random(seed)
        self.seen_ids = set()
        self.minibatches_sampled = 0
        self.metric_calls_at_full_coverage = None

    def _classify_ids(self, loader) -> None:
        all_ids = list(loader.all_ids())
        examples = loader.fetch(all_ids)
        self._aime_ids = [
            data_id
            for data_id, example in zip(all_ids, examples, strict=True)
            if example["input"].startswith("Put the final integer answer")
        ]
        self._amc_ids = [
            data_id
            for data_id, example in zip(all_ids, examples, strict=True)
            if example["input"].startswith("Put only the letter")
        ]
        if len(self._aime_ids) + len(self._amc_ids) != len(all_ids):
            raise ValueError("Every training problem must be identifiable as AIME or AMC")
        if self._aime_ids and self._amc_ids and self.minibatch_size % 2:
            raise ValueError("A balanced mixed-domain minibatch must have an even size")
        self._cached_loader_size = len(loader)
        self._aime_queue = []
        self._amc_queue = []
        self.seen_ids = set()
        self.minibatches_sampled = 0
        self.metric_calls_at_full_coverage = None

    def _draw(self, group_ids: list, queue: list, count: int) -> list:
        selected = []
        while len(selected) < count:
            if not queue:
                queue.extend(group_ids)
                self._rng.shuffle(queue)
            take = min(count - len(selected), len(queue))
            selected.extend(queue[:take])
            del queue[:take]
        return selected

    def next_minibatch_ids(self, loader, state) -> list:
        """Return a reproducible batch split as evenly as possible by task type."""
        if len(loader) != self._cached_loader_size:
            self._classify_ids(loader)

        if self._aime_ids and self._amc_ids:
            aime_count = self.minibatch_size // 2
            amc_count = self.minibatch_size // 2
        elif self._aime_ids:
            aime_count = self.minibatch_size
            amc_count = 0
        else:
            aime_count = 0
            amc_count = self.minibatch_size

        selected = self._draw(self._aime_ids, self._aime_queue, aime_count)
        selected += self._draw(self._amc_ids, self._amc_queue, amc_count)
        self._rng.shuffle(selected)
        self.seen_ids.update(selected)
        self.minibatches_sampled += 1
        return selected

    @property
    def total_examples(self) -> int:
        return len(self._aime_ids) + len(self._amc_ids)

    @property
    def coverage_complete(self) -> bool:
        return self.total_examples > 0 and len(self.seen_ids) == self.total_examples

    def coverage_stats(self) -> dict[str, int | float | bool | None]:
        total = self.total_examples
        return {
            "unique_training_examples_seen": len(self.seen_ids),
            "total_training_examples": total,
            "coverage_fraction": len(self.seen_ids) / total if total else 0.0,
            "coverage_complete": self.coverage_complete,
            "minibatches_sampled": self.minibatches_sampled,
            "minibatch_size": self.minibatch_size,
            "aime_examples": len(self._aime_ids),
            "amc_examples": len(self._amc_ids),
            "metric_calls_at_full_coverage": self.metric_calls_at_full_coverage,
        }


class TrainingCoverageStopper:
    """Stop GEPA immediately after the sampler has covered every training ID."""

    def __init__(self, sampler: BalancedMathBatchSampler) -> None:
        self.sampler = sampler

    def __call__(self, gepa_state) -> bool:
        complete = self.sampler.coverage_complete
        if complete and self.sampler.metric_calls_at_full_coverage is None:
            self.sampler.metric_calls_at_full_coverage = gepa_state.total_num_evals
        return complete


class OptimizationStatsCallback:
    """Persist coverage and prompt proposal/acceptance counts during a run."""

    def __init__(self, sampler: BalancedMathBatchSampler, output_path: Path) -> None:
        self.sampler = sampler
        self.output_path = output_path
        self.reflective_proposals = 0
        self.merge_proposals = 0
        self.accepted_prompts = 0
        self.last_metric_calls = 0

    def _report(self) -> dict[str, int | float | bool | None]:
        proposals = self.reflective_proposals + self.merge_proposals
        report = {
            **self.sampler.coverage_stats(),
            "reflective_prompts_proposed": self.reflective_proposals,
            "merge_prompts_proposed": self.merge_proposals,
            "prompts_proposed": proposals,
            "prompts_accepted": self.accepted_prompts,
            "acceptance_rate": self.accepted_prompts / proposals if proposals else 0.0,
            "current_metric_calls": self.last_metric_calls,
        }
        return report

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self._report(), indent=2) + "\n")

    def on_optimization_start(self, event) -> None:
        self._write()

    def on_proposal_end(self, event) -> None:
        self.reflective_proposals += 1
        self._write()

    def on_merge_attempted(self, event) -> None:
        self.merge_proposals += 1
        self._write()

    def on_candidate_accepted(self, event) -> None:
        self.accepted_prompts += 1
        self._write()

    def on_iteration_end(self, event) -> None:
        self.last_metric_calls = event["state"].total_num_evals
        self._write()

    def on_budget_updated(self, event) -> None:
        self.last_metric_calls = event["metric_calls_used"]
        self._write()

    def on_optimization_end(self, event) -> None:
        self.last_metric_calls = event["total_metric_calls"]
        self._write()

    def report(self) -> dict[str, int | float | bool | None]:
        return self._report()


class ReflectorTraceCallback:
    """Print each prompt sent to the reflector and its raw response."""

    def on_proposal_end(self, event) -> None:
        iteration = event["iteration"]
        prompts = event.get("prompts", {})
        responses = event.get("raw_lm_outputs", {})

        for component, prompt in prompts.items():
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt, indent=2)
            response = responses.get(component, "<no reflector response>")
            print(
                f"\n{'=' * 24} REFLECTOR TRACE: ITERATION {iteration} "
                f"({component}) {'=' * 24}\n"
                f"REFLECTOR INPUT:\n{prompt}\n\n"
                f"REFLECTOR OUTPUT:\n{response}\n"
                f"{'=' * 80}",
                flush=True,
            )


def assert_local_vllm_is_running(api_base: str, local_api_base: str) -> None:
    """Fail early when the configured local vLLM endpoint is unavailable."""
    if api_base.rstrip("/") != local_api_base.rstrip("/"):
        return

    models_url = f"{local_api_base.rstrip('/')}/models"
    try:
        with urlopen(models_url, timeout=5) as response:
            payload = json.load(response)
        server_is_running = response.status == 200 and "data" in payload
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        server_is_running = False

    assert server_is_running, (
        f"No vLLM server found at {local_api_base}. Start one first, for example:\n"
        "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000"
    )


def exact_boxed_match(response: str, expected_answer: str) -> bool:
    """Return True when the response contains exactly one correct boxed answer."""
    boxed_answers = [value.strip() for value in _BOXED_ANSWER.findall(response)]
    if len(boxed_answers) != 1:
        return False

    actual = boxed_answers[0]
    expected = expected_answer.strip()

    # AMC answers are letters; AIME answers are normally integers.
    if expected.upper() in {"A", "B", "C", "D", "E"}:
        return actual.upper() == expected.upper()
    try:
        return int(actual.replace(",", "")) == int(expected.replace(",", ""))
    except ValueError:
        return actual == expected


class ExactBoxedMathAdapter(AnyMathsAdapter):
    """GEPA's math adapter with strict boxed-answer scoring.

    The task model is called through an OpenAI-compatible API. This works with
    OpenAI itself and local servers such as vLLM.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None,
        max_workers: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        optimization_rollouts: int = 4,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            max_litellm_workers=max_workers,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.optimization_rollouts = optimization_rollouts

    async def _generate(
        self,
        requests: list[list[dict[str, str]]],
        rollouts: int,
    ) -> list[list[str]]:
        semaphore = asyncio.Semaphore(self.max_litellm_workers)
        client_options = {
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "timeout": 1800.0,
            "max_retries": 0,
        }
        if self.api_base:
            client_options["base_url"] = self.api_base

        async with AsyncOpenAI(**client_options) as client:

            async def complete(messages: list[dict[str, str]]) -> str:
                async with semaphore:
                    for attempt in range(3):
                        try:
                            result = await client.chat.completions.create(
                                model=self.model.removeprefix("openai/"),
                                messages=messages,
                                n=rollouts,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                seed=self.seed,
                            )
                            return [
                                choice.message.content or ""
                                for choice in result.choices
                            ]
                        except Exception as error:
                            if attempt == 2:
                                print(f"Task-model request failed: {error}", flush=True)
                                return [""] * rollouts
                            await asyncio.sleep(2**attempt)

            return await asyncio.gather(*(complete(request) for request in requests))

    def generate_rollouts(self, batch, candidate, rollouts: int) -> list[list[str]]:
        """Generate a requested number of responses for every problem."""
        system_prompt = candidate["system_prompt"]
        requests = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["input"]},
            ]
            for example in batch
        ]
        return asyncio.run(self._generate(requests, rollouts))

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        responses = self.generate_rollouts(
            batch, candidate, self.optimization_rollouts
        )
        rollout_scores = [
            [exact_boxed_match(response, example["answer"]) for response in samples]
            for example, samples in zip(batch, responses, strict=True)
        ]
        scores = [sum(problem_scores) / self.optimization_rollouts
                  for problem_scores in rollout_scores]
        rendered_responses = [
            "\n\n".join(
                f"ROLLOUT {index + 1}:\n{response}"
                for index, response in enumerate(samples)
            )
            for samples in responses
        ]
        outputs = [
            {"full_assistant_response": response}
            for response in rendered_responses
        ]
        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "data": example,
                    "full_assistant_response": rendered,
                    "rollout_scores": problem_scores,
                }
                for example, rendered, problem_scores in zip(
                    batch, rendered_responses, rollout_scores, strict=True
                )
            ]
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch) * self.optimization_rollouts,
        )

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ):
        """Tell the reflector how consistently each problem was answered."""
        if len(components_to_update) != 1:
            raise ValueError("The math adapter expects one prompt component")
        component = components_to_update[0]
        items = []
        for trajectory in eval_batch.trajectories or []:
            data = trajectory["data"]
            rollout_scores = trajectory["rollout_scores"]
            correct = sum(rollout_scores)
            total = len(rollout_scores)
            feedback = (
                f"{correct} of {total} rollouts were correct. "
                f"The correct answer is: {data['answer']}. Improve consistency "
                "while obeying the required boxed-answer format."
            )
            items.append(
                {
                    "Inputs": data["input"],
                    "Generated Outputs": trajectory["full_assistant_response"],
                    "Feedback": feedback,
                }
            )
        if not items:
            raise ValueError("No evaluation traces were available for reflection")
        return {component: items}
