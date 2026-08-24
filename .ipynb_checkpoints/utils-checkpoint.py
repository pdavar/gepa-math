"""Small helpers used by the GEPA math experiment."""

from __future__ import annotations

import asyncio
import json
import os
import re
from urllib.error import URLError
from urllib.request import urlopen

from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsAdapter
from gepa.core.adapter import EvaluationBatch
from openai import AsyncOpenAI


_BOXED_ANSWER = re.compile(r"\\boxed\s*\{([^{}]*)\}")


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

    async def _generate(self, requests: list[list[dict[str, str]]]) -> list[str]:
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
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                seed=self.seed,
                                tools=[],
                            )
                            return result.choices[0].message.content or ""
                        except Exception as error:
                            if attempt == 2:
                                print(f"Task-model request failed: {error}", flush=True)
                                return ""
                            await asyncio.sleep(2**attempt)

            return await asyncio.gather(*(complete(request) for request in requests))

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        system_prompt = candidate["system_prompt"]
        requests = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["input"]},
            ]
            for example in batch
        ]
        responses = asyncio.run(self._generate(requests))
        scores = [
            float(exact_boxed_match(response, example["answer"]))
            for example, response in zip(batch, responses, strict=True)
        ]
        outputs = [
            {"full_assistant_response": response} for response in responses
        ]
        trajectories = None
        if capture_traces:
            trajectories = [
                {"data": example, "full_assistant_response": response}
                for example, response in zip(batch, responses, strict=True)
            ]
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )
