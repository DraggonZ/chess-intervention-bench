"""Inspect model provider for TypeSafe's Jev models, used as ``typesafe/jev-<version>``.

Jev is not a chat model: its API takes a JSON state and a choice question and
returns one of the listed options. This provider sends the supervisor's
observation as the state and the task instructions as the question, then
returns the chosen action as the same JSON answer chat models give, so the
task's solver and scorer treat every model alike.
"""

from __future__ import annotations

import json
import os
import time

import httpx
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    modelapi,
)

from .task import ACTION_DESCRIPTIONS, ACTIONS, TASK_INSTRUCTIONS

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TIMEOUT_SECONDS = 60


class TypeSafeError(Exception):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class TypeSafeAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008 - Inspect's signature
        **model_args: object,
    ) -> None:
        super().__init__(model_name, base_url or ENDPOINT, api_key, ["TYPESAFE_API_KEY"], config)
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

    async def aclose(self) -> None:
        await self.client.aclose()

    def should_retry(self, ex: BaseException) -> bool:
        return isinstance(ex, httpx.TransportError) or (
            isinstance(ex, TypeSafeError) and ex.retryable
        )

    async def generate(self, input: list[ChatMessage], tools, tool_choice, config):
        users = [message for message in input if isinstance(message, ChatMessageUser)]
        if tools or len(users) != 1:
            raise ValueError("Jev takes exactly one observation message and no tools")
        payload = {
            "model": self.model_name,
            "state": json.loads(users[0].text),
            "questions": {
                "intervention": {
                    "type": "choice",
                    "instructions": TASK_INSTRUCTIONS,
                    "criteria": ACTION_DESCRIPTIONS,
                }
            },
        }
        if not self.api_key:
            raise TypeSafeError("TYPESAFE_API_KEY is not set")
        start = time.perf_counter()
        response = await self.client.post(
            self.base_url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        call = ModelCall.create(request=payload, response=_json_or_text(response))
        call.time = time.perf_counter() - start
        if response.status_code != 200:
            retryable = response.status_code == 429 or response.status_code >= 500
            return TypeSafeError(f"HTTP {response.status_code}", retryable), call
        body = response.json()
        choice = body.get("answers", {}).get("intervention", {}).get("choice")
        if choice not in ACTIONS:
            return TypeSafeError(f"unexpected answer: {body.get('answers')!r}"), call
        output = ModelOutput.from_content(self.model_name, json.dumps({"action": choice}))
        usage = body.get("usage", {})
        output.usage = ModelUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        )
        return output, call


def _json_or_text(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {"text": response.text[:2000]}
    return body if isinstance(body, dict) else {"body": body}


@modelapi(name="typesafe")
def typesafe() -> type[ModelAPI]:
    return TypeSafeAPI
