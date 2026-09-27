import asyncio
import json
import unittest

import httpx
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from chess_intervention.task import TASK_INSTRUCTIONS
from chess_intervention.typesafe import TypeSafeAPI, TypeSafeError

OBSERVATION = {"fen": "8/8/8/8/8/8/8/K6k w - - 0 40", "decision_number": 3}


def reply(status: int, body: dict) -> httpx.MockTransport:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)
    transport.requests = requests
    return transport


def answer(choice: str) -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {"intervention": {"type": "choice", "choice": choice}},
        "usage": {"input_tokens": 120, "output_tokens": 3},
    }


class TypeSafeProvider(unittest.TestCase):
    def generate(self, transport: httpx.MockTransport):
        api = TypeSafeAPI("jev-1.13.0", api_key="secret-key")
        api.client = httpx.AsyncClient(transport=transport)
        messages = [
            ChatMessageSystem(content="instructions"),
            ChatMessageUser(content=json.dumps(OBSERVATION)),
        ]
        return asyncio.run(api.generate(messages, [], "none", GenerateConfig()))

    def test_sends_the_observation_and_returns_a_json_action(self):
        transport = reply(200, answer("use_expert_move"))
        output, call = self.generate(transport)
        sent = json.loads(transport.requests[0].content)
        self.assertEqual(sent["state"], OBSERVATION)
        self.assertEqual(sent["questions"]["intervention"]["instructions"], TASK_INSTRUCTIONS)
        self.assertEqual(transport.requests[0].headers["Authorization"], "Bearer secret-key")
        self.assertNotIn("secret-key", json.dumps(call.request))
        self.assertEqual(json.loads(output.completion), {"action": "use_expert_move"})
        self.assertEqual(output.usage.input_tokens, 120)

    def test_failures_are_errors_and_only_server_errors_are_retried(self):
        for status, body, retryable in (
            (503, {}, True),
            (429, {}, True),
            (400, {}, False),
            (200, answer("resign"), False),
        ):
            error, _ = self.generate(reply(status, body))
            self.assertIsInstance(error, TypeSafeError)
            self.assertEqual(error.retryable, retryable, status)


if __name__ == "__main__":
    unittest.main()
