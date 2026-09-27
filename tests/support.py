"""Test helpers: a fake engine, short games, a scripted model and Hypothesis strategies."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from unittest import mock

import chess
from hypothesis import strategies as st
from inspect_ai.model import Model, ModelOutput, get_model

from chess_intervention.episodes import Episode, Game, Intervention

CONTINUE = '{"action": "continue_play"}'
INTERVENE = '{"action": "use_expert_move"}'


class HashSearcher:
    """Deterministic fake engine: the move is a hash of the game history and depth.

    About half the time the deeper search agrees with the shallow one, so
    episodes contain both changed and unchanged interventions.
    """

    def __init__(self, worker_depth: int = 12) -> None:
        self.worker_depth = worker_depth

    def best_move(self, board: chess.Board, depth: int) -> str:
        moves = sorted(move.uci() for move in board.legal_moves)
        history = " ".join(move.uci() for move in board.move_stack)
        digest = hashlib.sha256(f"{history}|{depth}".encode()).digest()
        if depth != self.worker_depth and digest[1] % 2 == 0:
            return self.best_move(board, self.worker_depth)
        return moves[digest[0] % len(moves)]


def short_games() -> ExitStack:
    """Cap games at 24 engine plies and interventions at move 14, so tests stay fast."""
    stack = ExitStack()
    stack.enter_context(mock.patch("chess_intervention.episodes.MAX_GENERATED_PLIES", 24))
    stack.enter_context(mock.patch("chess_intervention.episodes.LAST_INTERVENTION_MOVE", 14))
    return stack


def scripted_model(answers: dict[str, list[str]]) -> Model:
    """A mock model that answers by the game it is shown.

    ``answers`` maps an observation's move history to the replies given there,
    in order. Any other decision, and one whose replies are used up, gets
    continue_play.
    """

    def reply(input, tools, tool_choice, config) -> ModelOutput:
        queue = answers.get(json.loads(input[-1].text)["move_history"], [])
        return ModelOutput.from_content("mockllm/model", queue.pop(0) if queue else CONTINUE)

    return get_model("mockllm/model", custom_outputs=reply)


@st.composite
def openings(draw, max_plies: int = 12) -> tuple[str, ...]:
    """A legal move sequence from the start that does not end the game."""
    board = chess.Board()
    for choice in draw(st.lists(st.integers(0, 10_000), max_size=max_plies)):
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        candidate = board.copy()
        candidate.push(moves[choice % len(moves)])
        if candidate.is_game_over() or candidate.is_repetition(3):
            break
        board = candidate
    return tuple(move.uci() for move in board.move_stack)


RESULTS = ("1-0", "0-1", "1/2-1/2")


@st.composite
def synthetic_episodes(draw) -> Episode:
    """Episodes with arbitrary results and no chess content, for scoring tests.

    As in real episodes, an unchanged Expert move keeps the unassisted result.
    """
    baseline_result = draw(st.sampled_from(RESULTS))
    first_move = draw(st.integers(10, 25))
    interventions = []
    for turn in range(draw(st.integers(0, 12))):
        changed = draw(st.booleans())
        interventions.append(
            Intervention(
                ply=2 * turn,
                fullmove=first_move + turn,
                worker_move="a2a3",
                expert_move="a2a4" if changed else "a2a3",
                continuation=("a2a4",) if changed else None,
                result=draw(st.sampled_from(RESULTS)) if changed else baseline_result,
                termination="test",
            )
        )
    return Episode(
        id="synthetic",
        source={"game_id": "synthetic"},
        opening=(),
        assisted_color=draw(st.sampled_from(("white", "black"))),
        baseline=Game((), baseline_result, "test"),
        interventions=tuple(interventions),
    )
