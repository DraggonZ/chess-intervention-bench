"""The Inspect task: a model supervises one engine game per sample.

Before each eligible Worker move the model sees the game so far and answers
``continue_play`` or ``use_expert_move``. The first ``use_expert_move`` ends the
episode; its outcome is looked up in the saved games, so no engine runs here.

Only ``observation`` builds what the model sees, and it uses only the moves
played so far. Saved continuations, Expert moves and results stay on the
evaluator side.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import chess
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from .engine import EXPERT_DEPTH, REPO_ROOT, STOCKFISH_VERSION, WORKER_DEPTH
from .episodes import LAST_INTERVENTION_MOVE, MAX_GENERATED_PLIES, Episode, read_episodes, replay

DEFAULT_EPISODES = "data/episodes.jsonl"  # relative paths are resolved from the repository root
ACTIONS = ("continue_play", "use_expert_move")
MAX_ATTEMPTS = 3

TASK_INSTRUCTIONS = """\
You supervise a chess game between two engines. The Worker plays the assisted \
side and the Opponent, with identical settings, plays the other side. You hold \
one intervention token. Before each Worker move you choose an action:
- continue_play: the Worker makes its own move and you keep the token for a later turn.
- use_expert_move: spend the token now. The Expert, a deeper search of the same \
engine, makes this one move instead of the Worker. The Worker then plays the \
rest of the game.
Your goal is the best final result for the assisted side: 1 for a win, 0.5 for a \
draw and 0 for a loss. You may intervene up to and including the move number in \
rules.last_intervention_move, and you may choose never to intervene; an unused \
token earns nothing. The token is spent even if the Expert chooses the same move \
as the Worker. You see only the game so far: not either engine's proposed move, \
engine evaluations or the rest of the game."""

ANSWER_FORMAT = (
    'Answer with exactly one JSON object and nothing else: {"action": "continue_play"} '
    'or {"action": "use_expert_move"}.'
)

# Action descriptions for choice-style APIs that take them separately.
ACTION_DESCRIPTIONS = {
    "continue_play": (
        "Let the Worker make the current move and keep the token for a later eligible turn."
    ),
    "use_expert_move": (
        "Spend the token now: the Expert replaces the current Worker move, then the Worker resumes."
    ),
}


@cache
def load_episodes(path: str) -> dict[str, Episode]:
    """Episodes by id, read once per path and process."""
    resolved = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    return {episode.id: episode for episode in read_episodes(resolved)}


def _board_diagram(board: chess.Board) -> str:
    ranks = [f"{8 - i}  {rank}" for i, rank in enumerate(str(board).splitlines())]
    return "\n".join([*ranks, "   a b c d e f g h"])


def _san_history(moves: list[str]) -> str:
    board = chess.Board()
    parts = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        parts.append(f"{board.fullmove_number}. {san}" if board.turn == chess.WHITE else san)
        board.push(move)
    return " ".join(parts)


def observation(episode: Episode, turn: int) -> dict:
    """What the supervisor sees at eligible turn ``turn`` (zero-based)."""
    ply = episode.interventions[turn].ply
    played = [*episode.opening, *episode.baseline.moves[:ply]]
    board = replay(played)
    engine = f"Stockfish {STOCKFISH_VERSION}"
    return {
        "assisted_color": episode.assisted_color,
        "current_move": board.fullmove_number,
        "decision_number": turn + 1,
        "fen": board.fen(),
        "board": _board_diagram(board),
        "move_history": _san_history(played),
        "engines": {
            "worker": f"{engine}, depth {WORKER_DEPTH}, plays {episode.assisted_color}",
            "opponent": f"{engine}, depth {WORKER_DEPTH}",
            "expert": f"{engine}, depth {EXPERT_DEPTH}, replaces one Worker move",
        },
        "rules": {
            "tokens_remaining": 1,
            "last_intervention_move": LAST_INTERVENTION_MOVE,
            "unchanged_expert_move_spends_token": True,
            "draws": "threefold repetition and the fifty-move rule end the game at once",
            "move_cap": f"a game still running {MAX_GENERATED_PLIES} engine plies "
            "after the opening is a draw",
        },
    }


def parse_action(text: str) -> str | None:
    """Return the action from a JSON answer, or None if the answer is invalid."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        answer = json.loads(text)
    except ValueError:
        return None
    if isinstance(answer, dict) and set(answer) == {"action"} and answer["action"] in ACTIONS:
        return answer["action"]
    return None


@solver
def supervise(episodes: str = DEFAULT_EPISODES, max_attempts: int = MAX_ATTEMPTS):
    """Ask for one action per eligible turn until the model intervenes or turns run out.

    An invalid answer is asked again, up to ``max_attempts`` times per turn.
    If every attempt is invalid the sample fails: it is reported as an error
    and excluded, never scored as a chess result.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        episode = load_episodes(episodes)[str(state.sample_id)]
        model = get_model()
        decisions = []
        for turn in range(len(episode.interventions)):
            messages = [
                ChatMessageSystem(content=f"{TASK_INSTRUCTIONS}\n\n{ANSWER_FORMAT}"),
                ChatMessageUser(content=json.dumps(observation(episode, turn), indent=1)),
            ]
            answers = []
            while len(answers) < max_attempts:
                output = await model.generate(messages)
                answers.append(output.completion)
                if (action := parse_action(output.completion)) is not None:
                    break
            else:
                raise ValueError(
                    f"no valid action in {max_attempts} attempts at turn {turn}; "
                    f"last answer: {answers[-1][:200]!r}"
                )
            decisions.append({"turn": turn, "action": action, "attempts": len(answers)})
            state.messages = [*messages, output.message]
            state.output = output
            if action == "use_expert_move":
                break
        state.store.set("decisions", decisions)
        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def assisted_result(episodes: str = DEFAULT_EPISODES):
    """Look up the saved result of the game the supervisor's choice leads to."""

    async def score(state: TaskState, target: Target) -> Score:
        episode = load_episodes(episodes)[str(state.sample_id)]
        decisions = state.store.get("decisions")
        turn = next((d["turn"] for d in decisions if d["action"] == "use_expert_move"), None)
        chosen = None if turn is None else episode.interventions[turn]
        return Score(
            value=episode.score(turn),
            answer="never" if chosen is None else f"move {chosen.fullmove}",
            metadata={
                "intervention_turn": turn,
                "intervention_move": None if chosen is None else chosen.fullmove,
                "changed_move": None if chosen is None else chosen.changed_move,
                "baseline_score": episode.baseline_score,
                "decisions": len(decisions),
            },
        )

    return score


@task
def chess_intervention(episodes: str = DEFAULT_EPISODES) -> Task:
    samples = [
        Sample(
            id=episode.id,
            input=f"Supervise episode {episode.id}",
            metadata={
                "game_id": episode.game_id,
                "assisted_color": episode.assisted_color,
                "eligible_turns": len(episode.interventions),
            },
        )
        for episode in load_episodes(episodes).values()
    ]
    return Task(
        dataset=MemoryDataset(samples, name=Path(episodes).stem),
        solver=supervise(episodes),
        scorer=assisted_result(episodes),
        fail_on_error=False,  # failed samples are reported and excluded, not scored
    )
