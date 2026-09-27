"""Episode records: one opening, its unassisted game, and every one-move intervention.

Vocabulary used throughout the package:

- An *opening* is the full move list from the standard starting position, taken
  from a real game. Engines play everything after it.
- A *generated ply* is a half-move played by an engine after the opening.
  ``Game.moves[i]`` is generated ply ``i`` (zero-based).
- The *assisted color* is the side a supervisor may help. The Worker engine plays
  that side and the Opponent engine plays the other side, with identical settings.
- An *eligible turn* is a position in the unassisted game where the assisted side
  is to move and the absolute game move (``board.fullmove_number``) is at most
  ``LAST_INTERVENTION_MOVE``.
- An *intervention* replaces the Worker's move at one eligible turn with the
  Expert's move (a deeper search). The engines then play on as usual.
- A *score* is the final result from the assisted side's perspective:
  1 for a win, 0.5 for a draw and 0 for a loss.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

LAST_INTERVENTION_MOVE = 50
MAX_GENERATED_PLIES = 400
COLORS = ("white", "black")


@dataclass(frozen=True)
class Ending:
    result: str  # "1-0", "0-1" or "1/2-1/2"
    termination: str


def game_ending(board: chess.Board, generated_plies: int) -> Ending | None:
    """Return how the game ends at this position, or None if play continues.

    Draws by threefold repetition and the fifty-move rule end the game as soon
    as they occur, which is when a player could first claim them. Games still
    running after ``MAX_GENERATED_PLIES`` engine plies are scored as draws; the
    cap keeps generation bounded and is rarely reached.
    """
    outcome = board.outcome()
    if outcome is not None:
        return Ending(outcome.result(), outcome.termination.name.lower())
    if board.is_repetition(3):
        return Ending("1/2-1/2", "threefold_repetition")
    if board.halfmove_clock >= 100:
        return Ending("1/2-1/2", "fifty_moves")
    if generated_plies >= MAX_GENERATED_PLIES:
        return Ending("1/2-1/2", "move_cap")
    return None


def assisted_score(result: str, assisted_color: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    winner = "white" if result == "1-0" else "black"
    return 1.0 if winner == assisted_color else 0.0


@dataclass(frozen=True)
class Game:
    """Engine moves after the opening, in UCI notation, and how the game ended."""

    moves: tuple[str, ...]
    result: str
    termination: str


@dataclass(frozen=True)
class Intervention:
    """The Expert replaces the Worker's move at generated ply ``ply``.

    ``continuation`` holds the generated moves from that ply to the end of the
    game, starting with the Expert's move. It is None when the Expert chose the
    Worker's move: searches are deterministic, so that game is identical to the
    unassisted game. The token is spent either way.
    """

    ply: int
    fullmove: int
    worker_move: str
    expert_move: str
    continuation: tuple[str, ...] | None
    result: str
    termination: str

    @property
    def changed_move(self) -> bool:
        return self.expert_move != self.worker_move


@dataclass(frozen=True)
class Episode:
    id: str
    source: dict
    opening: tuple[str, ...]
    assisted_color: str
    baseline: Game
    interventions: tuple[Intervention, ...]

    @property
    def game_id(self) -> str:
        """Both colors of one opening share a source game; analysis groups on it."""
        return self.source["game_id"]

    @property
    def baseline_score(self) -> float:
        return assisted_score(self.baseline.result, self.assisted_color)

    def intervention_score(self, turn: int) -> float:
        """Score when the supervisor intervenes at eligible turn ``turn`` (zero-based)."""
        return assisted_score(self.interventions[turn].result, self.assisted_color)

    def score(self, turn: int | None) -> float:
        """Score for a supervisor that intervenes at ``turn``, or never (None)."""
        return self.baseline_score if turn is None else self.intervention_score(turn)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> Episode:
        baseline = data["baseline"]
        return cls(
            id=data["id"],
            source=data["source"],
            opening=tuple(data["opening"]),
            assisted_color=data["assisted_color"],
            baseline=Game(tuple(baseline["moves"]), baseline["result"], baseline["termination"]),
            interventions=tuple(
                Intervention(
                    ply=item["ply"],
                    fullmove=item["fullmove"],
                    worker_move=item["worker_move"],
                    expert_move=item["expert_move"],
                    continuation=(
                        None if item["continuation"] is None else tuple(item["continuation"])
                    ),
                    result=item["result"],
                    termination=item["termination"],
                )
                for item in data["interventions"]
            ),
        )


def write_episodes(path: Path, episodes: Iterable[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for episode in episodes:
            file.write(episode.to_json() + "\n")


def read_episodes(path: Path) -> list[Episode]:
    with path.open(encoding="utf-8") as file:
        return [Episode.from_dict(json.loads(line)) for line in file if line.strip()]


def replay(moves: Iterable[str], board: chess.Board | None = None) -> chess.Board:
    """Play UCI moves from ``board`` (default: the standard start), checking legality."""
    board = chess.Board() if board is None else board.copy()
    for index, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} at index {index} in {board.fen()}")
        board.push(move)
    return board


def is_eligible(board: chess.Board, assisted_color: str) -> bool:
    color = board.turn == chess.WHITE
    return color == (assisted_color == "white") and board.fullmove_number <= LAST_INTERVENTION_MOVE


def eligible_positions(episode: Episode) -> Iterator[tuple[int, chess.Board]]:
    """Yield (generated ply, board) for every eligible turn of the unassisted game."""
    board = replay(episode.opening)
    for ply, uci in enumerate(episode.baseline.moves):
        if is_eligible(board, episode.assisted_color):
            yield ply, board.copy()
        board.push_uci(uci)


def _check_game(start: chess.Board, moves: tuple[str, ...], first_ply: int, ending: Ending) -> None:
    """Moves must be legal, the game must not end early, and must end as recorded."""
    board = start.copy()
    for offset, uci in enumerate(moves):
        if game_ending(board, first_ply + offset) is not None:
            raise ValueError(f"game continues after it ended at generated ply {first_ply + offset}")
        board = replay([uci], board)
    actual = game_ending(board, first_ply + len(moves))
    if actual != ending:
        raise ValueError(f"recorded ending {ending} but replay gives {actual}")


def verify_episode(episode: Episode) -> None:
    """Replay every recorded game and check the episode's internal consistency.

    This checks chess legality and endings, not engine choices: re-running the
    engines is the job of the generation script.
    """
    if episode.assisted_color not in COLORS:
        raise ValueError(f"{episode.id}: unknown assisted color {episode.assisted_color}")
    start = replay(episode.opening)
    if game_ending(start, 0) is not None:
        raise ValueError(f"{episode.id}: opening ends the game")
    baseline = episode.baseline
    _check_game(start, baseline.moves, 0, Ending(baseline.result, baseline.termination))

    positions = list(eligible_positions(episode))
    if [ply for ply, _ in positions] != [item.ply for item in episode.interventions]:
        raise ValueError(f"{episode.id}: interventions do not match the eligible turns")
    for (ply, board), item in zip(positions, episode.interventions, strict=True):
        where = f"{episode.id} ply {ply}"
        if item.fullmove != board.fullmove_number or item.worker_move != baseline.moves[ply]:
            raise ValueError(f"{where}: move number or Worker move does not match the game")
        if not item.changed_move:
            if item.continuation is not None:
                raise ValueError(f"{where}: unchanged move must reuse the unassisted game")
            if (item.result, item.termination) != (baseline.result, baseline.termination):
                raise ValueError(f"{where}: unchanged move must keep the unassisted ending")
            continue
        if item.continuation is None or item.continuation[0] != item.expert_move:
            raise ValueError(f"{where}: continuation must start with the Expert move")
        _check_game(board, item.continuation, ply, Ending(item.result, item.termination))
