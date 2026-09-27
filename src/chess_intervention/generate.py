"""Play the unassisted game and every possible intervention for one episode.

Evaluating a supervisor later needs no engine: each of its possible choices
(never intervene, or intervene at eligible turn k) already has a finished game.
"""

from __future__ import annotations

import chess

from .engine import EXPERT_DEPTH, WORKER_DEPTH, Searcher
from .episodes import Ending, Episode, Game, Intervention, game_ending, is_eligible, replay


def play_out(
    board: chess.Board, searcher: Searcher, first_ply: int, depth: int = WORKER_DEPTH
) -> tuple[tuple[str, ...], Ending]:
    """Let both sides play at ``depth`` from ``board`` until the game ends.

    ``first_ply`` is the number of generated plies already played, which the
    move cap counts. The board is not modified.
    """
    board = board.copy()
    moves: list[str] = []
    while (ending := game_ending(board, first_ply + len(moves))) is None:
        move = searcher.best_move(board, depth)
        board.push_uci(move)
        moves.append(move)
    return tuple(moves), ending


def build_episode(
    episode_id: str,
    source: dict,
    opening: tuple[str, ...],
    assisted_color: str,
    searcher: Searcher,
    worker_depth: int = WORKER_DEPTH,
    expert_depth: int = EXPERT_DEPTH,
) -> Episode:
    start = replay(opening)
    moves, ending = play_out(start, searcher, 0, worker_depth)
    baseline = Game(moves, ending.result, ending.termination)

    interventions = []
    board = start
    for ply, worker_move in enumerate(moves):
        if is_eligible(board, assisted_color):
            expert_move = searcher.best_move(board, expert_depth)
            if expert_move == worker_move:
                continuation, branch_ending = None, ending
            else:
                after = replay([expert_move], board)
                rest, branch_ending = play_out(after, searcher, ply + 1, worker_depth)
                continuation = (expert_move, *rest)
            interventions.append(
                Intervention(
                    ply=ply,
                    fullmove=board.fullmove_number,
                    worker_move=worker_move,
                    expert_move=expert_move,
                    continuation=continuation,
                    result=branch_ending.result,
                    termination=branch_ending.termination,
                )
            )
        board = replay([worker_move], board)

    return Episode(
        id=episode_id,
        source=source,
        opening=tuple(opening),
        assisted_color=assisted_color,
        baseline=baseline,
        interventions=tuple(interventions),
    )
