"""Pinned Stockfish settings and a searcher whose moves depend only on the game.

Stockfish's hash table and search histories carry information from one search to
the next, so the same position can get a different move depending on what was
searched before it. Clearing them before every search (``ucinewgame``) makes each
move a function of the game's move history and the search depth alone. That lets
the generator play every intervention independently and in parallel, and lets
anyone reproduce a game from its moves.

The hash size is pinned too: with a fixed depth, a different hash size can
change the chosen move.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import chess
import chess.engine

WORKER_DEPTH = 12  # the Worker and the Opponent, which have identical settings
EXPERT_DEPTH = 18
OPTIONS = {"Threads": 1, "Hash": 16}

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engines"
STOCKFISH_VERSION = "17.1"
_RELEASE = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1"
DOWNLOADS = {
    "win32": {
        "url": f"{_RELEASE}/stockfish-windows-x86-64-avx2.zip",
        "sha256": "92a77f8d8116b4331696eeb7b232bd03db30d6641a6ce1be1759478b8931d28b",
        "executable": "stockfish/stockfish-windows-x86-64-avx2.exe",
    },
    "linux": {
        "url": f"{_RELEASE}/stockfish-ubuntu-x86-64-avx2.tar",
        "sha256": "09e953222dbe80aaea5c33dab265413b295cb8376418445c877708c61e31987d",
        "executable": "stockfish/stockfish-ubuntu-x86-64-avx2",
    },
}


def default_engine_path() -> Path:
    if sys.platform not in DOWNLOADS:
        raise RuntimeError(f"No pinned Stockfish build for {sys.platform}; pass --engine")
    return ENGINE_DIR / f"stockfish-{STOCKFISH_VERSION}" / DOWNLOADS[sys.platform]["executable"]


class Searcher(Protocol):
    def best_move(self, board: chess.Board, depth: int) -> str: ...


class Stockfish:
    """A Stockfish process that forgets everything between searches."""

    def __init__(self, path: Path | None = None) -> None:
        path = default_engine_path() if path is None else path
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run scripts/0_get_stockfish.py")
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self._engine.configure(OPTIONS)
        self.name = self._engine.id.get("name", "unknown")

    def best_move(self, board: chess.Board, depth: int) -> str:
        # A new ``game`` object makes python-chess send ucinewgame first. The full
        # move stack is sent with the position, so repetitions are visible.
        result = self._engine.play(board, chess.engine.Limit(depth=depth), game=object())
        if result.move is None:
            raise RuntimeError(f"Stockfish returned no move for {board.fen()}")
        return result.move.uci()

    def evaluate(self, board: chess.Board, depth: int) -> int | None:
        """Centipawns from White's point of view, or None when a mate is found."""
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
        return info["score"].white().score()

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> Stockfish:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
