"""Sample balanced openings from real games.

Each opening is the first 15 to 20 full moves of a game from the Lichess Elite
Database (games between players rated 2400+ and 2200+). An opening is kept if
Stockfish at depth 18 rates the position within 1.5 pawns, so neither side
starts out lost. The output is small and committed, so later steps do not need
the 300 MB source file.

    uv run python scripts/1_sample_openings.py --pgn lichess_elite_2020-05.pgn

Source: https://database.nikonoel.fr/ (lichess_elite_2020-05.zip). Lichess game
data is released under CC0.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import chess.pgn

from chess_intervention.engine import REPO_ROOT, Stockfish
from chess_intervention.episodes import game_ending, replay

OUTPUT = REPO_ROOT / "data" / "openings.jsonl"
BALANCE_DEPTH = 18
MAX_BALANCE_CP = 150
FIRST_CUT_MOVE, LAST_CUT_MOVE = 15, 20


def game_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open(encoding="utf-8") as file:
        while True:
            offset = file.tell()
            headers = chess.pgn.read_headers(file)
            if headers is None:
                return offsets
            if headers.get("Variant", "Standard") == "Standard" and "FEN" not in headers:
                offsets.append(offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()

    sha256 = hashlib.sha256(args.pgn.read_bytes()).hexdigest()
    offsets = game_offsets(args.pgn)
    print(f"{len(offsets)} standard games in {args.pgn.name} (SHA-256 {sha256})")
    rng = random.Random(args.seed)
    order = list(range(len(offsets)))
    rng.shuffle(order)

    openings, tried = [], 0
    with Stockfish(args.engine) as engine, args.pgn.open(encoding="utf-8") as file:
        for index in order:
            if len(openings) == args.count:
                break
            tried += 1
            cut_move = rng.randint(FIRST_CUT_MOVE, LAST_CUT_MOVE)
            file.seek(offsets[index])
            game = chess.pgn.read_game(file)
            moves = [move.uci() for move in game.mainline_moves()]
            if len(moves) <= 2 * cut_move:
                continue  # the real game ended before the cut
            opening = moves[: 2 * cut_move]
            board = replay(opening)
            if game_ending(board, 0) is not None:
                continue
            eval_cp = engine.evaluate(board, BALANCE_DEPTH)
            if eval_cp is None or abs(eval_cp) > MAX_BALANCE_CP:
                continue
            headers = game.headers
            openings.append(
                {
                    "source": {
                        "game_id": f"elite-2020-05-{index:06d}",
                        "event": headers.get("Event"),
                        "date": headers.get("Date"),
                        "white_elo": headers.get("WhiteElo"),
                        "black_elo": headers.get("BlackElo"),
                        "eco": headers.get("ECO"),
                        "opening_name": headers.get("Opening"),
                        "eval_cp": eval_cp,
                    },
                    "opening": opening,
                }
            )
            print(f"{len(openings)}/{args.count} kept after {tried} games", end="\r")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as out:
        for item in openings:
            out.write(json.dumps(item) + "\n")
    meta = {
        "source_file": args.pgn.name,
        "source_sha256": sha256,
        "standard_games": len(offsets),
        "games_tried": tried,
        "seed": args.seed,
        "cut_moves": [FIRST_CUT_MOVE, LAST_CUT_MOVE],
        "balance_depth": BALANCE_DEPTH,
        "max_balance_cp": MAX_BALANCE_CP,
        "engine": engine.name,
    }
    OUTPUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nWrote {len(openings)} openings to {OUTPUT} ({tried} games tried)")


if __name__ == "__main__":
    main()
