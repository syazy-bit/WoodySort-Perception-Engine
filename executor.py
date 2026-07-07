"""Move executor for Woody Sort.

Converts solver moves (source_tube, dest_tube) into ADB tap sequences
using the detected tube centre coordinates.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from adb import tap
from detect import Tube

# Delay between the source tap and destination tap within a single move.
INTRA_MOVE_DELAY_S = 0.2

# Delay after a complete move before the next action (e.g. verification).
# If this is too short, the verification retry loop will catch it and wait automatically.
POST_MOVE_DELAY_S = 1.5


def execute_move(
    src_tube: Tube,
    dst_tube: Tube,
) -> None:
    """Execute a single move by tapping source then destination tube.

    Parameters
    ----------
    src_tube : Tube
        The tube to pick a ball from (tap its centre).
    dst_tube : Tube
        The tube to drop the ball into (tap its centre).
    """
    sx, sy = src_tube.center
    dx, dy = dst_tube.center

    tap(sx, sy)
    time.sleep(INTRA_MOVE_DELAY_S)
    tap(dx, dy)


def execute_moves(
    moves: List[Tuple[int, int]],
    tubes: List[Tube],
    *,
    delay_s: float = POST_MOVE_DELAY_S,
) -> None:
    """Execute a full sequence of moves.

    Parameters
    ----------
    moves : list of (src_index, dst_index)
        0-indexed tube indices from the solver.
    tubes : list of Tube
        Detected tubes with centre coordinates.
    delay_s : float
        Seconds to wait between moves.

    Raises
    ------
    IndexError
        If a move references a tube index outside the detected range.
    """
    for src_idx, dst_idx in moves:
        if src_idx < 0 or src_idx >= len(tubes):
            raise IndexError(f"Source tube index {src_idx} out of range (0-{len(tubes) - 1})")
        if dst_idx < 0 or dst_idx >= len(tubes):
            raise IndexError(f"Destination tube index {dst_idx} out of range (0-{len(tubes) - 1})")
        execute_move(tubes[src_idx], tubes[dst_idx])
        time.sleep(delay_s)
