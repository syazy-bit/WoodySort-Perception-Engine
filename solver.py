from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Tuple

from board import Board

_MAX_VISITED = 500_000


def _canonical_state(board: Board) -> Tuple[Tuple[int, Tuple[str, ...]], ...]:
    """Produce a position-independent state hash by pairing each tube's
    capacity with its contents, then sorting.
    """
    return tuple(sorted(zip(board.capacities, board.tubes)))


def solve(
    board: Board,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[List[Tuple[int, int]]]:
    start = board.copy()
    visited = {_canonical_state(start)}
    queue: deque = deque()
    queue.append((start, []))

    while queue:
        if len(visited) > _MAX_VISITED:
            return None

        current, path = queue.popleft()

        if progress_callback is not None:
            progress_callback(len(visited), len(path))

        if current.is_solved():
            return path

        for src, dst in current.valid_moves():
            next_board = current.move(src, dst)
            if next_board is None:
                continue

            state = _canonical_state(next_board)
            if state in visited:
                continue

            visited.add(state)
            queue.append((next_board, path + [(src, dst)]))

    return None
