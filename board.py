from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Board:
    tubes: Tuple[Tuple[str, ...], ...]
    capacities: Tuple[int, ...]

    """
    A frozen representation of the Woody Sort game board.

    Each tube is stored bottom-to-top (index 0 = bottom of the tube,
    last index = top of the tube). This makes push/pop O(1) since
    moving a ball from the top is a pop/append operation.

    Each tube has its own capacity (e.g. some may hold 3 balls while
    others hold 5).  The ``capacities`` tuple must have the same length
    as ``tubes``.
    """

    def __post_init__(self) -> None:
        normalized = []
        for tube in self.tubes:
            cleaned = tuple(item for item in tube if item not in {"", None})
            normalized.append(cleaned)
        object.__setattr__(self, "tubes", tuple(normalized))
        if len(self.tubes) != len(self.capacities):
            raise ValueError(
                f"Tube count ({len(self.tubes)}) does not match "
                f"capacities count ({len(self.capacities)})"
            )

    def __iter__(self):
        return iter(self.tubes)

    def __len__(self):
        return len(self.tubes)

    def copy(self) -> "Board":
        return Board(
            [list(tube) for tube in self.tubes],
            capacities=self.capacities,
        )

    def can_move(self, src: int, dst: int) -> bool:
        if src == dst:
            return False

        src_tube = self.tubes[src]
        dst_tube = self.tubes[dst]
        if not src_tube:
            return False
        if len(dst_tube) >= self.capacities[dst]:
            return False
        if not dst_tube:
            # Game rule: moving a complete uniform tube to an empty is
            # pointless, but moving a single ball is allowed.
            if len(set(src_tube)) == 1 and len(src_tube) == self.capacities[src]:
                return False
            return True
        return dst_tube[-1] == src_tube[-1]

    def move(self, src: int, dst: int) -> Optional["Board"]:
        if not self.can_move(src, dst):
            return None

        src_tube = list(self.tubes[src])
        dst_tube = list(self.tubes[dst])

        # In Woody Sort, a single tap moves all contiguous balls of the same color
        # from the top of the source tube that can fit into the destination tube.
        color = src_tube[-1]

        # Count contiguous balls of the same color at the top of src
        count_src = 0
        for i in range(len(src_tube) - 1, -1, -1):
            if src_tube[i] == color:
                count_src += 1
            else:
                break

        # Count available space in dst
        available_space = self.capacities[dst] - len(dst_tube)

        # Number of balls to move
        balls_to_move = min(count_src, available_space)

        for _ in range(balls_to_move):
            dst_tube.append(src_tube.pop())

        new_tubes = [tuple(tube) for tube in self.tubes]
        new_tubes[src] = tuple(src_tube)
        new_tubes[dst] = tuple(dst_tube)
        return Board(new_tubes, capacities=self.capacities)

    def is_solved(self) -> bool:
        # Every tube must be homogeneous (or empty)
        if not all(len(tube) == 0 or len(set(tube)) == 1 for tube in self.tubes):
            return False
            
        # No two tubes can share the same color
        colors_seen = set()
        for tube in self.tubes:
            if tube:
                color = tube[0]
                if color in colors_seen:
                    return False
                colors_seen.add(color)
        return True

    def valid_moves(self) -> List[Tuple[int, int]]:
        empty_indices = [i for i, t in enumerate(self.tubes) if not t]
        moves = []
        for src in range(len(self.tubes)):
            if not self.tubes[src]:
                continue
            for dst in range(len(self.tubes)):
                if not self.tubes[dst]:
                    if dst not in empty_indices:
                        continue
                if self.can_move(src, dst):
                    moves.append((src, dst))
        return moves

    def __repr__(self) -> str:
        return str(list(zip(self.capacities, self.tubes)))
