import pytest

from board import Board
from solver import solve


def test_simple_solution_exists():
    # Board where each tube is already uniform → solved immediately.
    board = Board(
        [
            ["Red"],
            ["Red", "Red"],
        ],
        capacities=(3, 3),
    )
    solution = solve(board)
    assert solution is not None
    # Already solved: each non-empty tube contains only one colour.
    assert len(solution) == 0


def test_unsolvable_board_no_empty_tube():
    board = Board(
        [
            ["Red", "Blue"],
            ["Red", "Blue"],
        ],
        capacities=(5, 5),
    )
    assert solve(board) is None


def test_board_init_filters_empty_strings():
    board = Board([["Red", "", "Blue"]], capacities=(5,))
    assert len(board.tubes[0]) == 2
    assert "Red" in board.tubes[0]
    assert "Blue" in board.tubes[0]


def test_board_capacity_mismatch_raises():
    with pytest.raises(ValueError):
        Board([["Red"], ["Blue"]], capacities=(3,))


def test_board_capacities_per_tube():
    board = Board([["Red"]], capacities=(4,))
    assert board.capacities[0] == 4
