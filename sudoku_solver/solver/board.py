"""Sudoku board validation helpers for 9x9 puzzles."""

from __future__ import annotations


Board = list[list[int]]

BOARD_SIZE = 9
BOX_SIZE = 3
VALID_VALUES = set(range(10))


def is_valid_board(board: Board) -> bool:
    """Return True when the board is a valid 9x9 Sudoku state."""
    if not _has_valid_shape(board):
        return False

    for row in board:
        if any(not isinstance(value, int) or value not in VALID_VALUES for value in row):
            return False

    return (
        _all_units_valid(board)
        and _all_units_valid(_iter_columns(board))
        and _all_units_valid(_iter_boxes(board))
    )


def _has_valid_shape(board: Board) -> bool:
    return (
        isinstance(board, list)
        and len(board) == BOARD_SIZE
        and all(isinstance(row, list) and len(row) == BOARD_SIZE for row in board)
    )


def _all_units_valid(units: list[list[int]]) -> bool:
    return all(_unit_has_no_duplicates(unit) for unit in units)


def _unit_has_no_duplicates(unit: list[int]) -> bool:
    seen: set[int] = set()
    for value in unit:
        if value == 0:
            continue
        if value in seen:
            return False
        seen.add(value)
    return True


def _iter_columns(board: Board) -> list[list[int]]:
    return [[board[row_index][column_index] for row_index in range(BOARD_SIZE)] for column_index in range(BOARD_SIZE)]


def _iter_boxes(board: Board) -> list[list[int]]:
    boxes: list[list[int]] = []
    for start_row in range(0, BOARD_SIZE, BOX_SIZE):
        for start_column in range(0, BOARD_SIZE, BOX_SIZE):
            box = [
                board[row_index][column_index]
                for row_index in range(start_row, start_row + BOX_SIZE)
                for column_index in range(start_column, start_column + BOX_SIZE)
            ]
            boxes.append(box)
    return boxes
