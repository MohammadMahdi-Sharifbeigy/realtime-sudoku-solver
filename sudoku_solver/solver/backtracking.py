"""Backtracking Sudoku solver."""

from __future__ import annotations

from .board import BOARD_SIZE, Board, BOX_SIZE, is_valid_board


def solve_board(board: Board) -> Board | None:
    """Return a solved deep copy of the board or None if unsolvable."""
    if not is_valid_board(board):
        return None

    candidate = [row[:] for row in board]
    if _solve_in_place(candidate):
        return candidate
    return None


def _solve_in_place(board: Board) -> bool:
    empty_position = _find_empty_cell(board)
    if empty_position is None:
        return True

    row_index, column_index = empty_position
    for value in range(1, BOARD_SIZE + 1):
        if _can_place(board, row_index, column_index, value):
            board[row_index][column_index] = value
            if _solve_in_place(board):
                return True
            board[row_index][column_index] = 0

    return False


def _find_empty_cell(board: Board) -> tuple[int, int] | None:
    for row_index in range(BOARD_SIZE):
        for column_index in range(BOARD_SIZE):
            if board[row_index][column_index] == 0:
                return row_index, column_index
    return None


def _can_place(board: Board, row_index: int, column_index: int, value: int) -> bool:
    if value in board[row_index]:
        return False

    if any(board[current_row][column_index] == value for current_row in range(BOARD_SIZE)):
        return False

    start_row = (row_index // BOX_SIZE) * BOX_SIZE
    start_column = (column_index // BOX_SIZE) * BOX_SIZE
    for current_row in range(start_row, start_row + BOX_SIZE):
        for current_column in range(start_column, start_column + BOX_SIZE):
            if board[current_row][current_column] == value:
                return False

    return True
