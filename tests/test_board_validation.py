from sudoku_solver.solver.board import is_valid_board


def test_is_valid_board_accepts_partial_valid_board():
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    assert is_valid_board(board) is True


def test_is_valid_board_rejects_non_9x9_shape():
    board = [[0] * 9 for _ in range(8)]

    assert is_valid_board(board) is False


def test_is_valid_board_rejects_out_of_range_values():
    board = [[0] * 9 for _ in range(9)]
    board[4][4] = 10

    assert is_valid_board(board) is False


def test_is_valid_board_rejects_duplicate_in_row():
    board = [[0] * 9 for _ in range(9)]
    board[0][0] = 5
    board[0][3] = 5

    assert is_valid_board(board) is False


def test_is_valid_board_rejects_duplicate_in_column():
    board = [[0] * 9 for _ in range(9)]
    board[0][1] = 7
    board[8][1] = 7

    assert is_valid_board(board) is False


def test_is_valid_board_rejects_duplicate_in_box():
    board = [[0] * 9 for _ in range(9)]
    board[0][0] = 9
    board[1][1] = 9

    assert is_valid_board(board) is False
