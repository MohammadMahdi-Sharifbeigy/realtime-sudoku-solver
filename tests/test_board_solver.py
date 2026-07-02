from sudoku_solver.solver.backtracking import solve_board


def test_solve_board_solves_known_puzzle():
    puzzle = [
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

    solved = solve_board(puzzle)

    assert solved is not None
    assert solved[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]
    assert solved[8] == [3, 4, 5, 2, 8, 6, 1, 7, 9]
    assert puzzle[0] == [5, 3, 0, 0, 7, 0, 0, 0, 0]


def test_solve_board_returns_none_for_invalid_puzzle():
    puzzle = [[1] * 9 for _ in range(9)]

    assert solve_board(puzzle) is None


def test_solve_board_returns_none_for_unsolvable_but_well_formed_puzzle():
    puzzle = [
        [5, 1, 6, 8, 4, 9, 7, 3, 2],
        [3, 0, 7, 6, 0, 5, 0, 0, 0],
        [8, 0, 9, 7, 0, 0, 0, 6, 5],
        [1, 3, 5, 0, 6, 0, 9, 0, 7],
        [4, 7, 2, 5, 9, 1, 0, 0, 6],
        [9, 6, 8, 3, 7, 0, 5, 0, 0],
        [2, 5, 3, 1, 8, 6, 0, 7, 4],
        [6, 8, 4, 2, 0, 7, 0, 5, 0],
        [7, 9, 1, 0, 5, 0, 6, 0, 8],
    ]

    assert solve_board(puzzle) is None
