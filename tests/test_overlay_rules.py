import numpy as np

from sudoku_solver.cv.overlay import overlay_solution, should_render_digit


def test_should_render_digit_only_for_originally_empty_cells():
    assert should_render_digit(original_value=0, solved_value=9) is True
    assert should_render_digit(original_value=5, solved_value=5) is False
    assert should_render_digit(original_value=4, solved_value=7) is False
    assert should_render_digit(original_value=0, solved_value=0) is False


def test_overlay_solution_renders_only_missing_digits():
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    corners = np.array([[0, 0], [179, 0], [179, 179], [0, 179]], dtype=np.float32)
    original_board = [[0] + [5] * 8] + [[5] * 9 for _ in range(8)]
    solved_board = [[9] + [5] * 8] + [[5] * 9 for _ in range(8)]

    overlay = overlay_solution(image, corners, original_board, solved_board)

    first_cell = overlay[0:20, 0:20]
    second_cell = overlay[0:20, 20:40]
    assert np.count_nonzero(first_cell) > 0
    assert np.count_nonzero(second_cell) == 0


def test_overlay_solution_preserves_grayscale_input_shape_with_color_output():
    image = np.zeros((90, 90), dtype=np.uint8)
    corners = np.array([[0, 0], [89, 0], [89, 89], [0, 89]], dtype=np.float32)
    original_board = [[0] * 9 for _ in range(9)]
    solved_board = [[1] * 9 for _ in range(9)]

    overlay = overlay_solution(image, corners, original_board, solved_board)

    assert overlay.shape == (90, 90, 3)
