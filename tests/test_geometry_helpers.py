import numpy as np

from sudoku_solver.cv.board_detect import detect_board
from sudoku_solver.cv.cells import extract_cells
from sudoku_solver.cv.warp import order_corners, warp_board


def test_order_corners_returns_top_left_first():
    corners = np.array([[200, 50], [50, 50], [50, 200], [200, 200]], dtype=np.float32)

    ordered = order_corners(corners)

    assert ordered[0].tolist() == [50.0, 50.0]
    assert ordered[1].tolist() == [200.0, 50.0]
    assert ordered[2].tolist() == [200.0, 200.0]
    assert ordered[3].tolist() == [50.0, 200.0]


def test_warp_board_returns_square_image_for_axis_aligned_corners():
    image = np.zeros((120, 120), dtype=np.uint8)
    image[20:100, 20:100] = 255
    corners = np.array([[20, 20], [99, 20], [99, 99], [20, 99]], dtype=np.float32)

    warped = warp_board(image, corners)

    assert warped.shape == (79, 79)
    assert warped[10:60, 10:60].mean() > 200


def test_detect_board_finds_largest_quadrilateral_contour():
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    image[30:210, 30:210] = 255

    detection = detect_board(image)

    assert detection is not None
    assert detection.area > 30000
    assert detection.corners.shape == (4, 2)
    assert np.allclose(detection.corners[0], [30.0, 30.0], atol=6.0)
    assert np.allclose(detection.corners[2], [209.0, 209.0], atol=6.0)


def test_extract_cells_returns_81_row_major_crops():
    board = np.arange(90 * 90, dtype=np.int32).reshape(90, 90)

    cells = extract_cells(board)

    assert len(cells) == 81
    assert cells[0].shape == (10, 10)
    assert cells[0][0, 0] == board[0, 0]
    assert cells[1][0, 0] == board[0, 10]
    assert cells[9][0, 0] == board[10, 0]
