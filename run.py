import numpy as np
import scanner
import solver
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf


def generate_board_tiles(path):
    img = scanner.extract_sudoku(path)
    s = 30
    tiles = [img[x + 1:x + s - 1, y + 1:y + s - 1] for x in range(0, img.shape[0], s) for y in
             range(0, img.shape[1], s)]
    tiles = np.array(tiles).astype(np.float32).reshape((81, 28, 28, 1)) / 255
    return tiles


def solve_sudoku(path_img, path_model, print_unsolved=False):
    """
    print_unsolved will show you the board after extraction
    if the solver tell you that your board cannot be solved, the digit recognition probably went wrong
    might add some kind of board editing tool later on
    """
    model = tf.keras.models.load_model(path_model)
    try:
        tiles = generate_board_tiles(path_img)
    except:
        print("Couldn't read your image file. "
              "Please verify the Path or if the whole board is correctly included in the picture.")
        return
    board_raw = model(tiles)
    board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
    if print_unsolved:
        print(board_clean)
    solver.run_solver(board_clean)


# image_path = "test_pic/sudoku_shit_angle.jpg"
image_path = input("Please submit the directory of your sudoku picture: ")
model_path = "digit_reco_model"
solve_sudoku(image_path, model_path)