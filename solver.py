import numpy as np


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print('- - - - - - - - - - - -')

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(str(bo[i][j]))
            else:
                print(str(bo[i][j]) + " ", end="")


# check if proposed number is valid
def ok_num(num, board, pos):
    if num in board[pos[0]]: return False
    for col in board:
        if num == col[pos[1]]: return False
    return ok_square(num, board, pos)


# check for the 3x3 square
def ok_square(num, board, pos):
    pos_modulo = (pos[0]%3, pos[1]%3)
    square = np.zeros((3, 3), dtype=int)
    for i in range(3):  # 0, 1, 2
        for j in range(3):
            if (i, j) == pos_modulo:
                pass
            square[i, j] = board[pos[0] + i - pos_modulo[0]][pos[1] + j - pos_modulo[1]]
    if num in square: return False
    else: return True


def solve_board(board):
    new_board = False
    temp_board = board.copy()
    if 0 not in np.concatenate(board): return board
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for x in range(1, 10):
                    if ok_num(x, board, (i, j)):
                        temp_board[i][j] = x
                        new_board = solve_board(temp_board)
                    if np.any(new_board): return new_board
                return False


# run this function with a 9x9 array representing a sudoku board
def run_solver(board):
    board = np.array(board)
    result = solve_board(board)
    if np.any(result): print_board(result)
    else: print('This board cannot be solved')


# run_solver(np.zeros((9,9), dtype=int))