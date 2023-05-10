import cv2
import numpy as np



def generate_chessboard(h, w):
    board = np.zeros((h, w), dtype=np.uint8)
    board[::2, ::2] = 255
    board[1::2, 1::2] = 255
    return board


if __name__ == '__main__':
    h = 10
    w = 10
    pattern = generate_chessboard(h, w)
    cv2.imwrite(f'./ENV_PATTERNS/chessboard_{h}x{w}.png', pattern)
