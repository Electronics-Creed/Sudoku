import numpy as np
import cv2
import tensorflow as tf
import random as rd


def draw_the_lines(img, lines):
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def sudoku_solver(board):
    found = empty_slot(board)
    if not found:
        return True
    else:
        sdk1, sdk2 = found
    for i in range(1, 10):
        if check(board, i, (sdk1, sdk2)):
            board[sdk1][sdk2] = i
            if sudoku_solver(board):
                return True
            board[sdk1][sdk2] = 0
    return False


def check(board, num, pos):
    for m in range(0, 9):
        if (num == board[pos[0]][m] and pos[1] != m) or (num == board[m][pos[1]] and pos[0] != m):
            return False
    a = pos[0] // 3
    b = pos[1] // 3
    for i in range(a * 3, (a + 1) * 3):
        for j in range(b * 3, (b + 1) * 3):
            if num == board[i][j] and pos != (i, j):
                return False
    return True


def empty_slot(board):
    for i in range(0, 9):
        for j in range(0, 9):
            if board[i][j] == 0:
                return i, j
    return False


def final(r):
    img = cv2.imread('D:\\Education\\NN weights\\sudoku_boards\\sudoku' + str(r) + '.jpg', 1)
    # D:\\Education\\NN weights\\sudoku_boards\\sudoku54.jpg
    cv2.imshow('sudoku', img)
    cv2.waitKey(2000)

    board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    print('Loading model!...')
    model = tf.keras.models.load_model('sudoku_model.h5')
    model.load_weights('sudoku_weights.h5')

    canny = cv2.Canny(img, 100, 255)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 60, 50, lines=np.array([]), minLineLength=60, maxLineGap=40)
    image_with_no_lines = draw_the_lines(img, lines)

    print('Scanning Image!...')

    for i in range(0, 252, 28):
        for j in range(0, 252, 28):
            x = image_with_no_lines[i:i + 28, j:j + 28]
            x = x.reshape(1, x.shape[0], x.shape[1], 3)
            c = model.predict(x)[0].tolist()
            board[i // 28][j // 28] = c.index(max(c))
    return board


def print_in_opencv(prev_board, board):
    img = cv2.imread('sudoku_grid.jpg', 1)
    for i in range(0, 9):
        for j in range(0, 9):
            if board[i][j] != prev_board[i][j]:
                cv2.putText(img, str(board[i][j]), ((80 * j) + 30, (80 * i) + 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 0, 0), 2)
            else:
                cv2.putText(img, str(board[i][j]), ((80 * j) + 30, (80 * i) + 50), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    cv2.imshow('final', img)
    cv2.waitKey(0)


r = rd.randint(2, 55)
board = final(r)
print('Solving wait.....')
sudoku_solver(board)
prev_board = final(r)
print_in_opencv(prev_board, board)

cv2.waitKey(0)
cv2.destroyAllWindows()
