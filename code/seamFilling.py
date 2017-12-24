import cv2
import numpy as np
import time

# function calculate energy
def energy(I):
    def energy_grey(grey):
        # sobel filters
        dx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        dy = np.transpose(dx)
        return np.abs(cv2.filter2D(grey, -1, dx)) + \
            np.abs(cv2.filter2D(grey, -1, dy))

    num_channel = I.shape[2] if len(I.shape) == 3 else 1
    E = np.zeros(I.shape[:-1])
    if num_channel == 1:
        E = energy_grey(I)
    else:
        for x in [I[:, :, c] for c in range(num_channel)]:
            E += energy_grey(x)
    return E


def detect(edge, T=5):
    """ Return the black edge index 
    [Args]
        edge: An arrary of edge
        T: Threshold
    """
    x = np.where(edge == 0)[0]
    if len(x) == 0:
        return None

    idxs = np.where((x[1:] - x[:-1]) >= T)[0]
    if len(idxs) == 0:
        return [[0, x[-1]]]

    non_black = [[x[idx], x[idx + 1]] for idx in idxs]
    result = []
    result.append([x[0], non_black[0][0]])
    for i in range(1, len(non_black)):
        result.append([non_black[i - 1][1], non_black[i][0]])
    result.append([non_black[len(non_black) - 1][1], x[-1]])
    return [[beg, end] for beg, end in result if beg != end]


def optimalSeam(E):
    UL = 0  # upper left
    UP = 1  # up
    UR = 2  # upper right
    M = np.zeros(E.shape)
    M = np.pad(M, ((0, 0), (1, 1)), mode='constant',
               constant_values=np.finfo(np.float32).max)
    D = np.zeros(M.shape)
    # assign 1st row
    M[0, 1:-1] = E[0, :]
    for h in range(1, M.shape[0]):
        for w in range(1, M.shape[1] - 1):
            # pool = [UL, UP, UR]
            pool = [M[h - 1, w - 1], M[h - 1, w], M[h - 1, w + 1]]
            min_idx = np.argmin(pool)
            M[h, w] = E[h, w - 1] + pool[min_idx]
            D[h, w] = min_idx

    # min of last row
    curr = M.shape[0] - 1
    min_idx = np.argmin(M[curr, 1:-1]) + 1
    # minus 1 due to padding earlier
    seam = [[curr, min_idx - 1]]
    while curr != 0:
        direction = D[curr, min_idx]
        if direction == UL:
            offset = -1
        elif direction == UP:
            offset = 0
        elif direction == UR:
            offset = 1
        else:
            print('Error occurs when computing optimal seam.')
        curr -= 1
        min_idx += offset
        # minus 1 due to padding earlier
        seam.append([curr, min_idx - 1])
    return seam[::-1], M


def seamFilling(im, seam, is_vertical=True):
    h, w, d = im.shape
    if is_vertical:
        _im = np.zeros([h, w + 1, d])
        for i in range(h):
            _im[i, :seam[i][1], :] = im[i, :seam[i][1], :]
            _im[i, seam[i][1], :] = im[i, seam[i][1], :]
            _im[i, seam[i][1] + 1:, :] = im[i, seam[i][1]:, :]
    else:
        _im = np.zeros([h + 1, w, d])
        for j in range(w):
            _im[:seam[j][1], j, :] = im[:seam[j][1], j, :]
            _im[seam[j][1], j, :] = im[seam[j][1], j, :]
            _im[seam[j][1] + 1:, j, :] = im[seam[j][1]:, j, :]
    return _im


def fill_up(im, bound, k=200):
    ST = time.time()
    black_edge = detect(im[0, :, 0])
    cnt = 0
    while black_edge and cnt <= k:
        for beg, end in black_edge:
            _im = im[bound[0]:bound[1], beg:end, :]
            s, m = optimalSeam(np.transpose(energy(_im)))
            _im = seamFilling(_im, s, is_vertical=False)
            im[:bound[0] - 1, :, :] = im[1:bound[0], :, :]
            im[bound[0] - 1:bound[1], beg:end, :] = _im
        print(cnt)
        cnt += 1
        black_edge = detect(im[0, :, 0])
    print('%f(sec)' % (time.time() - ST))
    return im


def fill_down(im, bound, k=200):
    ST = time.time()
    black_edge = detect(im[-1, :, 0])
    cnt = 0
    while black_edge and cnt <= k:
        for beg, end in black_edge:
            _im = im[bound[0]:bound[1], beg:end, :]
            s, m = optimalSeam(np.transpose(energy(_im)))
            _im = seamFilling(_im, s, is_vertical=False)
            im[bound[1] + 1:, :, :] = im[bound[1]:-1, :, :]
            im[bound[0]:bound[1] + 1, beg:end, :] = _im
        print(cnt)
        cnt += 1
        black_edge = detect(im[-1, :, 0])
    print('%f(sec)' % (time.time() - ST))
    return im


def fill_left(im, bound, k=200):
    ST = time.time()
    black_edge = detect(im[:, 0, 0])
    cnt = 0
    while black_edge and cnt <= k:
        for beg, end in black_edge:
            _im = im[beg:end, bound[0]:bound[1], :]
            s, m = optimalSeam(energy(_im))
            _im = seamFilling(_im, s, is_vertical=True)
            im[:, :bound[0] - 1, :] = im[:, 1:bound[0], :]
            im[beg:end, bound[0] - 1:bound[1], :] = _im
        print(cnt)
        cnt += 1
        black_edge = detect(im[:, 0, 0])
    print('%f(sec)' % (time.time() - ST))
    return im


def fill_right(im, bound, k):
    ST = time.time()
    black_edge = detect(im[:, -1, 0])
    cnt = 0
    while black_edge and cnt <= k:
        for beg, end in black_edge:
            _im = im[beg:end, bound[0]:bound[1], :]
            s, m = optimalSeam(energy(_im))
            _im = seamFilling(_im, s, is_vertical=True)
            im[:, bound[1] + 1:, :] = im[:, bound[1]:-1, :]
            im[beg:end, bound[0]:bound[1] + 1, :] = _im
        print(cnt)
        cnt += 1
        black_edge = detect(im[:, -1, 0])
    print('%f(sec)' % (time.time() - ST))
    return im
