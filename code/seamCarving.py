import cv2
import numpy as np
import time
import copy


def fill(im, iter=50):
    def _max(L):
        x = 0
        for y in L:
            x = max(x, y)
        return x

    UP, DN, LT, RT = 0, 1, 2, 3
    H, W, D = im.shape
    ST = time.time()
    cnt = 0
    while(True):
        edge = [detect(im[0, :, :]),
                detect(im[-1, :, :]),
                detect(im[:, 0, :]),
                detect(im[:, -1, :])]
        if len(edge[UP]) + len(edge[DN]) + len(edge[LT]) + len(edge[RT]) == 0 or cnt >= iter:
            break
        candidates = [_max([end - beg for beg, end in edge[i]])
                      for i in [UP, DN, LT, RT]]
        win = np.argmax(candidates)
        arg = np.argmax([end - beg for beg, end in edge[win]])
        beg, end = edge[win][arg]
        if win == UP:
            im = fill_up(im, 0, W)
        elif win == DN:
            im = fill_down(im, 0, W)
        elif win == LT:
            im = fill_left(im, 0, H)
        elif win == RT:
            im = fill_right(im, 0, H)
        cnt += 1
        if cnt % 10 == 0:
            print('filling step:', cnt)
    print('Filling takes %f(sec)' % (time.time() - ST))
    return im


def fill_up(im, beg, end):
    im = copy.deepcopy(im)
    _im = im[:, beg:end, :]
    s, m = optimalSeam(np.transpose(energy_with_penalty(_im)))
    _im = seamFilling(_im, s, is_vertical=False)
    im[:, beg:end, :] = _im[1:, :, :]
    return im


def fill_down(im, beg, end):
    im = copy.deepcopy(im)
    _im = im[:, beg:end, :]
    s, m = optimalSeam(np.transpose(energy_with_penalty(_im)))
    _im = seamFilling(_im, s, is_vertical=False)
    im[:, beg:end, :] = _im[:-1, :, :]
    return im


def fill_left(im, beg, end):
    im = copy.deepcopy(im)
    _im = im[beg:end, :, :]
    s, m = optimalSeam(energy_with_penalty(_im))
    _im = seamFilling(_im, s, is_vertical=True)
    im[beg:end, :, :] = _im[:, 1:, :]
    return im


def fill_right(im, beg, end):
    im = copy.deepcopy(im)
    _im = im[beg:end, :, :]
    s, m = optimalSeam(energy_with_penalty(_im))
    _im = seamFilling(_im, s, is_vertical=True)
    im[beg:end, :, :] = _im[:, :-1, :]
    return im


def detect(edge, T=5, min_len=5):
    """ Return the black edge index 
    [Args]
        edge: An arrary of edge
        T: Threshold
        min_len: black edges' length >= min_length will be detected
    """
    result = list()
    x0 = np.where(edge[:, 0] <= T, True, False)
    x1 = np.where(edge[:, 1] <= T, True, False)
    x2 = np.where(edge[:, 2] <= T, True, False)
    x = np.logical_and(x0, np.logical_and(x1, x2))
    cnt = 0
    for i in range(len(x)):
        if x[i]:
            cnt += 1
        elif cnt >= min_len:
            result.append([i - cnt, i])
            cnt = 0
    if cnt >= min_len:
        result.append([len(x) - cnt, len(x)])
    return result


def preprocess(im):
    """ Crop the all black edges
    """
    while True:
        edge_up = detect(im[0, :, :])
        if len(edge_up) != 1:
            break
        beg, end = edge_up[0]
        if (end - beg) != len(im[0, :, 0]):
            break
        im = im[1:, :, :]

    while True:
        edge_down = detect(im[-1, :, :])
        if len(edge_down) != 1:
            break
        beg, end = edge_down[0]
        if (end - beg) != len(im[-1, :, 0]):
            break
        im = im[:-1, :, :]

    while True:
        edge_left = detect(im[:, 0, :])
        if len(edge_left) != 1:
            break
        beg, end = edge_left[0]
        if (end - beg) != len(im[:, 0, 0]):
            break
        im = im[:, 1:, :]

    while True:
        edge_right = detect(im[:, -1, :])
        if len(edge_right) != 1:
            break
        beg, end = edge_right[0]
        if (end - beg) != len(im[:, -1, 0]):
            break
        im = im[:, :-1, :]
    return im


def energy(I):
    """ Calculate energy
    """
    def energy_grey(grey):
        dx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        dy = np.transpose(dx)
        return np.abs(cv2.filter2D(grey, -1, dx)) + \
            np.abs(cv2.filter2D(grey, -1, dy))

    I = blur = cv2.GaussianBlur(I, (5, 5), 0)
    num_channel = I.shape[2] if len(I.shape) == 3 else 1
    E = np.zeros(I.shape[:-1])
    if num_channel == 1:
        E = energy_grey(I)
    else:
        for x in [I[:, :, c] for c in range(num_channel)]:
            E += energy_grey(x)
    return E


def energy_with_penalty(I, penalty=1e4):
    """ Calculate energy with penalty to black pixels
    """
    E = energy(I)
    height, width = E.shape
    for h in range(height):
        for beg, end in detect(I[h, :, :]):
            E[h, beg:end] = np.full(end - beg, penalty)
    for w in range(width):
        for beg, end in detect(I[:, w, :]):
            E[beg:end, w] = np.full(end - beg, penalty)
    return E


def optimalSeam(E):
    """ Get optimal seam """
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
