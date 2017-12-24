# ---------------
# Author: Tu, Tao
# ---------------

import numpy as np

def im2float(im):
    if im.dtype == np.uint8:
        return im / 255.0

def float2im(im):
    if im.dtype == np.float32 or im.dtype == np.float64:
        return (im * 255.0).astype(np.uint8)


def normalize(im):
    _im = im - np.min(im)
    return _im / np.max(_im)









