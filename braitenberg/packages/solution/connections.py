from typing import Tuple
import math
import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")

    # Bumpers
    bumpW = 250
    bumpH = 150
    res[0:bumpH, 0:bumpW] = 0.4
    res[0:bumpH, shape[1]-bumpW:shape[1]] = -0.4

    # The peak of our mountain shape
    peak = 250
    mid = shape[1] // 2
    # The y coord at which point & below which we accept everything
    full = math.floor(0.8*shape[0])

    # The rectangle at the bottom
    res[full:, :mid] = -1
    res[full:, (mid + 1):] = 1

    # The sloped portion
    for i in range(peak, full):
        partial = (i-peak)/(full-peak)
        # res[i, mid-math.floor((i**2)/(1.5*full)):mid] = partial**2
        # res[i, (mid + 1):(mid + math.floor((i**1.95)/(full)))] = -(partial**2)
        res[i, mid-math.floor((mid * partial)):mid] = -1
        res[i, (mid + 1):(mid + math.floor(mid * partial))] = 1
    
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.fliplr(get_motor_left_matrix(shape))
    return res