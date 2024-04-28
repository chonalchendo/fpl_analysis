import numpy as np


def average_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    return np.mean(stack, axis=0)
