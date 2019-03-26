import numpy as np


def euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v2 - v1)


def manhattan(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the Manhattan distance - the sum of
    the distances along every dimension
    """
    return sum(map(abs, v2 - v1))
