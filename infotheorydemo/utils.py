import numpy as np
from typing import List, Tuple


def to_pyitlib_format(co_mat: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    convert data in co-occurrence matrix to two lists, each with realisations of one discrete RV.
    """
    targets = []
    contexts = []
    for i in range(co_mat.shape[0]):
        for j in range(co_mat.shape[1]):
            num = co_mat[i, j]
            if num > 0:
                targets += [i] * num
                contexts += [j] * num

    return targets, contexts
