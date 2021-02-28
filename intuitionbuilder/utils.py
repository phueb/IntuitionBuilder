import numpy as np
from typing import List, Tuple, Dict, Union
import pandas as pd

from intuitionbuilder.helpers import Candidate
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd


def make_candidate(m: np.array,
                   ) -> Candidate:

    # convert matrix to RVs
    xs, ys = to_pyitlib_format(m)

    u, s, v = randomized_svd(m, n_components=m.shape[1])

    return Candidate(matrix=m,
                     df=pd.DataFrame(data=to_columnar(m)),  # matrix in df format
                     hxy=drv.entropy_conditional(xs, ys).round(2),
                     hyx=drv.entropy_conditional(ys, xs).round(2),
                     ami=adjusted_mutual_info_score(xs, ys, average_method="arithmetic").round(2),
                     s1p=s[0] / s.sum(),
                     )


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


def to_columnar(m: np.array,
                ) -> Dict[str, List[Union[float, int]]]:
    """
    convert a matrix into a dict,
    where each entry corresponds to a single matrix element:
    - the row index
    - the col index
    - the element's value

      """
    res = {'x': [],
           'y': [],
           'c': [],
           }

    for yi in range(m.shape[0]):
        for xi in range(m.shape[1]):
            res['y'].append(yi)
            res['x'].append(xi)
            res['c'].append(m[yi, xi])

    return res
