
"""
A visual demonstration of how mutual information, variation information, and joint entropy are related,
and change with the co-occurrence structure between two discrete random variables

"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd

from infotheorydemo.figs import plot_heatmap
from infotheorydemo.utils import to_pyitlib_format

shape = (16, 16)
o = np.ones(shape, dtype=np.int)
z = np.ones(shape, dtype=np.int) - 1

co_mat1 = np.block(
    [[o, z],
     [z, o]]
)

co_mat2 = np.eye(co_mat1.shape[0], dtype=np.int) * np.sum(co_mat1) // (shape[0] * 2)

co_mat3 = co_mat1.copy()
co_mat3 = co_mat3.flatten()
co_mat3 = np.random.permutation(co_mat3).reshape(co_mat1.shape)

for co_mat in [co_mat1, co_mat2, co_mat3]:

    print()
    print(co_mat.sum())
    print(co_mat.shape)
    plot_heatmap(co_mat, [], [])

    # convert data into format used by pyitlib
    xs, ys = to_pyitlib_format(co_mat)

    print(f' mi={drv.information_mutual(xs, ys):>6.3f}')
    print(f'nmi={drv.information_mutual_normalised(xs, ys):>6.3f}')
    print(f'ami={adjusted_mutual_info_score(xs, ys, average_method="arithmetic"):>6.3f}')

    # factor analysis
    print('Factor analysis...')
    u, s, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True):
        print(s[:2])