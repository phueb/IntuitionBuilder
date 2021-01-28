
"""
A visual demonstration of how mutual information, and conditional entropy change
when there is more overlap but overlapping regions are less frequent than non-overlapping

"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd

from infotheorydemo.figs import plot_heatmap
from infotheorydemo.utils import to_pyitlib_format

shape = (16, 16)
FILL_VAL1 = 32
DIFF = 16


o = np.ones(shape, dtype=np.int) * FILL_VAL1
z = np.ones(shape, dtype=np.int) - 1
co_mat1 = np.block(
    [[o, z],
     [z, o]]
)


co_mat2 = np.block(
    [[o - DIFF, z + DIFF],
     [z + DIFF, o - DIFF]]
)


for co_mat in [co_mat1, co_mat2]:

    print()
    print(co_mat.sum())
    print(co_mat.shape)
    plot_heatmap(co_mat, [], [])

    # convert data into format used by pyitlib
    xs, ys = to_pyitlib_format(co_mat)

    print(f' xy={drv.entropy_conditional(xs, ys):>6.3f}')
    print(f' yx={drv.entropy_conditional(ys, xs):>6.3f}')
    print(f'ami={adjusted_mutual_info_score(xs, ys, average_method="arithmetic"):>6.3f}')

    # factor analysis
    print('Factor analysis...')
    u, o, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True):
        print(o[:2])