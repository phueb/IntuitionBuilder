
"""
A visual demonstration of how mutual information, and conditional entropy change
when parts of the main diagonal are moved

"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd

from infotheorydemo.figs import plot_heatmap
from infotheorydemo.utils import to_pyitlib_format

FILL_VAL1 = 64

f = np.eye(4, dtype=np.int) * FILL_VAL1
e = np.zeros_like(f)
n = lambda: 1 # np.random.randint(1, size=f.shape)

co_mat1 = np.block(
    [
        [f + n(), e + n(), e + n(), e + n()],
        [e + n(), f + n(), e + n(), e + n()],
        [e + n(), e + n(), f + n(), e + n()],
        [e + n(), e + n(), e + n(), f + n()],
    ]
)

co_mat2 = np.block(
    [
        [f + n(), e + n(), e + n(), e + n()],
        [f + n(), e + n(), e + n(), e + n()],
        [f + n(), e + n(), e + n(), e + n()],
        [f + n(), e + n(), e + n(), e + n()],
    ]
)


for co_mat in [co_mat1, co_mat2]:
    co_mat = co_mat[:, ~np.all(co_mat == 0, axis=0)]

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
    u, s, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True, linewidth=128):
        print(s)
