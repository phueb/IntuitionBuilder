
"""
A visual demonstration of how mutual information, variation information, and joint entropy are related,
and change with the co-occurrence structure between two discrete random variables

"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score

from infotheorydemo.figs import plot_heatmap
from infotheorydemo.utils import to_pyitlib_format


FILL_VAL1 = 32
FILL_VAL2 = 2
NUM_COLS = 32

co_mat1 = np.eye(NUM_COLS, dtype=np.int) * FILL_VAL1
co_mat1[0, 1] = FILL_VAL2
co_mat1[1:, 0] = FILL_VAL2

co_mat2 = np.eye(*co_mat1.shape, dtype=np.int) * FILL_VAL1
for i in range(len(co_mat2)):
    if i == 0 or i % 2 == 0:
        j = i + 1
    else:
        j = i - 1
    co_mat2[i, j] = FILL_VAL2



for co_mat in [co_mat1, co_mat2]:

    print()
    print(co_mat.sum())
    print(co_mat.shape)
    plot_heatmap(co_mat, [], [])

    # convert data into format used by pyitlib
    xs, ys = to_pyitlib_format(co_mat)

    print(f'xye={drv.entropy_conditional(xs, ys):>6.3f}')
    print(f'yxe={drv.entropy_conditional(ys, xs):>6.3f}')
    print(f' mi={drv.information_mutual(xs, ys):>6.3f}')
    print(f'nmi={drv.information_mutual_normalised(xs, ys):>6.3f}')
    print(f'ami={adjusted_mutual_info_score(xs, ys, average_method="arithmetic"):>6.3f}')
