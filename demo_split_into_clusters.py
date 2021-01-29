
"""
A visual demonstration of how mutual information, and conditional entropy change
with ?

"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.utils.extmath import randomized_svd

from infotheorydemo.figs import plot_heatmap
from infotheorydemo.utils import to_pyitlib_format

SHAPE = (16, 16)
NUM_TOSSES = 1000
SHARPNESS = 1  # higher values result in more frequent co-occurrences occurring more frequently

CLUSTER_SIZE = 4


def roll(row_id):
    """responsible for cluster creation, by shifting a row by shift_amount"""
    cluster_id = row_id // CLUSTER_SIZE
    shift_amount = cluster_id * CLUSTER_SIZE
    return shift_amount


tmp = np.logspace(SHARPNESS, 0, num=SHAPE[1])
ps = tmp / np.sum(tmp)
co_mat1 = np.random.multinomial(NUM_TOSSES, ps, size=SHAPE[0])

co_mat2 = np.vstack(
    [np.roll(co_mat1[i], roll(i))
     for i in range(co_mat1.shape[0])]
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
    u, s, v = randomized_svd(co_mat, n_components=co_mat.shape[1])
    with np.printoptions(precision=2, suppress=True):
        print(s[:2])
        print(s[0] - s[1])