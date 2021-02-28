"""
Define pairs of matrices with same size and sum, differing in pattern of elements.
"""
import numpy as np
from typing import Tuple

from intuitionbuilder import configs


def get_pair(pair_id: int,
             ) -> Tuple[np.array, np.array]:
    if pair_id == 1:
        return pair1()
    elif pair_id == 2:
        return pair2()
    elif pair_id == 3:
        return pair3()
    elif pair_id == 4:
        return pair4()
    elif pair_id == 5:
        return pair5()
    elif pair_id == 6:
        return pair6()
    elif pair_id == 7:
        return pair7()
    elif pair_id == 8:
        return pair8()
    elif pair_id == 9:
        return pair9()
    else:
        raise ValueError(f'IDd not find data for pair ID={pair_id}')


def pair1(num_tosses: int = 200,
          log_space_start: int = 1,  # higher values result in more frequent co-occurrences occurring more frequently
          n_clusters: int = 4,
          ) -> Tuple[np.array, np.array]:
    def roll(row_id):
        """responsible for cluster creation, by shifting a row by shift_amount"""
        cluster_id = row_id // n_clusters
        shift_amount = cluster_id * n_clusters
        return shift_amount

    tmp = np.logspace(log_space_start, 0, num=configs.Mat.shape[1])
    ps = tmp / np.sum(tmp)
    m1 = np.random.multinomial(num_tosses, ps, size=configs.Mat.shape[0])

    m2 = np.vstack(
        [np.roll(m1[i], roll(i))
         for i in range(m1.shape[0])]
    )

    return m1, m2


def pair2(fill_val: int = 64,
          ) -> Tuple[np.array, np.array]:
    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int) * fill_val
    z = np.ones(inner_shape, dtype=np.int) - 1

    m1 = np.block(
        [[o, z],
         [z, o]]
    )

    m2 = np.block(
        [[o, z],
         [o, z],
         [z, o]]
    )[len(o) // 2:len(o) // 2 + len(m1)]

    return m1, m2


def pair3(
        fill_val1: int = 16,
        fill_val2: int = 8,
        n_cols: int = 16,
) -> Tuple[np.array, np.array]:
    m1 = np.eye(n_cols, dtype=np.int) * fill_val1
    m1[0, 1] = fill_val2
    m1[1:, 0] = fill_val2

    m2 = np.eye(*m1.shape, dtype=np.int) * fill_val1
    for i in range(len(m2)):
        if i == 0 or i % 2 == 0:
            j = i + 1
        else:
            j = i - 1
        m2[i, j] = fill_val2

    return m1, m2


def pair4(fill_val: int = 16,
          ) -> Tuple[np.array, np.array]:
    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int) * fill_val
    z = np.ones(inner_shape, dtype=np.int) - 1

    m1 = np.block(
        [[o, z],
         [z, o]]
    )

    m2 = m1.copy()
    m2 = m2.flatten()
    m2 = np.random.permutation(m2).reshape(m1.shape)

    return m1, m2


def pair5(fill_val: int = 64,
          ) -> Tuple[np.array, np.array]:
    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int)
    z = np.ones(inner_shape, dtype=np.int) - 1

    m1 = np.block(
        [[o, z],
         [z, o]]
    ) * fill_val

    data1 = []
    for n, row in enumerate(m1):
        data1.append(np.roll(row, n))
    m2 = np.vstack(data1)

    return m1, m2


def pair6(fill_val1: int = 64,
          ) -> Tuple[np.array, np.array]:

    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int) * fill_val1
    z = np.ones(inner_shape, dtype=np.int) - 1

    m1 = np.block(
        [[o, z],
         [z, o]]
    )

    m2 = m1.copy()
    m2[:, 0] = 0
    m2[:, -1] = 0
    m2[:, inner_shape[0] - 1] = fill_val1
    m2[:, inner_shape[0] - 0] = fill_val1

    return m1, m2


def pair7(fill_val1: int = 64,
          ) -> Tuple[np.array, np.array]:

    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int) * fill_val1
    z = np.ones(inner_shape, dtype=np.int) - 1
    m1 = np.block(
        [[o, z],
         [z, o]]
    )

    z1 = z.copy()
    o1 = o.copy()
    np.fill_diagonal(o1, 0)
    np.fill_diagonal(z1, fill_val1)
    m2 = np.block(
        [[o1, z1],
         [z1, o1]]
    )

    return m1, m2


def pair8(fill_val1: int = 64,
          ) -> Tuple[np.array, np.array]:
    inner_size = configs.Mat.shape[0] // 4
    f = np.eye(inner_size, dtype=np.int) * fill_val1
    e = np.zeros_like(f)
    n = lambda: 1  # np.random.randint(1, size=f.shape)

    m1 = np.block(
        [
            [f + n(), e + n(), e + n(), e + n()],
            [e + n(), f + n(), e + n(), e + n()],
            [e + n(), e + n(), f + n(), e + n()],
            [e + n(), e + n(), e + n(), f + n()],
        ]
    )

    m2 = np.block(
        [
            [f + n(), e + n(), e + n(), e + n()],
            [f + n(), e + n(), e + n(), e + n()],
            [f + n(), e + n(), e + n(), e + n()],
            [f + n(), e + n(), e + n(), e + n()],
        ]
    )

    return m1, m2


def pair9(fill_val1: int = 64,
          difference: int = 8,
          ) -> Tuple[np.array, np.array]:
    inner_shape = [i // 2 for i in configs.Mat.shape]

    o = np.ones(inner_shape, dtype=np.int) * fill_val1
    z = np.ones(inner_shape, dtype=np.int) - 1

    m1 = np.block(
        [[o, z],
         [z, o]]
    )

    l = np.ones((inner_shape[0], inner_shape[1] // 2), dtype=np.int) * fill_val1 + difference
    s = np.ones((inner_shape[0], inner_shape[1] // 2), dtype=np.int) * fill_val1 - difference
    m2 = np.block(
        [[l, s, z],
         [z, s, l]]
    )

    return m1, m2
