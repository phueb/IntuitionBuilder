"""
A visual demonstration of how unadjusted mutual information is influenced by
the alphabet size (number of types in each RV) and number of samples (N)

"""

from pyitlib import discrete_random_variable as drv
import numpy as np


for N in [100, 10_000, 1_000_000]:
    print()
    print(f'N={N}')
    NUM_TARGETS = 1024

    xs = np.random.randint(0, NUM_TARGETS, N)

    for NUM_CONTEXTS in [1024, 512, 256]:
        ys = np.random.randint(0, NUM_CONTEXTS, N)

        print(f' mi={drv.information_mutual(xs, ys):>6.3f}')
