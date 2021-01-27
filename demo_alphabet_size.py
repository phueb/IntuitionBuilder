"""
A visual demonstration of how mutual information, variation information, and joint entropy are related,
and change with the alphabet size (number of types in each RV) and number of samples (N)

"""


from pyitlib import discrete_random_variable as drv
import numpy as np


for N in [100, 10_000, 1_000_000]:
    print()
    print(f'N={N}')
    NUM_TARGETS = 1024

    targets = np.random.randint(0, NUM_TARGETS, N)

    for NUM_CONTEXTS in [1024, 512, 256]:
        contexts = np.random.randint(0, NUM_CONTEXTS, N)

        mi_drv = drv.information_mutual(targets, contexts)
        vi_drv = drv.information_variation(targets, contexts)
        xy = np.vstack((targets, contexts))
        je_drv = drv.entropy_joint(xy)  # rvs in rows
        diff = je_drv - vi_drv
        print(' '.join([f'{mi:.4f}' for mi in [mi_drv, vi_drv, je_drv, diff]]))
