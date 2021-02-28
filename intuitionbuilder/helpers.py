from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    matrix: np.array
    df: pd.DataFrame

    # info-theoretic values
    hxy: float
    hyx: float
    ami: float

    # result of svd
    s1p: float  # proportion of variance explained by s1
