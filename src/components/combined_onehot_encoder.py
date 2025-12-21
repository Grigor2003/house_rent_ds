from typing import Any


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def joint_one_hot(X, sizes):

    N = X.shape[0]
    split_points = np.cumsum(sizes[:-1])
    # groups = [g1, g2, ..., gn]
    groups = np.split(X, split_points, axis=1)

    # indices = [n1, n2, ..., nn]
    indices = [np.argmax(g, axis=1) for g in groups]

    multipliers = np.cumprod(sizes[::-1])[::-1]
    multipliers = np.append(multipliers[1:], 1)

    joint_index = sum(idx * m for idx, m in zip(indices, multipliers))

    joint = np.zeros((N, np.prod(sizes)), dtype=int)
    joint[np.arange(N), joint_index] = 1

    return joint


class CombinedOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_indices, sizes, keep_originals=False):
        self.column_indices = column_indices
        self.sizes = sizes
        self.keep_originals = keep_originals
    
    def fit(self, X, y=None):
        if sum(self.sizes) != len(self.column_indices):
            raise ValueError(
                f"Sum of sizes ({sum(self.sizes)}) must equal "
                f"the number of column_indices ({len(self.column_indices)})"
            )
        return self
    
    def transform(self, X):
        X = np.array(X)
        
        columns_to_combine = X[:, self.column_indices]
        
        combined = joint_one_hot(columns_to_combine, self.sizes)
        
        if self.keep_originals:
            # Keep original columns, just append combined at the end
            result = np.hstack([X, combined])
        else:
            # Remove original columns
            all_indices = set(range(X.shape[1]))
            indices_to_keep = sorted(all_indices - set(self.column_indices))
            X_remaining = X[:, indices_to_keep]
            result = np.hstack([X_remaining, combined])
        
        return result

