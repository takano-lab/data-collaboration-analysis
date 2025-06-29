from typing import Optional, TypeVar

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


def reduce_dimensions_with_svd(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    anchor: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    #if n_components <= min(X_train.shape[1]) - 1:
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X_train)
    X_train_svd = svd.transform(X_train)
    X_test_svd = svd.transform(X_test)
    #else:
    #    svd = TruncatedSVD(n_components=n_components, svd_solver='full')
    #    svd.fit(X_train)
    #    X_train_svd = svd.transform(X_train)
    #    X_test_svd = svd.transform(X_test)
    if anchor is not None:
        return X_train_svd, X_test_svd, svd.transform(anchor)
    # print("X_train_svd.shape:", X_train_svd.shape, "X_test_svd.shape", X_test_svd.shape)
    return X_train_svd, X_test_svd