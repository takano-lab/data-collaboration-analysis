from typing import Optional, Tuple, TypeVar

import numpy as np
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SVDScratch:
    """フル SVD を計算し、上位 *k* 成分を保持する。

    Parameters
    ----------
    n_components : int | None
        残す成分数。``None`` は全成分。
    center : bool, default=False
        ``True`` なら列平均でセンタリング（PCA 相当）。
    full_matrices : bool, default=False
        ``np.linalg.svd`` の `full_matrices` にそのまま渡す。
    """

    def __init__(self, n_components=None, *, center=False, full_matrices=False):
        self.n_components = n_components
        self.center = center
        self.full_matrices = full_matrices
        # 学習後にセット
        self.mean_ = None          # 列平均
        self.components_ = None    # (k, d)
        self.singular_values_ = None  # (k,)

    # ------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.center:
            self.mean_ = X.mean(axis=0)
            X = X - self.mean_
        U, S, Vt = np.linalg.svd(X, full_matrices=self.full_matrices)
        k = self.n_components or len(S)
        self.singular_values_ = S[:k]
        self.components_ = Vt[:k]
        return self

    # ------------------------------------------------------------
    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("まず fit を呼んでください")
        X = np.asarray(X, dtype=float)
        if self.center and self.mean_ is not None:
            X = X - self.mean_
        return X @ self.components_.T  # (n_samples, k)

    # ------------------------------------------------------------
    def inverse_transform(self, X_proj):
        X_rec = X_proj @ self.components_
        if self.center and self.mean_ is not None:
            X_rec += self.mean_
        return X_rec

    # ------------------------------------------------------------
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def reduce_dimensions_with_svd(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    anchor: Optional[np.ndarray] = None,
    F_type = "svd",
) -> Tuple[np.ndarray, ...]:

    # --- SVD / KernelPCA の選択 ---
    # USE_KERNEL = n_components >= X_train.shape[1]

    if F_type == "kernel_pca":
    # --- スケーリング ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        anchor_scaled = scaler.transform(anchor) if anchor is not None else None
        gamma = 1.0 / X_train.shape[1]
        model = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=gamma,
            eigen_solver="auto",
            n_jobs=-1,
        )
        # --- フィッティングと変換 ---
        X_train_svd = model.fit_transform(X_train_scaled)
        X_test_svd = model.transform(X_test_scaled)
        
        if anchor_scaled is not None:
            X_anchor_svd = model.transform(anchor_scaled)
            return X_train_svd, X_test_svd, X_anchor_svd

        return X_train_svd, X_test_svd
    
    else:
        model = SVDScratch(n_components=n_components, center=True)
        # --- フィッティングと変換 ---
        X_train_svd = model.fit_transform(X_train)
        X_test_svd = model.transform(X_test)

        if anchor is not None:
            X_anchor_svd = model.transform(anchor)
            return X_train_svd, X_test_svd, X_anchor_svd

        return X_train_svd, X_test_svd


def reduce_dimensions_with_svd_(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    anchor: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    #if n_components <= min(X_train.shape[1]) - 1: こうならないようにデータを注意
    if True:
        svd = SVDScratch(n_components=n_components, center=True)
        svd.fit(X_train)
        X_train_svd = svd.transform(X_train)
        X_test_svd = svd.transform(X_test)
    elif False:
        # スクラッチで実装したほうが良い
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X_train)
        X_train_svd = svd.transform(X_train)
        X_test_svd = svd.transform(X_test)
    else:
        svd = PCA(n_components=n_components, svd_solver='full')
        svd.fit(X_train)
        X_train_svd = svd.transform(X_train)
        X_test_svd = svd.transform(X_test)
    if anchor is not None:
        return X_train_svd, X_test_svd, svd.transform(anchor)
    # print("X_train_svd.shape:", X_train_svd.shape, "X_test_svd.shape", X_test_svd.shape)
    return X_train_svd, X_test_svd