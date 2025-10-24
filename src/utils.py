from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# torch, UMAP は遅延インポート（_run_umap 内で import）

# ============================================================
# 次元削減アルゴリズム（定義）
# ============================================================

class LPPScratch:
    def __init__(self, n_components: int = 2, t: float = 0.01, k_neighbors: int = 5):
        self.n_components = n_components
        self.t = t
        self.k_neighbors = k_neighbors
        self.A: Optional[np.ndarray] = None  # 射影行列

    def _construct_weight_matrix(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        knn.fit(X)
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            neighbors = knn.kneighbors([X[i]], return_distance=False)[0]
            for j in neighbors:
                if i != j:
                    diff = X[i] - X[j]
                    W[i, j] = W[j, i] = np.exp(-np.dot(diff, diff) / self.t)
        return W

    def fit(self, X: np.ndarray) -> "LPPScratch":
        X = X.astype(np.float64)
        W = self._construct_weight_matrix(X)
        D = np.diag(W.sum(axis=1))
        L = D - W
        XT_D_X = X.T @ D @ X
        XT_L_X = X.T @ L @ X
        reg = 1e-9 * np.eye(XT_D_X.shape[0])  # 正則化
        try:
            eigvals, eigvecs = eigh(XT_L_X, XT_D_X + reg)
        except np.linalg.LinAlgError:
            # フォールバック（単位行列の先頭列）
            self.A = np.eye(X.shape[1])[:, :self.n_components]
            return self
        sorted_indices = np.argsort(eigvals)
        self.A = eigvecs[:, sorted_indices[:self.n_components]]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("LPPScratch: fit を先に呼んでください")
        return X @ self.A

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SVDScratch:
    """
    フル SVD（np.linalg.svd）で上位 k 成分を保持。k が実数成分より大きい場合はゼロでパディング。
    """
    def __init__(self, n_components: Optional[int] = None, *, center: bool = False, full_matrices: bool = False):
        self.n_components = n_components
        self.center = center
        self.full_matrices = full_matrices
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # (k, d)
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SVDScratch":
        X = np.asarray(X, dtype=float)
        if self.center:
            self.mean_ = X.mean(axis=0)
            X = X - self.mean_
        U, S, Vt = np.linalg.svd(X, full_matrices=self.full_matrices)
        k = self.n_components if self.n_components is not None else len(S)
        actual_k = len(S)
        if k > actual_k:
            padded_S = np.zeros(k)
            padded_S[:actual_k] = S
            self.singular_values_ = padded_S
            padded_Vt = np.zeros((k, X.shape[1]))
            padded_Vt[:actual_k, :] = Vt[:actual_k, :]
            self.components_ = padded_Vt
        else:
            self.singular_values_ = S[:k]
            self.components_ = Vt[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("SVDScratch: fit を先に呼んでください")
        X = np.asarray(X, dtype=float)
        if self.center and self.mean_ is not None:
            X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        X_rec = X_proj @ self.components_
        if self.center and self.mean_ is not None:
            X_rec += self.mean_
        return X_rec

class KCCAScratch:
    """
    Kernel CCA（X と Y の相関最大化）
    """
    def __init__(self, n_components: int, reg: float = 1e-4, kernel_x: str = 'rbf', kernel_y: str = 'linear',
                 gamma_x: Optional[float] = None, gamma_y: Optional[float] = None):
        self.n_components = n_components
        self.reg = reg
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.alpha: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None

    def _get_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None, kernel_type: str = 'rbf',
                    gamma: Optional[float] = None) -> np.ndarray:
        from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
        if kernel_type == 'rbf':
            if gamma is None:
                gamma = 1.0 / X.shape[1]
            return rbf_kernel(X, Y, gamma=gamma)
        elif kernel_type == 'poly':
            return polynomial_kernel(X, Y, degree=3, coef0=1)
        else:
            return linear_kernel(X, Y)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "KCCAScratch":
        self.X_train = X
        n = X.shape[0]
        Kx = self._get_kernel(X, kernel_type=self.kernel_x, gamma=self.gamma_x)
        Ky = self._get_kernel(Y, kernel_type=self.kernel_y, gamma=self.gamma_y)
        N = np.eye(n) - np.ones((n, n)) / n
        Kx_c = N @ Kx @ N
        Ky_c = N @ Ky @ N
        R = Kx_c @ Ky_c
        LHS = R @ Kx_c
        RHS = Kx_c @ Kx_c + self.reg * np.eye(n)
        try:
            eigvals, eigvecs = eigh(LHS, RHS)
            sorted_indices = np.argsort(eigvals)[::-1]
            self.alpha = eigvecs[:, sorted_indices[:self.n_components]]
        except np.linalg.LinAlgError:
            # フォールバック: KernelPCA の固有ベクトルで代用
            kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel_x, gamma=self.gamma_x)
            kpca.fit(X)
            self.alpha = kpca.alphas_
        return self

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.X_train is None:
            raise RuntimeError("KCCAScratch: fit を先に呼んでください")
        K_new = self._get_kernel(X_new, self.X_train, kernel_type=self.kernel_x, gamma=self.gamma_x)
        return K_new @ self.alpha

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        self.fit(X, Y)
        Kx = self._get_kernel(X, kernel_type=self.kernel_x, gamma=self.gamma_x)
        return Kx @ self.alpha

# ============================================================
# 共通ユーティリティ
# ============================================================

def self_tuning_gamma(
    X: np.ndarray, *,
    k: int = 7,
    standardize: bool = True,
    summary: str = "median"
):
    """
    k-th 最近傍距離で各点のスケールを決める自己調整 γ
    """
    X = np.asarray(X, dtype=float)
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(X)
    dists, _ = nbrs.kneighbors(X, return_distance=True)
    sigma_i = dists[:, k]
    sigma_i[sigma_i == 0] = np.finfo(float).eps
    gamma_i = 1.0 / sigma_i
    gamma_i = gamma_i / 3.0  # 調整

    if summary is None:
        return gamma_i, sigma_i
    if summary == "median":
        return float(np.median(gamma_i))
    if summary == "mean":
        return float(np.mean(gamma_i))
    raise ValueError("summary must be 'median', 'mean', or None")

def _to_tuple4(a, b, c=None, d=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    return a, b, c, d

def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    try:
        if isinstance(cfg, dict):
            v = cfg.get(name, default)
        else:
            v = getattr(cfg, name, default)
    except Exception:
        return default
    return v

def _cfg_int(cfg, name, default: int) -> int:
    v = _cfg_get(cfg, name, default)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def _cfg_float(cfg, name, default: float) -> float:
    v = _cfg_get(cfg, name, default)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return float(v)
    except Exception:
        return default

def _cfg_str(cfg, name, default: str) -> str:
    v = _cfg_get(cfg, name, default)
    if v is None:
        return default
    s = str(v).strip()
    if not s or s.lower() == "none":
        return default
    return s

# ============================================================
# 実行ラッパ（F_type ごとの実装）
# すべて「4要素タプル (train, test, anchor, anchor_test)」を返す
# ============================================================

def _run_svd(X_train, X_test, n_components, *, anchor=None, anchor_test=None, **kwargs):
    model = SVDScratch(n_components=n_components, center=True)
    Xt = model.fit_transform(X_train)
    Xv = model.transform(X_test)
    Xa = model.transform(anchor) if anchor is not None else None
    Xat = model.transform(anchor_test) if anchor_test is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


def _run_diffspan(X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs):
    svd = SVDScratch(n_components=n_components, center=True)
    Ft_train = svd.fit_transform(X_train)
    Ft_test = svd.transform(X_test)
    # 直交性を崩すランダム行列 E
    seed = 0
    if config is not None and hasattr(config, "seed") and hasattr(config, "f_seed"):
        seed = int(config.seed) + int(config.f_seed)
    rng = np.random.default_rng(seed)
    E = rng.uniform(low=-1.0, high=1.0, size=(n_components, n_components))
    Xt = Ft_train @ E
    Xv = Ft_test @ E
    Xa = svd.transform(anchor) @ E if anchor is not None else None
    Xat = svd.transform(anchor_test) @ E if anchor_test is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


def _run_samespan_orth(X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs):
    m = X_train.shape[1]
    l = n_components
    if l > m:
        raise ValueError("samespan_orth: l(=n_components) は m(特徴量) 以下である必要があります。")
    # 列直交 F'（seed: f_seed）
    f_seed = int(getattr(config, "f_seed", 0))
    rng = np.random.default_rng(f_seed)
    A = rng.standard_normal(size=(m, l))
    Q, R = np.linalg.qr(A, mode="reduced")
    signs = np.sign(np.diag(R))
    Q *= signs
    F_prime = Q
    # ランダム直交 E（seedなし）
    Rmat = np.random.standard_normal(size=(l, l))
    QE, _ = np.linalg.qr(Rmat)
    E = QE
    F = F_prime @ E
    Xt = X_train @ F
    Xv = X_test @ F
    Xa = anchor @ F if anchor is not None else None
    Xat = anchor_test @ F if anchor_test is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


def _run_samespan(X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs):
    m = X_train.shape[1]
    l = n_components
    if l > m:
        raise ValueError("samespan: l(=n_components) は m(特徴量) 以下である必要があります。")
    seed = int(getattr(config, "seed", 0))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(m, l))
    Q, R = np.linalg.qr(A, mode="reduced")
    signs = np.sign(np.diag(R))
    Q *= signs
    F_prime = Q
    E = np.random.standard_normal(size=(l, l))  # 直交までは要求しない
    F = F_prime @ E
    Xt = X_train @ F
    Xv = X_test @ F
    Xa = anchor @ F if anchor is not None else None
    Xat = anchor_test @ F if anchor_test is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


def _run_lpp(X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs):
    # 近傍数の既定: num_institution_user * 0.2（無い場合は 5）
    if config is not None and hasattr(config, "num_institution_user") and config.num_institution_user:
        k = max(1, int(config.num_institution_user * 0.2))
    else:
        k = 5
    model = LPPScratch(n_components=n_components, t=0.01, k_neighbors=k)
    Xt = model.fit_transform(X_train)
    Xv = model.transform(X_test)
    Xa = model.transform(anchor) if anchor is not None else None
    Xat = model.transform(anchor_test) if anchor_test is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


def _run_kcca(X_train, X_test, n_components, *, y_train=None, anchor=None, anchor_test=None, **kwargs):
    # スケーリング
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None

    # y がカテゴリなら OneHot
    if y_train is None:
        raise ValueError("kcca には y_train が必要です")
    if np.issubdtype(y_train.dtype, np.integer):
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Yv = enc.fit_transform(y_train.reshape(-1, 1))
    else:
        Yv = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

    gamma_x = 1.0 / X_train.shape[1]
    model = KCCAScratch(n_components=n_components, reg=1e-4, kernel_x='rbf', kernel_y='linear', gamma_x=gamma_x)
    Xt = model.fit_transform(Xts, Yv)
    Xv = model.transform(Xvs)
    Xa = model.transform(Xas) if Xas is not None else None
    # anchor_test は y が無いので変換不可
    return _to_tuple4(Xt, Xv, Xa, None)


def _run_kpca_family(X_train, X_test, n_components, *, mode: str, config=None, seed=None,
                     anchor=None, anchor_test=None, **kwargs):
    """
    mode in {"auto", "kernel_pca_self_tuning"}
    """
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None
    Xats = scaler.transform(anchor_test) if anchor_test is not None else None

    if mode == "auto":
        gamma = 1.0 / X_train.shape[1]
        ratio = float(getattr(config, "gamma_ratio", 1.0)) if config is not None else 1.0
        gamma *= ratio
    elif mode == "kernel_pca_self_tuning":
        gamma = self_tuning_gamma(Xts, standardize=False, k=7, summary='median')
        ratio = float(getattr(config, "gamma_ratio", 1.0)) if config is not None else 1.0
        gamma *= ratio
        if config is not None:
            if not hasattr(config, "nl_gammas") or config.nl_gammas is None:
                config.nl_gammas = []
            config.nl_gammas.append(gamma)
    elif mode == "kernel_pca_gamma_fixed":
        gamma = float(getattr(config, "gamma_ratio", 1.0)) if config is not None else 0.0001
        if config is not None:
            if not hasattr(config, "nl_gammas") or config.nl_gammas is None:
                config.nl_gammas = []
            config.nl_gammas.append(gamma)
    else:
        raise ValueError(f"unknown kpca mode: {mode}")
    
    model = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma, eigen_solver="auto", n_jobs=-1)
    Xt = model.fit_transform(Xts)
    Xv = model.transform(Xvs)
    Xa = model.transform(Xas) if Xas is not None else None
    Xat = model.transform(Xats) if Xats is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)

# 追加: UMAP ランナー
def _run_umap(
    X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs
):
    # 遅延インポート（使う時だけ読み込む）
    try:
        from umap import UMAP
    except Exception as e:
        raise RuntimeError("UMAPを使うには 'umap-learn' が必要です。uv sync -E umap で導入してください。") from e

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None
    Xats = scaler.transform(anchor_test) if anchor_test is not None else None

    n_neighbors = _cfg_int(config, "max_umap_nb", 15)
    min_dist = _cfg_float(config, "umap_min_dist", 0.1)
    metric = _cfg_str(config, "umap_metric", "euclidean")  # ← 修正
    seed = _cfg_int(config, "seed", 0)

    # 追加オプション（必要に応じて）
    extra_params = {}
    tm = _cfg_str(config, "umap_transform_mode", None)
    if tm:
        extra_params["transform_mode"] = tm
    rs = _cfg_float(config, "umap_repulsion_strength", None)
    if rs is not None:
        extra_params["repulsion_strength"] = float(rs)
    init = _cfg_str(config, "umap_init", None)
    if init:
        extra_params["init"] = init
    mix = _cfg_float(config, "umap_set_op_mix_ratio", None)
    if mix is not None:
        extra_params["set_op_mix_ratio"] = float(mix)

    model = UMAP(
        n_components=n_components,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
        **extra_params,
    )
    Xt = model.fit_transform(Xts)
    Xv = model.transform(Xvs)
    Xa = model.transform(Xas) if Xas is not None else None
    Xat = model.transform(Xats) if Xats is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


# 追加: Diffusion Maps ランナー（Nyström 拡張付き）
def _run_dm(
    X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs
):
    def _median_gamma(X: np.ndarray) -> float:
        D = pairwise_distances(X, metric="euclidean")
        vals = D[D > 0]
        med = np.median(vals) if vals.size else 1.0
        if not np.isfinite(med) or med <= 0:
            return 1.0
        return 1.0 / (2.0 * (med ** 2))

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None
    Xats = scaler.transform(anchor_test) if anchor_test is not None else None

    gamma = _cfg_float(config, "dm_gamma", -1.0)
    if gamma is None or gamma <= 0:
        gamma = _median_gamma(Xts)
    t = _cfg_float(config, "dm_t", 1.0)

    # K, P を学習データ上で構築
    D2 = pairwise_distances(Xts, metric="sqeuclidean")
    K = np.exp(-gamma * D2)
    np.fill_diagonal(K, 0.0)
    d = K.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    P = K / d

    w, V = np.linalg.eig(P)
    # 実部へ
    w = np.real(w)
    V = np.real(V)
    order = np.argsort(-np.abs(w))
    w = w[order]
    V = V[:, order]
    # 先頭(λ≈1)は無視して次の n_components を使用
    start = 1 if V.shape[1] > 1 else 0
    k = min(n_components, max(0, V.shape[1] - start))
    eigvals_sel = w[start:start + k]
    eigvecs_sel = V[:, start:start + k]
    # 拡散距離スケール（t乗）
    if k > 0:
        scale = np.power(np.abs(eigvals_sel), t)
        Xt = eigvecs_sel * scale
    else:
        Xt = np.zeros((Xts.shape[0], 0))

    def _embed_new(Xnew: np.ndarray) -> np.ndarray:
        if k == 0 or Xnew is None:
            return None
        D2n = pairwise_distances(Xnew, Xts, metric="sqeuclidean")
        Kny = np.exp(-gamma * D2n)
        row = Kny.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        Pny = Kny / row
        # Nyström: phi_j(y) = (1/lambda_j) sum_i P(y,i) phi_j(i) ; その後 t 乗でスケール
        Phi = (Pny @ eigvecs_sel) / np.maximum(np.abs(eigvals_sel), 1e-12)
        Phi *= np.power(np.abs(eigvals_sel), t)
        return Phi

    Xv = _embed_new(Xvs)
    Xa = _embed_new(Xas) if Xas is not None else None
    Xat = _embed_new(Xats) if Xats is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)


# 追加: Laplacian Eigenmaps ランナー（RBF+KNN, Nyström 拡張）
def _run_le(
    X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs
):
    def _median_gamma(X: np.ndarray) -> float:
        D = pairwise_distances(X, metric="euclidean")
        vals = D[D > 0]
        med = np.median(vals) if vals.size else 1.0
        if not np.isfinite(med) or med <= 0:
            return 1.0
        return 1.0 / (2.0 * (med ** 2))

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None
    Xats = scaler.transform(anchor_test) if anchor_test is not None else None

    k_nb = _cfg_int(config, "le_neighbors", 10)
    gamma = _cfg_float(config, "le_gamma", -1.0)
    if gamma is None or gamma <= 0:
        gamma = _median_gamma(Xts)

    n = Xts.shape[0]
    # KNN グラフ作成（対称化）
    nn = NearestNeighbors(n_neighbors=min(max(1, k_nb), max(1, n - 1))).fit(Xts)
    ind = nn.kneighbors(return_distance=False)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in ind[i]:
            if i == j:
                continue
            diff = Xts[i] - Xts[j]
            w = np.exp(-gamma * float(np.dot(diff, diff)))
            W[i, j] = w
            W[j, i] = max(W[j, i], w)  # 対称化（最大）
    D = np.diag(W.sum(axis=1))
    D_sqrt_inv = np.diag(1.0 / np.maximum(np.sqrt(np.diag(D)), 1e-12))
    Lsym = D_sqrt_inv @ W @ D_sqrt_inv  # = I - L_norm ではなく、類似度正規化（固有値は [0,1]）

    # 固有分解（最大固有値の次から n_components）
    evals, evecs = eigh(Lsym)
    order = np.argsort(evals)[::-1]  # 大きい順
    evals = evals[order]
    evecs = evecs[:, order]
    start = 1 if evecs.shape[1] > 1 else 0  # 先頭はトリビアル
    k = min(n_components, max(0, evecs.shape[1] - start))
    U = evecs[:, start:start + k]
    Xt = U

    # Nyström 拡張: psi_j(y) ≈ (1/sqrt(d(y))) Σ_i W(y,i)/sqrt(d_i) * psi_j(i)
    d_i_sqrt = np.sqrt(np.maximum(np.diag(D), 1e-12))

    def _embed_new(Xnew: np.ndarray) -> np.ndarray:
        if Xnew is None or k == 0:
            return None
        nn_new = NearestNeighbors(n_neighbors=min(max(1, k_nb), n)).fit(Xts)
        neigh_ind = nn_new.kneighbors(Xnew, return_distance=False)
        Z = np.zeros((Xnew.shape[0], k), dtype=float)
        for r, nbrs in enumerate(neigh_ind):
            wrow = np.zeros(n, dtype=float)
            for j in nbrs:
                diff = Xnew[r] - Xts[j]
                wrow[j] = np.exp(-gamma * float(np.dot(diff, diff)))
            d_y = wrow.sum()
            if d_y <= 0:
                continue
            coef = (wrow / np.maximum(d_y, 1e-12)) / np.maximum(d_i_sqrt, 1e-12)
            Z[r] = (coef @ U) / np.sqrt(d_y)
        return Z

    Xv = _embed_new(Xvs)
    Xa = _embed_new(Xas) if Xas is not None else None
    Xat = _embed_new(Xats) if Xats is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)

def _run_autoencoder(
    X_train, X_test, n_components, *, config=None, anchor=None, anchor_test=None, **kwargs
):
    # 遅延インポート（使う時だけ読み込む）
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except Exception as e:
        raise RuntimeError("AutoEncoderを使うには 'torch' が必要です。uv sync -E ae で導入してください。") from e

    # スケーリング
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train).astype(np.float32)
    Xvs = scaler.transform(X_test).astype(np.float32)
    Xas = scaler.transform(anchor).astype(np.float32) if anchor is not None else None
    Xats = scaler.transform(anchor_test).astype(np.float32) if anchor_test is not None else None

    # ここを安全取得に変更
    epochs = _cfg_int(config, "ae_epochs", 20)
    batch = _cfg_int(config, "ae_batch", 256)

    # モデル定義（ここで nn を使える）
    class _AE(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            xhat = self.decoder(z)
            return xhat

    def _train_torch_autoencoder(X_train_np: np.ndarray, input_dim: int, latent_dim: int) -> "_AE":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _AE(input_dim, latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        X = torch.from_numpy(X_train_np.astype(np.float32))
        ds = TensorDataset(X)
        n_total = len(ds)
        n_val = max(1, int(0.1 * n_total))
        n_tr = max(1, n_total - n_val)
        tr_set, va_set = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
        dl_tr = DataLoader(tr_set, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va_set, batch_size=batch, shuffle=False)

        best_state, best_val, patience, bad = None, float("inf"), 3, 0
        for _ in range(epochs):
            model.train()
            for (xb,) in dl_tr:
                xb = xb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), xb)
                loss.backward()
                opt.step()
            # val
            model.eval()
            va = 0.0
            with torch.no_grad():
                for (xb,) in dl_va:
                    xb = xb.to(device)
                    va += crit(model(xb), xb).item() * xb.size(0)
            va /= max(1, len(va_set))
            if va + 1e-8 < best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to("cpu")
        return model

    model = _train_torch_autoencoder(Xts, input_dim=Xts.shape[1], latent_dim=int(n_components))

    with torch.no_grad():
        Xt = model.encoder(torch.from_numpy(Xts)).numpy()
        Xv = model.encoder(torch.from_numpy(Xvs)).numpy()
        Xa = model.encoder(torch.from_numpy(Xas)).numpy() if Xas is not None else None
        Xat = model.encoder(torch.from_numpy(Xats)).numpy() if Xats is not None else None
    return _to_tuple4(Xt, Xv, Xa, Xat)

# F_type → 実行関数マップ
_RUNNERS: Dict[str, Any] = {
    "svd": _run_svd,
    "diffspan": _run_diffspan,
    "samespan_orth": _run_samespan_orth,
    "samespan": _run_samespan,
    "lpp": _run_lpp,
    "kcca": _run_kcca,
    "kernel_pca": lambda *a, **kw: _run_kpca_family(*a, mode="auto", **kw),
    "kernel_pca_self_tuning": lambda *a, **kw: _run_kpca_family(*a, mode="kernel_pca_self_tuning", **kw),
    "kernel_pca_gamma_fixed": lambda *a, **kw: _run_kpca_family(*a, mode="kernel_pca_gamma_fixed", **kw),
    # 追加
    "umap": _run_umap,
    "dm": _run_dm,
    "le": _run_le,
    "autoencoder": _run_autoencoder,
    "ae": _run_autoencoder,
}

# ============================================================
# 公開API
# ============================================================

def reduce_dimensions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    y_train: Optional[np.ndarray] = None,
    anchor: Optional[np.ndarray] = None,
    anchor_test: Optional[np.ndarray] = None,
    F_type: str = "kernel_pca",
    seed: Optional[int] = None,
    param: Any = None,
    config: Any = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    次元削減の統一エントリポイント。
    常に (X_train_reduced, X_test_reduced, anchor_reduced or None, anchor_test_reduced or None) を返す。
    """
    if F_type not in _RUNNERS:
        raise ValueError(f"未知の F_type: {F_type}")
    runner = _RUNNERS[F_type]
    return runner(
        X_train, X_test, n_components,
        y_train=y_train, anchor=anchor, anchor_test=anchor_test,
        F_type=F_type, seed=seed, param=param, config=config
    )