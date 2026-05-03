from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import pairwise_distances

# torch, UMAP は遅延インポート（_run_umap 内で import）

logger = logging.getLogger(__name__)

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
def median_heuristic_gamma(X: np.ndarray, *, standardize: bool = True) -> float:
    X = np.asarray(X, dtype=float)
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    D = pairwise_distances(X, metric="euclidean")
    # 対角は 0 なので除外
    d = np.median(D[D > 0])
    gamma = 1.0 / (2.0 * d ** 2)   # RBF: exp(-gamma ||x-y||^2)
    return float(gamma)

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

Projector = Callable[[Optional[np.ndarray]], Optional[np.ndarray]]

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


def _resolve_seed(seed: Optional[int], config, default: int = 0) -> int:
    """
    seed 引数 -> config.f_seed -> config.seed の優先度でシード値を決める
    """
    if seed is not None:
        return int(seed)
    f_seed = None
    if config is not None:
        f_seed = getattr(config, "f_seed", None)
    if f_seed is not None:
        return int(f_seed)
    return int(_cfg_int(config, "seed", default))


def build_shared_subspace_projectors(
    num_features: int,
    num_institution: int,
    config: Any,
) -> list[Projector]:
    """
    共有部分空間 F_k を用いた線形写像を各機関分まとめて構成する。

    記号:
        m = num_features
        c = num_institution
        l = dim_intermediate  (最終次元, F_k の列数)
        r = dim_integrate     (共有部分 Q の列数)

    条件（スライドに対応）:
        c(l - r) <= m - r
    """
    m = int(num_features)
    c = int(num_institution)

    # 記号対応
    l = int(_cfg_int(config, "dim_intermediate", 10))
    r = int(getattr(config, "dim_integrate", l) or l)

    # l >= r を要求
    if r > l:
        raise ValueError(
            f"shared_subspace: dim_intermediate(={l}) must be >= dim_integrate(={r})."
        )

    # 各 R_k の列数 p = l - r
    p = l - r

    # スライドの条件: c(l - r) <= m - r
    if p > 0 and c * p > m - r:
        raise ValueError(
            f"shared_subspace: c*(l-r)={c*p} must be <= m-r={m-r}. "
            "条件を満たすように dim_intermediate / dim_integrate / num_institution を調整してください."
        )

    # 乱数シード
    q_seed = _cfg_int(config, "Q_seed", 0)
    r_seed = _cfg_int(config, "R_seed", 1)
    rng_q = np.random.default_rng(q_seed)
    rng_r = np.random.default_rng(r_seed)

    # Step B: 共有基底 Q (m × r)
    G = rng_q.standard_normal(size=(m, r))
    Q, R = np.linalg.qr(G, mode="reduced")
    # 列の向きの不定性を固定
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q *= signs

    # Step C: 直交補 Q_perp (m × (m-r))
    H = rng_q.standard_normal(size=(m, m - r))
    H1 = (np.eye(m) - Q @ Q.T) @ H
    Q_perp, _ = np.linalg.qr(H1, mode="reduced")

    # 各機関の私有基底 R_k (m × p)
    R_list: list[np.ndarray] = []
    if p > 0:
        # 単純なブロック分割
        for k in range(c):
            start = k * p
            end = start + p
            if end > Q_perp.shape[1]:
                raise ValueError(
                    "shared_subspace: Q_perp の列数が不足しています。"
                    "条件 c*(l-r) <= m-r を満たすように設定してください."
                )
            Rk = Q_perp[:, start:end]
            R_list.append(Rk)

    # 微小回転用の角度・スケール
    theta_ss = float(getattr(config, "theta_ss", 0.0) or 0.0)
    gamma_ss = float(getattr(config, "gamma_ss", 1.0) or 1.0)

    # Step 5: 微小回転用 U_k（θ_ss ≠ 0 のときだけ）
    Uk_list: list[np.ndarray] = []
    if abs(theta_ss) > 0.0:
        # U_k は Q^T U_k = 0 を満たすように構成
        for _ in range(c):
            Bk = rng_r.standard_normal(size=(m, r))
            Hk = (np.eye(m) - Q @ Q.T) @ Bk
            Uk, _ = np.linalg.qr(Hk, mode="reduced")
            Uk_list.append(Uk[:, :r])

    projectors: list[Projector] = []
    half_c = c // 2  # 前半・後半を分ける境目（0〜half_c-1 が前半）

    for k in range(c):
        Qk = Q
        if abs(theta_ss) > 0.0 and Uk_list:
            Uk = Uk_list[k]
            Qk = Q * np.cos(theta_ss) + Uk * np.sin(theta_ss)

        if p > 0:
            Rk = R_list[k]
            # 前半の機関だけ gamma_ss を掛ける、後半は 1.0
            gamma_k = gamma_ss if k < half_c else 1.0
            Fk = np.concatenate([Qk, gamma_k * Rk], axis=1)  # m × l
        else:
            Fk = Qk  # 私有成分なし (l == r)

        def _make_projector(F: np.ndarray) -> Projector:
            def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if data is None:
                    return None
                return np.asarray(data, dtype=float) @ F

            return projector

        projectors.append(_make_projector(Fk))

    return projectors

# ============================================================
# 実行ラッパ（F_type ごとの実装）
# すべて「4要素タプル (train, test, anchor, anchor_test)」を返す
# ============================================================

def _run_svd(X, n_components, *, config=None, **kwargs) -> Projector:
    raw_flag = getattr(config, "svd_standardize_before", False) if config is not None else False
    if isinstance(raw_flag, str):
        svd_standardize_before = raw_flag.strip().lower() in {"1", "true", "yes", "on"}
    else:
        svd_standardize_before = bool(raw_flag)

    scaler = StandardScaler() if svd_standardize_before else None
    X_fit = scaler.fit_transform(X) if scaler is not None else X

    model = SVDScratch(n_components=n_components, center=True)
    model.fit(X_fit)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        data_use = scaler.transform(data) if scaler is not None else data
        return model.transform(data_use)

    return projector


def _run_diffspan(X, n_components, *, config=None, seed=None, **kwargs) -> Projector:
    # データ駆動の基底（ノートブックと同様に上位特異ベクトルを利用）
    svd = SVDScratch(n_components=n_components, center=False)
    svd.fit(X)
    basis = svd.components_.T  # m × l
    seed_val = _resolve_seed(seed, config)
    rng = np.random.default_rng(seed_val)
    E = rng.uniform(low=-1.0, high=1.0, size=(n_components, n_components))

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return np.asarray(data, dtype=float) @ basis @ E

    return projector


def _run_diffspan_orth(X, n_components, *, config=None, seed=None, **kwargs) -> Projector:
    """
    diffspan_orth: ほぼ SVD と同じ振る舞い（直交基底）
    """
    model = SVDScratch(n_components=n_components, center=False)
    model.fit(X)
    base = model.components_.T  # m × l
    seed_val = _resolve_seed(seed, config)
    rng = np.random.default_rng(seed_val)
    Rmat = rng.standard_normal(size=(n_components, n_components))
    Qrot, _ = np.linalg.qr(Rmat)
    F = base @ Qrot  # 同じスパン内で直交回転

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return np.asarray(data, dtype=float) @ F

    return projector


def _run_samespan_orth(X, n_components, *, config=None, seed=None, **kwargs) -> Projector:
    m = X.shape[1]
    l = n_components
    if l > m:
        raise ValueError("samespan_orth: l(=n_components) > 特徴次元 は未対応です")
    seed_val = _resolve_seed(seed, config)
    rng = np.random.default_rng(seed_val)

    shared = None
    if config is not None:
        shared = getattr(config, "_shared_F_basis", None)

    if shared is not None:
        base = np.asarray(shared, dtype=float)
        if base.shape[0] != m:
            raise ValueError(f"samespan_orth: shared basis shape mismatch {base.shape} vs {m}")
        base = base[:, :l]
    else:
        # フォールバック: ローカルデータで SVD 基底を作る
        svd = SVDScratch(n_components=l, center=False)
        svd.fit(X)
        base = svd.components_.T

    Rmat = rng.standard_normal(size=(l, l))
    QE, _ = np.linalg.qr(Rmat)
    F = base @ QE

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return data @ F

    return projector

def _run_samespan(X, n_components, *, config=None, seed=None, **kwargs) -> Projector:
    m = X.shape[1]
    l = n_components
    if l > m:
        raise ValueError("samespan: l(=n_components) > 特徴次元 は未対応です")
    seed_val = _resolve_seed(seed, config)
    rng = np.random.default_rng(seed_val)

    shared = None
    if config is not None:
        shared = getattr(config, "_shared_F_basis", None)

    if shared is not None:
        base = np.asarray(shared, dtype=float)
        if base.shape[0] != m:
            raise ValueError(f"samespan: shared basis shape mismatch {base.shape} vs {m}")
        base = base[:, :l]
    else:
        svd = SVDScratch(n_components=l, center=False)
        svd.fit(X)
        base = svd.components_.T

    E = rng.standard_normal(size=(l, l))
    F = base @ E

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return data @ F

    return projector

def _run_random_projection(
    X,
    n_components,
    *,
    config=None,
    seed=None,
    **kwargs,
) -> Projector:
    seed_val = seed if seed is not None else _cfg_int(config, "seed", 0)
    proj_kind = _cfg_str(config, "random_projection_type", "gaussian").lower()
    if proj_kind not in ("gaussian", "sparse"):
        proj_kind = "gaussian"

    model_kwargs: Dict[str, Any] = {"n_components": n_components}
    if seed_val is not None:
        model_kwargs["random_state"] = int(seed_val)

    if proj_kind == "sparse":
        density = _cfg_float(config, "random_projection_density", -1.0)
        if density > 0.0:
            model_kwargs["density"] = float(max(min(density, 1.0), 1e-12))
        model = SparseRandomProjection(**model_kwargs)
    else:
        model = GaussianRandomProjection(**model_kwargs)

    model.fit(X)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return model.transform(data)

    return projector


def _run_lpp(X, n_components, *, config=None, **kwargs) -> Projector:
    if config is not None and hasattr(config, "num_institution_user") and config.num_institution_user:
        k = max(1, int(config.num_institution_user * 0.2))
    else:
        k = 5
    model = LPPScratch(n_components=n_components, t=0.01, k_neighbors=k)
    model.fit(X)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return model.transform(data)

    return projector

def _run_kcca(X, n_components, *, y=None, **kwargs) -> Projector:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if y is None:
        raise ValueError("kcca には y が必要です")
    if np.issubdtype(np.asarray(y).dtype, np.integer):
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Yv = enc.fit_transform(np.asarray(y).reshape(-1, 1))
    else:
        y_arr = np.asarray(y)
        Yv = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    gamma_x = 1.0 / X.shape[1]
    model = KCCAScratch(n_components=n_components, reg=1e-4, kernel_x='rbf', kernel_y='linear', gamma_x=gamma_x)
    model.fit(Xs, Yv)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        data_scaled = scaler.transform(data)
        return model.transform(data_scaled)

    return projector

def _run_kpca_family(X, n_components, *, mode: str, config=None, seed=None, **kwargs) -> Projector:
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X)

    def _safe_ratio(default: float = 1.0) -> float:
        if config is None:
            return float(default)
        raw = getattr(config, "gamma_ratio", default)
        if raw in (None, ""):
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    if mode == "auto":
        gamma = 1.0 / X.shape[1]
        ratio = _safe_ratio(1.0)
        gamma *= ratio
    elif mode == "kernel_pca_self_tuning":
        #gamma = self_tuning_gamma(Xts, standardize=False, k=7, summary='median')
        gamma = median_heuristic_gamma(Xts, standardize=False)
        ratio = _safe_ratio(1.0)
        gamma *= ratio
    elif mode == "kernel_pca_gamma_fixed":
        gamma = _safe_ratio(1e-4)
    else:
        raise ValueError(f"unknown kpca mode: {mode}")

    model = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma, eigen_solver="auto", n_jobs=-1)
    model.fit(Xts)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        data_scaled = scaler.transform(data)
        return model.transform(data_scaled)

    return projector

def _resolve_umap_seed(config=None, seed=None, *, label: str = "UMAP") -> int:
    seed_val = seed
    if seed_val is None and config is not None:
        seed_val = getattr(config, "f_seed", None)
        print(f"{label} f_seed: {seed_val}")
    if seed_val is None:
        seed_val = _cfg_int(config, "seed", 0)
        print(f"{label} seed: {seed_val}")
    resolved = int(seed_val)
    print(f"{label} seed: {resolved}")
    return resolved


def _run_umap_with_ranges(
    X,
    n_components,
    *,
    config=None,
    seed=None,
    label: str = "UMAP",
    neighbor_low: int = 2,
    neighbor_high: int = 8,
    min_dist_low: float = 0.0,
    min_dist_high: float = 0.8,
) -> Projector:
    try:
        from umap import UMAP
    except Exception as e:
        raise RuntimeError("UMAP を利用するには 'umap-learn' のインストールが必要です") from e

    n_samples = X.shape[0]
    min_required = max(3, n_components + 1)
    if n_samples <= min_required:
        logger.warning(
            "[WARN] UMAP: サンプル数 %s 件 < 必要最小 %s 件のため SVD フォールバックを利用します",
            n_samples,
            min_required,
        )
        return _run_svd(X, n_components, config=config)

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X)

    # metric は seed%3 で切替、n_neighbors と min_dist は seed に応じてランダム化。
    # 明示的に渡された seed -> f_seed -> seed の優先順位で決定
    seed = _resolve_umap_seed(config=config, seed=seed, label=label)
    metric_choices = ("correlation", "cosine", "euclidean")
    metric = metric_choices[seed % 3]
    rng = np.random.default_rng(seed + 1337)
    nn_sample = int(rng.integers(low=neighbor_low, high=neighbor_high))  # high は排他的
    max_valid_neighbors = max(2, Xts.shape[0] - 1)
    n_neighbors = min(max_valid_neighbors, max(2, nn_sample))
    min_dist = float(rng.uniform(min_dist_low, min_dist_high))

    # vis circle
    # min_dist = 0.5
    # n_neighbors = 50
    # metric = "euclidean"
    # init="spectral"
    # spread=1.0

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
    
    #extra_params["repulsion_strength"] = 2.0

    model = UMAP(
        n_components=n_components,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
        **extra_params,
    )
    model.fit(Xts)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        data_scaled = scaler.transform(data)
        return model.transform(data_scaled)

    return projector


def _run_umap(
    X, n_components, *, config=None, seed=None, **kwargs
) -> Projector:
    return _run_umap_with_ranges(
        X,
        n_components,
        config=config,
        seed=seed,
        label="UMAP",
        neighbor_low=2,
        neighbor_high=8,
        min_dist_low=0.0,
        min_dist_high=0.8,
    )


def _run_umap_2(
    X, n_components, *, config=None, seed=None, **kwargs
) -> Projector:
    return _run_umap_with_ranges(
        X,
        n_components,
        config=config,
        seed=seed,
        label="UMAP_2",
        neighbor_low=8,
        neighbor_high=16,
        min_dist_low=0.05,
        min_dist_high=0.3,
    )


def _run_supervised_umap(
    X, n_components, *, y=None, config=None, seed=None, **kwargs
) -> Projector:
    """
    y を用いた supervised UMAP。
    乱数に関する挙動（seed から metric / n_neighbors / min_dist を決める）は
    _run_umap と同一に保つ。
    """
    try:
        from umap import UMAP
    except Exception as e:
        raise RuntimeError("UMAP を利用するには 'umap-learn' のインストールが必要です") from e

    if y is None:
        raise ValueError("supervised_umap には y が必要です")

    y_arr = np.asarray(y)
    if y_arr.shape[0] != X.shape[0]:
        raise ValueError("supervised_umap: X と y のサンプル数が一致していません")

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X)

    # _run_umap と同じ seed ロジック
    seed_val = seed
    if seed_val is None and config is not None:
        seed_val = getattr(config, "f_seed", None)
        print(f"Supervised UMAP f_seed: {seed_val}")
    if seed_val is None:
        seed_val = _cfg_int(config, "seed", 0)
        print(f"Supervised UMAP seed: {seed_val}")
    seed = int(seed_val)
    print(f"Supervised UMAP seed: {seed}")

    metric_choices = ("correlation", "cosine", "euclidean")
    metric = metric_choices[seed % 3]
    rng = np.random.default_rng(seed + 1337)
    nn_sample = int(rng.integers(low=2, high=8))  # high は排他的上限
    n_neighbors = max(2, min(nn_sample, max(2, Xts.shape[0] - 1)))
    min_dist = float(rng.uniform(0.0, 0.8))

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

    target_metric = _cfg_str(config, "umap_target_metric", None)
    if target_metric is None:
        if np.issubdtype(y_arr.dtype, np.integer):
            target_metric = "categorical"
        else:
            target_metric = "euclidean"
    extra_params["target_metric"] = target_metric

    # ラベル分離を強めたい場合の重み（0〜1）。指定がなければやや強めの 0.8。
    tw = _cfg_float(config, "umap_target_weight", None)
    if tw is None:
        tw = 0.5
    tw = float(max(0.0, min(1.0, tw)))
    extra_params["target_weight"] = tw

    model = UMAP(
        n_components=n_components,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
        **extra_params,
    )
    model.fit(Xts, y_arr)

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        data_scaled = scaler.transform(data)
        return model.transform(data_scaled)

    return projector

def _run_dm(
    X, n_components, *, config=None, **kwargs
) -> Projector:
    def _median_gamma(X_arr: np.ndarray) -> float:
        D = pairwise_distances(X_arr, metric="euclidean")
        vals = D[D > 0]
        med = np.median(vals) if vals.size else 1.0
        if not np.isfinite(med) or med <= 0:
            return 1.0
        return 1.0 / (2.0 * (med ** 2))

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X)

    gamma = _cfg_float(config, "dm_gamma", -1.0)
    if gamma is None or gamma <= 0:
        gamma = _median_gamma(Xts)
    t = _cfg_float(config, "dm_t", 1.0)

    D2 = pairwise_distances(Xts, metric="sqeuclidean")
    K = np.exp(-gamma * D2)
    np.fill_diagonal(K, 0.0)
    d = K.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    P = K / d

    w, V = np.linalg.eig(P)
    w = np.real(w)
    V = np.real(V)
    order = np.argsort(-np.abs(w))
    w = w[order]
    V = V[:, order]
    start_idx = 1 if V.shape[1] > 1 else 0
    k = min(n_components, max(0, V.shape[1] - start_idx))
    eigvals_sel = w[start_idx:start_idx + k]
    eigvecs_sel = V[:, start_idx:start_idx + k]
    scale = np.power(np.abs(eigvals_sel), t) if k > 0 else np.array([])

    def _embed_new(Xnew: np.ndarray) -> np.ndarray:
        if k == 0 or Xnew is None:
            return np.zeros((0, 0)) if Xnew is None else np.zeros((Xnew.shape[0], 0))
        Xnew = scaler.transform(Xnew)
        D2n = pairwise_distances(Xnew, Xts, metric="sqeuclidean")
        Kny = np.exp(-gamma * D2n)
        row = Kny.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        Pny = Kny / row
        Phi = (Pny @ eigvecs_sel) / np.maximum(np.abs(eigvals_sel), 1e-12)
        Phi *= scale
        return Phi

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return _embed_new(data)

    return projector

def _run_le(
    X, n_components, *, config=None, **kwargs
) -> Projector:
    def _median_gamma(X_arr: np.ndarray) -> float:
        D = pairwise_distances(X_arr, metric="euclidean")
        vals = D[D > 0]
        med = np.median(vals) if vals.size else 1.0
        if not np.isfinite(med) or med <= 0:
            return 1.0
        return 1.0 / (2.0 * (med ** 2))

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X)

    k_nb = _cfg_int(config, "le_neighbors", 10)
    gamma = _cfg_float(config, "le_gamma", -1.0)
    if gamma is None or gamma <= 0:
        gamma = _median_gamma(Xts)

    n = Xts.shape[0]
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
            W[j, i] = max(W[j, i], w)
    D = np.diag(W.sum(axis=1))
    D_sqrt_inv = np.diag(1.0 / np.maximum(np.sqrt(np.diag(D)), 1e-12))
    Lsym = D_sqrt_inv @ W @ D_sqrt_inv

    evals, evecs = eigh(Lsym)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    start_idx = 1 if evecs.shape[1] > 1 else 0
    k = min(n_components, max(0, evecs.shape[1] - start_idx))
    U = evecs[:, start_idx:start_idx + k]
    d_i_sqrt = np.sqrt(np.maximum(np.diag(D), 1e-12))

    def _embed_new(Xnew: np.ndarray) -> np.ndarray:
        if k == 0 or Xnew is None:
            return np.zeros((0, 0)) if Xnew is None else np.zeros((Xnew.shape[0], 0))
        Xnew_std = scaler.transform(Xnew)
        nn_new = NearestNeighbors(n_neighbors=min(max(1, k_nb), n)).fit(Xts)
        neigh_ind = nn_new.kneighbors(Xnew_std, return_distance=False)
        Z = np.zeros((Xnew_std.shape[0], k), dtype=float)
        for r, nbrs in enumerate(neigh_ind):
            wrow = np.zeros(n, dtype=float)
            for j in nbrs:
                diff = Xnew_std[r] - Xts[j]
                wrow[j] = np.exp(-gamma * float(np.dot(diff, diff)))
            d_y = wrow.sum()
            if d_y <= 0:
                continue
            coef = (wrow / np.maximum(d_y, 1e-12)) / np.maximum(d_i_sqrt, 1e-12)
            Z[r] = (coef @ U) / np.sqrt(d_y)
        return Z

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        return _embed_new(data)

    return projector

def _run_autoencoder(
    X, n_components, *, config=None, **kwargs
) -> Projector:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except Exception as e:
        raise RuntimeError("AutoEncoder を利用するには 'torch' が必要です") from e

    ae_seed = 0
    try:
        base_seed = int(_cfg_int(config, "seed", 0))
        fseed = int(getattr(config, "f_seed", 0))
        ae_seed = int(base_seed + fseed)
    except Exception:
        ae_seed = 0
    try:
        torch.manual_seed(ae_seed)
        try:
            torch.cuda.manual_seed_all(ae_seed)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass
    try:
        np.random.seed(ae_seed)
    except Exception:
        pass
    try:
        random.seed(ae_seed)
    except Exception:
        pass

    scaler = StandardScaler()
    Xts = scaler.fit_transform(X).astype(np.float32)

    epochs = _cfg_int(config, "ae_epochs", 20)
    batch = _cfg_int(config, "ae_batch", 256)

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
            return xhat, z

    input_dim = Xts.shape[1]
    model = _AE(input_dim, n_components)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(Xts))
    val_ratio = 0.1
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    dl_tr = DataLoader(train_set, batch_size=batch, shuffle=True)
    dl_va = DataLoader(val_set, batch_size=batch, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    best_state, best_val, patience, bad = None, float("inf"), 3, 0
    for _ in range(epochs):
        model.train()
        for (xb,) in dl_tr:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for (xb,) in dl_va:
                xb = xb.to(device)
                recon, _ = model(xb)
                va += crit(recon, xb).item() * xb.size(0)
        va /= max(1, len(val_set))
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
    model.eval()

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        arr = scaler.transform(data).astype(np.float32)
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(arr))
        return z.numpy()

    return projector


def _run_autoencoder_ae2(
    X, n_components, *, config=None, **kwargs
) -> Projector:
    """
    AE2 (paper table style):
      input -> 500(ReLU) -> 200(ReLU) -> bottleneck(ReLU)
      -> 200(ReLU) -> 500(ReLU) -> output(Sigmoid)
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except Exception as e:
        raise RuntimeError("AE2 を利用するには 'torch' が必要です") from e

    ae_seed = 0
    try:
        base_seed = int(_cfg_int(config, "seed", 0))
        fseed = int(getattr(config, "f_seed", 0))
        ae_seed = int(base_seed + fseed)
    except Exception:
        ae_seed = 0
    try:
        torch.manual_seed(ae_seed)
        try:
            torch.cuda.manual_seed_all(ae_seed)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass
    try:
        np.random.seed(ae_seed)
    except Exception:
        pass
    try:
        random.seed(ae_seed)
    except Exception:
        pass

    # Sigmoid output is naturally matched with [0, 1] inputs.
    use_minmax = bool(_cfg_get(config, "ae2_minmax", True))
    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    Xts = scaler.fit_transform(X).astype(np.float32)

    epochs = _cfg_int(config, "ae2_epochs", _cfg_int(config, "ae_epochs", 100))
    batch = _cfg_int(config, "ae2_batch", _cfg_int(config, "ae_batch", 256))
    lr = _cfg_float(config, "ae2_lr", 1e-3)

    class _AE2(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 500), nn.ReLU(),
                nn.Linear(500, 200), nn.ReLU(),
                nn.Linear(200, latent_dim), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 200), nn.ReLU(),
                nn.Linear(200, 500), nn.ReLU(),
                nn.Linear(500, input_dim), nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            xhat = self.decoder(z)
            return xhat, z

    input_dim = Xts.shape[1]
    model = _AE2(input_dim, n_components)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(torch.from_numpy(Xts))
    val_ratio = 0.1
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    dl_tr = DataLoader(train_set, batch_size=batch, shuffle=True)
    dl_va = DataLoader(val_set, batch_size=batch, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_state, best_val, patience, bad = None, float("inf"), 5, 0
    for _ in range(max(1, epochs)):
        model.train()
        for (xb,) in dl_tr:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for (xb,) in dl_va:
                xb = xb.to(device)
                recon, _ = model(xb)
                va += crit(recon, xb).item() * xb.size(0)
        va /= max(1, len(val_set))
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
    model.eval()

    def projector(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if data is None:
            return None
        arr = scaler.transform(data).astype(np.float32)
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(arr))
        return z.numpy()

    return projector

# ============================================================
# 公開API
# ============================================================

_RUNNERS: Dict[str, Any] = {
    "svd": _run_svd,
    "diffspan": _run_diffspan,
    "diffspan_orth": _run_diffspan_orth,
    "samespan_orth": _run_samespan_orth,
    "samespan": _run_samespan,
    "random_projection": _run_random_projection,
    "lpp": _run_lpp,
    "kcca": _run_kcca,
    "kernel_pca": lambda *a, **kw: _run_kpca_family(*a, mode="auto", **kw),
    "kernel_pca_self_tuning": lambda *a, **kw: _run_kpca_family(*a, mode="kernel_pca_self_tuning", **kw),
    "kernel_pca_gamma_fixed": lambda *a, **kw: _run_kpca_family(*a, mode="kernel_pca_gamma_fixed", **kw),
    "umap": _run_umap,
    "umap_2": _run_umap_2,
    "supervised_umap": _run_supervised_umap,
    "dm": _run_dm,
    "le": _run_le,
    "autoencoder": _run_autoencoder,
    "ae": _run_autoencoder,
    "ae2": _run_autoencoder_ae2,
}


def build_dimensionality_projector(
    X: np.ndarray,
    n_components: int,
    *,
    y: Optional[np.ndarray] = None,
    F_type: str = "kernel_pca",
    seed: Optional[int] = None,
    param: Any = None,
    config: Any = None,
) -> Projector:
    if F_type not in _RUNNERS:
        raise ValueError(f"未知の F_type: {F_type}")
    runner = _RUNNERS[F_type]
    return runner(
        X,
        n_components,
        y=y,
        seed=seed,
        param=param,
        config=config,
    )
