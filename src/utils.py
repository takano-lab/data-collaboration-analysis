from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
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


class SVDScratch_:
    """
    ゼロパディングなしの SVD（参考用途）
    """
    def __init__(self, n_components: Optional[int] = None, *, center: bool = False, full_matrices: bool = False):
        self.n_components = n_components
        self.center = center
        self.full_matrices = full_matrices
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SVDScratch_":
        X = np.asarray(X, dtype=float)
        if self.center:
            self.mean_ = X.mean(axis=0)
            X = X - self.mean_
        U, S, Vt = np.linalg.svd(X, full_matrices=self.full_matrices)
        k = self.n_components or len(S)
        self.singular_values_ = S[:k]
        self.components_ = Vt[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("SVDScratch_: fit を先に呼んでください")
        X = np.asarray(X, dtype=float)
        if self.center and self.mean_ is not None:
            X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


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

# 追加: 文字列用
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
    mode in {"fixed", "self_tuning", "unfixed"}
    """
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train)
    Xvs = scaler.transform(X_test)
    Xas = scaler.transform(anchor) if anchor is not None else None
    Xats = scaler.transform(anchor_test) if anchor_test is not None else None

    if mode == "fixed":
        gamma = 1.0 / X_train.shape[1]
    elif mode == "self_tuning":
        gamma = self_tuning_gamma(Xts, standardize=False, k=7, summary='median')
        ratio = float(getattr(config, "gamma_ratio", 1.0)) if config is not None else 1.0
        gamma *= ratio
        if config is not None:
            if not hasattr(config, "nl_gammas") or config.nl_gammas is None:
                config.nl_gammas = []
            config.nl_gammas.append(gamma)
    elif mode == "unfixed":
        gamma = self_tuning_gamma(Xts, standardize=False, k=7, summary='median')
        sd = 0 if seed is None else int(seed)
        if sd % 6 != 0:
            gamma = (10 ** ((sd % 6) - 3)) * gamma
        else:
            gamma = 1.0 / X_train.shape[1]
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

    model = UMAP(
        n_components=n_components,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
    )
    Xt = model.fit_transform(Xts)
    Xv = model.transform(Xvs)
    Xa = model.transform(Xas) if Xas is not None else None
    Xat = model.transform(Xats) if Xats is not None else None
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
    "kernel_pca": lambda *a, **kw: _run_kpca_family(*a, mode="fixed", **kw),
    "kernel_pca_self_tuning": lambda *a, **kw: _run_kpca_family(*a, mode="self_tuning", **kw),
    "kernel_pca_unfixed_gamma": lambda *a, **kw: _run_kpca_family(*a, mode="unfixed", **kw),
    # 追加
    "umap": _run_umap,
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


# 参考: 旧API（必要なら使用）。戻り値は4要素タプルに統一
def reduce_dimensions_with_svd_(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    anchor: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    svd = SVDScratch(n_components=n_components, center=True)
    svd.fit(X_train)
    Xt = svd.transform(X_train)
    Xv = svd.transform(X_test)
    Xa = svd.transform(anchor) if anchor is not None else None
    return _to_tuple4(Xt, Xv, Xa, None)


def make_random_kpca(n_components: int, seed: Optional[int] = None, param: Any = None) -> KernelPCA:
    rng = np.random.default_rng(seed)
    kernel = "rbf"
    params: Dict[str, Any] = {
        "n_components": n_components,
        "kernel": kernel,
        "eigen_solver": "auto",
        "n_jobs": -1,
    }
    # ランダム γ の例
    if seed is not None:
        if seed % 3 == 0:
            params["gamma"] = 0.1
        elif seed % 3 == 1:
            params["gamma"] = 1
        else:
            params["gamma"] = 5
    else:
        params["gamma"] = 1.0
    return KernelPCA(**params)