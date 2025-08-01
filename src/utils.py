import random
from typing import Optional, Tuple, TypeVar

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class LPPScratch:
    def __init__(self, n_components=2, t=0.01, k_neighbors=5):
        self.n_components = n_components
        self.t = t
        self.k_neighbors = k_neighbors
        self.A = None  # 射影行列

    def _construct_weight_matrix(self, X):
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

    def fit(self, X):
        X = X.astype(np.float64)
        W = self._construct_weight_matrix(X)
        D = np.diag(W.sum(axis=1))
        L = D - W

        # 一般化固有値問題を解く: X^T L X a = λ X^T D X a
        XT_D_X = X.T @ D @ X
        XT_L_X = X.T @ L @ X

        # 正則化を追加して XT_D_X が正定値になるようにする
        # これにより "not positive definite" エラーを回避する
        reg = 1e-9 * np.eye(XT_D_X.shape[0])
        
        # 対称な一般化固有値問題を解く
        try:
            eigvals, eigvecs = eigh(XT_L_X, XT_D_X + reg)
        except np.linalg.LinAlgError as e:
            print(f"LPPで固有値計算エラーが発生しました: {e}")
            # エラー発生時は、PCAにフォールバックするなどの代替処理も検討可能
            # ここでは、エラーを伝播させる代わりに、射影行列を単位行列として処理を続行させる
            self.A = np.eye(X.shape[1])[:, :self.n_components]
            return


        # 小さい固有値に対応するベクトルを選択
        sorted_indices = np.argsort(eigvals)
        self.A = eigvecs[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        return X @ self.A

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

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
        n_samples, n_features = X.shape

        if self.center:
            self.mean_ = X.mean(axis=0)
            X = X - self.mean_
            
        U, S, Vt = np.linalg.svd(X, full_matrices=self.full_matrices)
        
        k = self.n_components
        if k is None:
            k = len(S)

        actual_k = len(S) # SVDで得られた実際の成分数

        # 指定された成分数が実際の成分数より大きい場合、ゼロでパディングする
        if k > actual_k:
            # 特異値ベクトルのパディング
            padded_S = np.zeros(k)
            padded_S[:actual_k] = S
            self.singular_values_ = padded_S
            
            # 射影行列(Vt)のパディング
            padded_Vt = np.zeros((k, n_features))
            padded_Vt[:actual_k, :] = Vt[:actual_k, :]
            self.components_ = padded_Vt
        else:
            # 通常の処理
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

# ゼロパディングなしのため、サンプル数 > 特徴量で性能悪化
class SVDScratch_:
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

def self_tuning_gamma(
        X, *,
        k: int = 7,
        standardize: bool = True,
        summary: str = "median"):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        入力データ行列。欠損値は事前に処理しておくこと。
    k : int, default 7
        σᵢ を決める最近傍の順位 (k-th NN)。
    standardize : bool, default True
        True の場合、内部で Z スコア標準化してから距離を計算。
    summary : {'median', 'mean', None}, default 'median'
        - 'median' → 全 γᵢ の中央値を返す  
        - 'mean'   → 平均値を返す  
        - None     → 集約せず (γᵢ, σᵢ) ベクトルをそのまま返す

    Returns
    -------
    gamma : float
        summary が 'median' or 'mean' のとき: 代表 γ
    gamma_i : ndarray
        summary == None のとき: 各サンプルの γᵢ
    sigma_i : ndarray
        summary == None のとき: 各サンプルの σᵢ
    """

    X = np.asarray(X, dtype=float)
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    # k+1 近傍 (0番目は自分自身 → 距離0)
    nbrs = NearestNeighbors(n_neighbors=k + 1,
                            algorithm="auto",
                            metric="euclidean").fit(X)
    dists, _ = nbrs.kneighbors(X, return_distance=True)
    sigma_i = dists[:, k]
    sigma_i[sigma_i == 0] = np.finfo(float).eps  # 発散防止
    gamma_i = 1.0 / sigma_i
    gamma_i = gamma_i / 3 # 恣意的調整

    if summary is None:
        return gamma_i, sigma_i

    if summary == "median":
        return float(np.median(gamma_i))
    elif summary == "mean":
        return float(np.mean(gamma_i))
    else:
        raise ValueError("summary must be 'median', 'mean', or None")


def reduce_dimensions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    anchor: Optional[np.ndarray] = None,
    F_type = "kernel_pca",
    seed= None,
    param = None,
    config =  None
) -> Tuple[np.ndarray, ...]:
    # --- SVD / KernelPCA の選択 ---
    # USE_KERNEL = n_components >= X_train.shape[1]
    
    if F_type == "svd":
        model = SVDScratch(n_components=n_components, center=True)
        # --- フィッティングと変換 ---
        X_train_svd = model.fit_transform(X_train)
        X_test_svd = model.transform(X_test)

        if anchor is not None:
            X_anchor_svd = model.transform(anchor)
            return X_train_svd, X_test_svd, X_anchor_svd

        return X_train_svd, X_test_svd
    
    elif F_type == "diffspan":
        # 1. SVDScratchで次元削減 (F' に相当する処理)
        svd = SVDScratch(n_components=n_components, center=True)
        X_train_F_prime = svd.fit_transform(X_train)
        X_test_F_prime = svd.transform(X_test)

        # 2. 直交性を崩すためのランダム行列 E を作成
        # 再現性のためにseedを使用
        rng = np.random.default_rng(config.seed)
                # 各値が1から2までの一様分布に従うランダム行列を生成
        E = rng.uniform(size=(n_components, n_components))
        # 3. F = F'E を計算
        X_train_F = X_train_F_prime @ E
        X_test_F = X_test_F_prime @ E

        if anchor is not None:
            X_anchor_F_prime = svd.transform(anchor)
            X_anchor_F = X_anchor_F_prime @ E
            return X_train_F, X_test_F, X_anchor_F

        return X_train_F, X_test_F
    
    elif F_type == "samespan_orth":
        # 入力データの次元
        m = X_train.shape[1]
        l = n_components

        # l > m の場合はエラーを発生させる
        if l > m:
            raise ValueError("列直交行列を作るには l <= m が必要です。")

        # 1) F' を生成 (列直交行列)
        rng = np.random.default_rng(seed=config.seed)
        A = rng.standard_normal(size=(m, l))  # 乱数行列
        Q, R = np.linalg.qr(A, mode="reduced")  # QR 分解
        signs = np.sign(np.diag(R))
        Q *= signs  # 列の符号を統一
        F_prime = Q  # F'

        # 2) ランダムな直交行列 E を生成 (seed 指定なし)
        random_matrix = np.random.standard_normal(size=(l, l))
        Q_E, _ = np.linalg.qr(random_matrix)  # QR 分解で直交行列を生成
        E = Q_E  # 直交行列 E

        # 3) F = F' * E を計算
        F = F_prime @ E

        # 4) 次元削減を適用
        X_train_reduced = X_train @ F
        X_test_reduced = X_test @ F

        if anchor is not None:
            X_anchor_reduced = anchor @ F
            return X_train_reduced, X_test_reduced, X_anchor_reduced

        return X_train_reduced, X_test_reduced

    elif F_type == "samespan":
        # 入力データの次元
        m = X_train.shape[1]
        l = n_components

        # l > m の場合はエラーを発生させる
        if l > m:
            raise ValueError("列直交行列を作るには l <= m が必要です。")

        # 1) F' を生成 (列直交行列)
        rng = np.random.default_rng(seed=config.seed)
        A = rng.standard_normal(size=(m, l))  # 乱数行列
        Q, R = np.linalg.qr(A, mode="reduced")  # QR 分解
        signs = np.sign(np.diag(R))
        Q *= signs  # 列の符号を統一
        F_prime = Q  # F'

        # 2) ランダムな直交行列 E を生成 (seed 指定なし)
        random_matrix = np.random.standard_normal(size=(l, l))
        E = random_matrix  # 行列 E

        # 3) F = F' * E を計算
        F = F_prime @ E

        # 4) 次元削減を適用
        X_train_reduced = X_train @ F
        X_test_reduced = X_test @ F

        if anchor is not None:
            X_anchor_reduced = anchor @ F
            return X_train_reduced, X_test_reduced, X_anchor_reduced

        return X_train_reduced, X_test_reduced
    
    elif F_type == "lpp":
        k = int(config.num_institution_user * 0.2)
        model = LPPScratch(n_components=n_components, t=0.01, k_neighbors=k)

        X_train_lpp = model.fit_transform(X_train)
        X_test_lpp = model.transform(X_test)

        if anchor is not None:
            X_anchor_lpp = model.transform(anchor)
            return X_train_lpp, X_test_lpp, X_anchor_lpp

        return X_train_lpp, X_test_lpp


    else:
    # --- スケーリング ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        anchor_scaled = scaler.transform(anchor) if anchor is not None else None
        gamma = 1.0 / X_train.shape[1] # har だと 0.001 が精度良い
        gamma = self_tuning_gamma(X_train_scaled, standardize=False, k=7, summary='median')
        if config is not None:
            # config.gammas に追加
            if not hasattr(config, 'nl_gammas') or config.nl_gammas is None:
                config.nl_gammas = []  # gammas が存在しない場合、新しいリストを作成
            config.nl_gammas.append(gamma)  # gamma をリストに追加
        model = KernelPCA(
             n_components=n_components,
             kernel="rbf",
             gamma=gamma,
             eigen_solver="auto",
             n_jobs=-1,
        )
        #model = make_random_kpca(n_components, seed=seed, param=param)
        # --- フィッティングと変換 ---
        X_train_svd = model.fit_transform(X_train_scaled)
        X_test_svd = model.transform(X_test_scaled)
        
        if anchor_scaled is not None:
            X_anchor_svd = model.transform(anchor_scaled)
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

def make_random_kpca(n_components, seed=None, param=None):
    rng = np.random.default_rng(seed)

    # カーネルをランダムに選ぶ
    #kernel = rng.choice(["linear", "poly", "rbf"])
    #print(kernel)
    kernel = "rbf"  # 強制的に RBF カーネルを使用する場合
    # パラメータ辞書
    params = {
        "n_components": n_components,
        "kernel": kernel,
        "eigen_solver": "auto",
        "n_jobs": -1,
    }
    # カーネルごとのパラメータ設定
    if kernel in ["rbf", "poly", "sigmoid"]:
        if seed % 3 == 0:
            params["gamma"] = 0.1
        elif seed % 3 == 1:
            params["gamma"] = 1
        else:
            params["gamma"] = 5
        #params["gamma"] = 0.1
        #print("Random KPCA parameters (gamma):", params["gamma"])
    if kernel == "poly":
        params["degree"] = 2 #rng.integers(2, 6)
        params["coef0"] = 0.0 #rng.uniform(0, 1.0)
    if kernel == "sigmoid":
        params["coef0"] = rng.uniform(0, 1.0)
    #print("Random KPCA parameters:--------------------------------")
    #print(seed)
    #print(params)
    # KernelPCA インスタンスを返す
    return KernelPCA(**params)