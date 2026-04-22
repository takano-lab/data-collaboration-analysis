import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# Utility
# =========================
def set_seed(seed=42):
    np.random.seed(seed)


def make_random_F(d_in, d_out, rng):
    # 完全ランダムでよい、という設定
    return rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)


def make_F_with_shared_column_space(d_in, d_out, shared_basis, rng):
    # 全クライアントで同じ列空間を持つ F を生成する
    shared_dim = shared_basis.shape[1]
    if shared_basis.shape[0] != d_in:
        raise ValueError("shared_basis の行数が d_in と一致しません。")

    if d_out <= shared_dim:
        # square(または縦長)部分で回転させ、同じ列空間を保つ
        R, _ = np.linalg.qr(rng.standard_normal((shared_dim, shared_dim)))
        return shared_basis @ R[:, :d_out]

    # d_out > shared_dim の場合は、追加列を同一部分空間内の線形結合で作る
    extra = rng.standard_normal((shared_dim, d_out - shared_dim)) / np.sqrt(shared_dim)
    return np.concatenate([shared_basis, shared_basis @ extra], axis=1)


def random_orthonormal_matrix(n_rows, n_cols, rng):
    # n_rows x n_cols の直交列ベクトル行列を作る
    if n_cols > n_rows:
        raise ValueError("n_cols は n_rows 以下である必要があります。")
    M = rng.standard_normal((n_rows, n_cols))
    Q, _ = np.linalg.qr(M)
    return Q[:, :n_cols]


def top_orthonormal_basis(A, rank_q):
    # A = U S V^T の左特異ベクトルを使う
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    q = min(rank_q, U.shape[1])
    return U[:, :q]


def ridge_map(Ak, Z, lam=1e-6):
    # Gk = argmin_G ||Ak G - Z||_F^2 + lam ||G||_F^2
    #    = (Ak^T Ak + lam I)^(-1) Ak^T Z
    m = Ak.shape[1]
    lhs = Ak.T @ Ak + lam * np.eye(m)
    rhs = Ak.T @ Z
    # lhs は対称正定値を想定。Cholesky は solve より高速で安定。
    try:
        L = np.linalg.cholesky(lhs)
        y = np.linalg.solve(L, rhs)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        return np.linalg.solve(lhs, rhs)


def pca_from_concat(mats, z_dim, random_state=42):
    M = np.concatenate(mats, axis=1)   # 横結合
    # 大規模行列では randomized SVD を使って計算時間を削減
    min_dim = min(M.shape[0], M.shape[1])
    use_randomized = (min_dim >= 300) and (z_dim < min_dim)
    pca = PCA(
        n_components=z_dim,
        svd_solver="randomized" if use_randomized else "full",
        random_state=random_state,
    )
    Z = pca.fit_transform(M)           # shape: (rows_of_A, z_dim)
    return Z, pca


def cosine_similarity_abs(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(a, b)) / (na * nb))


def subspace_alignment_score(B1, B2):
    # 2つの部分空間の整合度を [0,1] で返す（主角の cos の平均）
    Q1, _ = np.linalg.qr(B1)
    Q2, _ = np.linalg.qr(B2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s))


# =========================
# Data split
# =========================
def split_into_k_clients(X, y, K=10, n_train_per_client=50, n_test_per_client=50, seed=42):
    rng = np.random.default_rng(seed)

    # train/test は半々
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # クライアント用に必要数だけ取り出す
    total_train_needed = K * n_train_per_client
    total_test_needed = K * n_test_per_client

    if len(X_train_all) < total_train_needed or len(X_test_all) < total_test_needed:
        raise ValueError("サンプル数が足りません。")

    train_idx = rng.choice(len(X_train_all), size=total_train_needed, replace=False)
    test_idx = rng.choice(len(X_test_all), size=total_test_needed, replace=False)

    X_train_pool = X_train_all[train_idx]
    y_train_pool = y_train_all[train_idx]
    X_test_pool = X_test_all[test_idx]
    y_test_pool = y_test_all[test_idx]

    clients = []
    for k in range(K):
        tr_slice = slice(k * n_train_per_client, (k + 1) * n_train_per_client)
        te_slice = slice(k * n_test_per_client, (k + 1) * n_test_per_client)

        clients.append({
            "X_train": X_train_pool[tr_slice].copy(),
            "y_train": y_train_pool[tr_slice].copy(),
            "X_test": X_test_pool[te_slice].copy(),
            "y_test": y_test_pool[te_slice].copy(),
        })
    return clients


# =========================
# Anchor A
# =========================
def sample_anchor_A(X, a=1000, seed=42):
    rng = np.random.default_rng(seed)
    if len(X) < a:
        raise ValueError("アンカーAの行数 a がデータ数より大きいです。")
    idx = rng.choice(len(X), size=a, replace=False)
    return X[idx].copy()


# =========================
# Noise
# =========================
def add_noise(A_base, noise_type, noise_sigma, common_U=None, common_V=None, rng=None):
    if noise_type == "none":
        return A_base.copy(), None, None

    if noise_type == "independent":
        E = noise_sigma * rng.standard_normal(A_base.shape)
        return A_base + E, None, None

    if noise_type == "common":
        if common_U is None or common_V is None:
            raise ValueError("common noise には common_U, common_V が必要です。")
        # rank-r ノイズ: E = sigma * U V^T
        E = noise_sigma * (common_U @ common_V.T)
        return A_base + E, common_U, common_V

    raise ValueError(f"unknown noise_type: {noise_type}")


# =========================
# One experiment
# =========================
def run_one_experiment(
    X,
    y,
    K=10,
    a=1000,
    n_train_per_client=50,
    n_test_per_client=50,
    latent_dim=64,        # Fk の出力次元
    z_dim=20,             # 統合空間Zの次元
    common_noise_dim=None,  # commonノイズの次元（Noneなら z_dim）
    q_dim=20,             # Qk の列数
    noise_type="none",    # none / independent / common
    noise_sigma=0.1,
    classifier="rf",      # rf / logreg
    seed=42,
):
    rng = np.random.default_rng(seed)

    # クライアント分割
    clients = split_into_k_clients(
        X, y, K=K,
        n_train_per_client=n_train_per_client,
        n_test_per_client=n_test_per_client,
        seed=seed
    )

    # アンカー
    A = sample_anchor_A(X, a=a, seed=seed + 1)

    d = X.shape[1]

    # Fk の列空間は「偶数機関用」と「奇数機関用」の2種類を使う
    shared_dim = min(d, latent_dim)
    even_shared_basis = random_orthonormal_matrix(d, shared_dim, rng)
    odd_shared_basis = random_orthonormal_matrix(d, shared_dim, rng)

    if common_noise_dim is None:
        common_noise_dim = z_dim
    common_noise_dim = int(common_noise_dim)
    if common_noise_dim < 1:
        raise ValueError("common_noise_dim は 1 以上である必要があります。")
    if common_noise_dim > min(a, latent_dim):
        raise ValueError("common_noise_dim は min(a, latent_dim) 以下である必要があります。")

    # 共通ノイズ部分空間（common の時だけ使う）
    common_U = random_orthonormal_matrix(a, common_noise_dim, rng)
    common_V = random_orthonormal_matrix(latent_dim, common_noise_dim, rng)

    F_list = []
    A_list = []
    Q_list = []
    XFg_train_list = []
    y_train_list = []
    XFg_test_list = []
    y_test_list = []

    # 共通ノイズ方向と Z の整合を見るために保存
    noise_alignment_info = {}

    for k in range(K):
        # 機関番号は 1 始まりで奇偶を判定
        basis_k = even_shared_basis if ((k + 1) % 2 == 0) else odd_shared_basis
        Fk = make_F_with_shared_column_space(d, latent_dim, basis_k, rng)
        F_list.append(Fk)

        # 変換後
        A_base = A @ Fk
        Xk_tilde = clients[k]["X_train"] @ Fk
        Xk_test_tilde = clients[k]["X_test"] @ Fk

        # ノイズ
        Ak, cu, cv = add_noise(
            A_base, noise_type=noise_type, noise_sigma=noise_sigma,
            common_U=common_U, common_V=common_V, rng=rng
        )
        A_list.append(Ak)

        Qk = top_orthonormal_basis(Ak, q_dim)
        Q_list.append(Qk)

        # 後で Gk を作ってから埋める
        noise_alignment_info[k] = {
            "Xk_tilde": Xk_tilde,
            "Xk_test_tilde": Xk_test_tilde
        }

    # -------------------------
    # Method 1: Ak concat -> Z
    # -------------------------
    Z_A, _ = pca_from_concat(A_list, z_dim=z_dim, random_state=seed)

    X_train_A_method = []
    y_train_A_method = []
    X_test_A_method = []
    y_test_A_method = []

    for k in range(K):
        Ak = A_list[k]
        Gk = ridge_map(Ak, Z_A, lam=1e-6)

        Xk_proj = noise_alignment_info[k]["Xk_tilde"] @ Gk
        Xk_test_proj = noise_alignment_info[k]["Xk_test_tilde"] @ Gk

        X_train_A_method.append(Xk_proj)
        y_train_A_method.append(clients[k]["y_train"])
        X_test_A_method.append(Xk_test_proj)
        y_test_A_method.append(clients[k]["y_test"])

    X_train_A_method = np.vstack(X_train_A_method)
    y_train_A_method = np.concatenate(y_train_A_method)
    X_test_A_method = np.vstack(X_test_A_method)
    y_test_A_method = np.concatenate(y_test_A_method)

    # -------------------------
    # Method 2: Qk concat -> Z
    # -------------------------
    Z_Q, _ = pca_from_concat(Q_list, z_dim=z_dim, random_state=seed)

    X_train_Q_method = []
    y_train_Q_method = []
    X_test_Q_method = []
    y_test_Q_method = []

    for k in range(K):
        Ak = A_list[k]
        Gk = ridge_map(Ak, Z_Q, lam=1e-6)

        Xk_proj = noise_alignment_info[k]["Xk_tilde"] @ Gk
        Xk_test_proj = noise_alignment_info[k]["Xk_test_tilde"] @ Gk

        X_train_Q_method.append(Xk_proj)
        y_train_Q_method.append(clients[k]["y_train"])
        X_test_Q_method.append(Xk_test_proj)
        y_test_Q_method.append(clients[k]["y_test"])

    X_train_Q_method = np.vstack(X_train_Q_method)
    y_train_Q_method = np.concatenate(y_train_Q_method)
    X_test_Q_method = np.vstack(X_test_Q_method)
    y_test_Q_method = np.concatenate(y_test_Q_method)

    # スケーリング
    scaler_A = StandardScaler()
    X_train_A_method = scaler_A.fit_transform(X_train_A_method)
    X_test_A_method = scaler_A.transform(X_test_A_method)

    scaler_Q = StandardScaler()
    X_train_Q_method = scaler_Q.fit_transform(X_train_Q_method)
    X_test_Q_method = scaler_Q.transform(X_test_Q_method)

    # 分類器
    if classifier == "rf":
        clf_A = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1
        )
        clf_Q = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1
        )
    elif classifier == "logreg":
        clf_A = LogisticRegression(
            max_iter=2000,
            random_state=seed,
            multi_class="auto"
        )
        clf_Q = LogisticRegression(
            max_iter=2000,
            random_state=seed,
            multi_class="auto"
        )
    else:
        raise ValueError("classifier must be 'rf' or 'logreg'")

    clf_A.fit(X_train_A_method, y_train_A_method)
    clf_Q.fit(X_train_Q_method, y_train_Q_method)

    pred_A = clf_A.predict(X_test_A_method)
    pred_Q = clf_Q.predict(X_test_Q_method)

    result = {
        "noise_type": noise_type,
        "noise_sigma": noise_sigma,
        "common_noise_dim": common_noise_dim if noise_type == "common" else 0,
        "classifier": classifier,
        "Ak_method_acc": accuracy_score(y_test_A_method, pred_A),
        "Ak_method_macro_f1": f1_score(y_test_A_method, pred_A, average="macro"),
        "Qk_method_acc": accuracy_score(y_test_Q_method, pred_Q),
        "Qk_method_macro_f1": f1_score(y_test_Q_method, pred_Q, average="macro"),
    }

    # 共通ノイズ方向との整合度を追加
    if noise_type == "common":
        # Z の行側部分空間（a 次元側）と共通ノイズ部分空間 U の整合を比較
        zA_basis = top_orthonormal_basis(Z_A, common_noise_dim)
        zQ_basis = top_orthonormal_basis(Z_Q, common_noise_dim)

        result["Ak_method_noise_alignment"] = subspace_alignment_score(zA_basis, common_U)
        result["Qk_method_noise_alignment"] = subspace_alignment_score(zQ_basis, common_U)

    return result


# =========================
# Main
# =========================
def main():
    set_seed(42)

    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)

    configs = [
        # {"noise_type": "none", "noise_sigma": 0.0},
        # {"noise_type": "independent", "noise_sigma": 0.10},
        {"noise_type": "common", "noise_sigma": 10},
        # {"noise_type": "common", "noise_sigma": 0.20},
        # {"noise_type": "common", "noise_sigma": 0.50},
    ]

    rows = []
    for cfg in configs:
        print(f"Running: {cfg}")
        out = run_one_experiment(
            X=X,
            y=y,
            K=50,
            a=1000,
            n_train_per_client=50,
            n_test_per_client=50,
            latent_dim=64,
            z_dim=5,
            q_dim=20,
            noise_type=cfg["noise_type"],
            noise_sigma=cfg["noise_sigma"],
            classifier="rf",
            seed=42,
        )
        rows.append(out)

    df = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
