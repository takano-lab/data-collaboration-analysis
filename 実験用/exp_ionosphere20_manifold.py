from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass
class ExpCfg:
    seed: int = 0
    n_samples: int = 20
    n_components: int = 3
    out_dir: Path = Path("output") / "ionosphere20_manifold"

    # small grids
    isomap_neighbors: Tuple[int, ...] = (6, 8, 10, 12)
    le_neighbors: Tuple[int, ...] = (6, 10, 14)
    umap_neighbors: Tuple[int, ...] = (5, 10, 15)
    umap_min_dist: Tuple[float, ...] = (0.0, 0.1)
    lin_svm_C: Tuple[float, ...] = (0.1, 1.0, 10.0)


def _label_encode(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind in {"i", "u"}:
        return y.astype(int)
    le = LabelEncoder()
    return le.fit_transform(y)


def load_ionosphere() -> Tuple[np.ndarray, np.ndarray]:
    """Load Ionosphere from OpenML robustly; keep only numeric features."""
    d = fetch_openml("ionosphere", as_frame=True)
    Xdf = d["data"]
    Xdf_num = Xdf.select_dtypes(include=[np.number])
    if Xdf_num.shape[1] == 0:
        raise RuntimeError("ionosphere dataset has no numeric columns after filtering")
    X = Xdf_num.to_numpy(dtype=np.float32)
    y = _label_encode(np.asarray(d["target"]))  # binary
    return X, y


def sample_balanced_20(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    k = len(classes)
    if n < k:
        chosen_classes = rng.choice(classes, size=n, replace=False)
        idx = np.array([rng.choice(np.where(y == c)[0]) for c in chosen_classes], dtype=int)
        return X[idx], y[idx], idx
    # ensure at least 1 per class
    pools = {int(c): np.where(y == c)[0] for c in classes}
    chosen: List[int] = []
    for c in classes:
        chosen.append(int(rng.choice(pools[int(c)])))
    # remove chosen
    chosen_set = set(chosen)
    for c in classes:
        pools[int(c)] = np.array([i for i in pools[int(c)] if i not in chosen_set], dtype=int)
    remain = n - k
    order = list(map(int, classes))
    ptr = 0
    while remain > 0 and any(len(pools[int(c)]) > 0 for c in order):
        c = order[ptr % k]
        if len(pools[c]) > 0:
            i = int(pools[c][0])
            pools[c] = pools[c][1:]
            chosen.append(i)
            remain -= 1
        ptr += 1
    idx = np.array(chosen[:n], dtype=int)
    return X[idx], y[idx], idx


def embed_pca(X: np.ndarray, k: int) -> np.ndarray:
    return PCA(n_components=k, random_state=0).fit_transform(X)


def embed_isomap(X: np.ndarray, k: int, n_neighbors: int) -> np.ndarray:
    return Isomap(n_neighbors=n_neighbors, n_components=k).fit_transform(X)


def embed_le(X: np.ndarray, k: int, n_neighbors: int) -> np.ndarray:
    se = SpectralEmbedding(n_components=k, n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=0)
    return se.fit_transform(X)


def embed_umap(X: np.ndarray, k: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        from umap import UMAP  # type: ignore
    except Exception as e:
        raise RuntimeError("umap-learn not installed") from e
    um = UMAP(n_components=k, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=0)
    return um.fit_transform(X)


def _median_heuristic_gamma(X: np.ndarray) -> float:
    D = pairwise_distances(X, metric="euclidean")
    med = np.median(D[D > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / (2.0 * (med ** 2))


def embed_diffusion_maps(X: np.ndarray, k: int, *, gamma: Optional[float] = None) -> np.ndarray:
    if gamma is None:
        gamma = _median_heuristic_gamma(X)
    D2 = pairwise_distances(X, metric="sqeuclidean")
    K = np.exp(-gamma * D2)
    row_sum = K.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = K / row_sum
    w, V = np.linalg.eig(P.T)
    idx = np.argsort(-np.abs(w))
    V = np.real(V[:, idx])
    comps = V[:, 1:k+1]
    return comps


def eval_linear_svc_cv(Z: np.ndarray, y: np.ndarray, C_list: Tuple[float, ...], seed: int) -> Tuple[float, Dict[str, float]]:
    best_acc = -np.inf
    best_params: Dict[str, float] = {}
    n = len(y)
    n_splits = 5 if n >= 5 else max(2, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for C in C_list:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="linear", C=float(C), random_state=seed))
        ])
        accs: List[float] = []
        for tr, te in kf.split(Z):
            pipe.fit(Z[tr], y[tr])
            pred = pipe.predict(Z[te])
            accs.append(accuracy_score(y[te], pred))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = {"C": float(C)}
    return best_acc, best_params


def plot_3d(Z: np.ndarray, y: np.ndarray, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax: Axes = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="tab10", s=40, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Ionosphere 20-sample manifold experiment (3D + Linear SVC)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ExpCfg().out_dir))
    args = ap.parse_args()

    cfg = ExpCfg(seed=int(args.seed), out_dir=Path(args.out))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_ionosphere()
    X20, y20, idx = sample_balanced_20(X, y, n=cfg.n_samples, seed=cfg.seed)
    scaler = StandardScaler()
    X20s = scaler.fit_transform(X20)

    results: List[Dict[str, object]] = []

    # PCA
    try:
        Z = embed_pca(X20s, cfg.n_components)
        acc, p = eval_linear_svc_cv(Z, y20, cfg.lin_svm_C, cfg.seed)
        plot_3d(Z, y20, f"PCA (acc={acc:.3f})", cfg.out_dir / "ionosphere20_pca_3d.png")
        results.append({"method": "PCA", "params": "{}", "acc": acc})
    except Exception as e:
        results.append({"method": "PCA", "params": "{}", "acc": np.nan, "error": str(e)})

    # Isomap
    for nn in cfg.isomap_neighbors:
        try:
            Z = embed_isomap(X20s, cfg.n_components, n_neighbors=nn)
            acc, p = eval_linear_svc_cv(Z, y20, cfg.lin_svm_C, cfg.seed)
            plot_3d(Z, y20, f"Isomap nn={nn} (acc={acc:.3f})", cfg.out_dir / f"ionosphere20_isomap_nn{nn}_3d.png")
            results.append({"method": "Isomap", "params": f"nn={nn}", "acc": acc, **p})
        except Exception as e:
            results.append({"method": "Isomap", "params": f"nn={nn}", "acc": np.nan, "error": str(e)})

    # Laplacian Eigenmaps
    for nn in cfg.le_neighbors:
        try:
            Z = embed_le(X20s, cfg.n_components, n_neighbors=nn)
            acc, p = eval_linear_svc_cv(Z, y20, cfg.lin_svm_C, cfg.seed)
            plot_3d(Z, y20, f"LaplacianEigenmaps nn={nn} (acc={acc:.3f})", cfg.out_dir / f"ionosphere20_le_nn{nn}_3d.png")
            results.append({"method": "LaplacianEigenmaps", "params": f"nn={nn}", "acc": acc, **p})
        except Exception as e:
            results.append({"method": "LaplacianEigenmaps", "params": f"nn={nn}", "acc": np.nan, "error": str(e)})

    # UMAP
    for nn in cfg.umap_neighbors:
        for md in cfg.umap_min_dist:
            try:
                Z = embed_umap(X20s, cfg.n_components, n_neighbors=nn, min_dist=md)
                acc, p = eval_linear_svc_cv(Z, y20, cfg.lin_svm_C, cfg.seed)
                plot_3d(Z, y20, f"UMAP nn={nn}, md={md} (acc={acc:.3f})", cfg.out_dir / f"ionosphere20_umap_nn{nn}_md{md}_3d.png")
                results.append({"method": "UMAP", "params": f"nn={nn},min_dist={md}", "acc": acc, **p})
            except Exception as e:
                results.append({"method": "UMAP", "params": f"nn={nn},min_dist={md}", "acc": np.nan, "error": str(e)})

    # Diffusion Maps (gamma via median heuristic and 2x)
    try:
        g_med = _median_heuristic_gamma(X20s)
    except Exception:
        g_med = 1.0
    for g in (g_med, 2 * g_med):
        try:
            Z = embed_diffusion_maps(X20s, cfg.n_components, gamma=g)
            acc, p = eval_linear_svc_cv(Z, y20, cfg.lin_svm_C, cfg.seed)
            plot_3d(Z, y20, f"DiffusionMaps g={g:.2e} (acc={acc:.3f})", cfg.out_dir / f"ionosphere20_diffmap_g{g:.2e}_3d.png")
            results.append({"method": "DiffusionMaps", "params": f"gamma={g:.3e}", "acc": acc, **p})
        except Exception as e:
            results.append({"method": "DiffusionMaps", "params": f"gamma={g:.3e}", "acc": np.nan, "error": str(e)})

    # Save
    res_df = pd.DataFrame(results)
    res_path = cfg.out_dir / "ionosphere20_results.csv"
    res_df.to_csv(res_path, index=False)

    info = {
        "seed": cfg.seed,
        "n_samples": cfg.n_samples,
        "n_components": cfg.n_components,
        "picked_indices": list(map(int, idx)),
        "class_counts": {int(c): int((y20 == c).sum()) for c in np.unique(y20)}
    }
    pd.DataFrame([info]).to_json(cfg.out_dir / "ionosphere20_meta.json", orient="records", force_ascii=False, indent=2)

    if not res_df.empty and res_df["acc"].notna().any():
        best_row = res_df.loc[res_df["acc"].idxmax()]
        print(f"Best: method={best_row['method']} params={best_row['params']} acc={best_row['acc']:.3f}")
    else:
        print("No valid results (all failed)")


if __name__ == "__main__":
    main()
