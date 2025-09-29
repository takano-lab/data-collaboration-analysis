from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml, load_iris
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ========= Datasets =========

# verbose flag (set in main)
VERBOSE: bool = False


def vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **{**kwargs, "flush": True})


def _label_encode(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind in {"i", "u"}:  # already ints
        return y.astype(int)
    le = LabelEncoder()
    return le.fit_transform(y)


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    name_low = name.lower()
    if name_low == "iris":
        X, y = load_iris(return_X_y=True, as_frame=False)
        return X.astype(np.float32), _label_encode(y)
    if name_low in ("balance", "balance_scale", "balance-scale"):
        d = fetch_openml("balance-scale", version=1, as_frame=False)
        X = d["data"].astype(np.float32)
        y = _label_encode(d["target"])  # L/B/R
        return X, y
    if name_low == "vowel":
        # Prefer a newer version without string columns; fallback robust handling
        try:
            d = fetch_openml("vowel", version=2, as_frame=True)
        except Exception:
            d = fetch_openml("vowel", as_frame=True)
        dfX = d["data"]
        # Keep only numeric columns to avoid values like 'Train'
        dfX_num = dfX.select_dtypes(include=[np.number])
        if dfX_num.shape[1] == 0:
            raise ValueError("vowel dataset has no numeric features after filtering")
        X = dfX_num.to_numpy(dtype=np.float32)
        y = _label_encode(np.asarray(d["target"]))  # 11 classes
        return X, y
    if name_low in ("ecoli", "e.coli", "e_coli"):
        d = fetch_openml("ecoli", version=1, as_frame=False)
        X = d["data"].astype(np.float32)
        y = _label_encode(d["target"])
        return X, y
    if name_low in ("mass_cyto_13", "mass", "cyto"):
        # Optional CSV: input/mass_cytometry_13.csv with columns f0..f12 and label
        csv_path = Path("input/mass_cytometry_13.csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Mass cytometry CSV not found: {csv_path}. Provide a CSV with 13 feature columns and 'label'"
            )
        df = pd.read_csv(csv_path)
        y = _label_encode(df["label"].values)
        X = df.drop(columns=["label"]).values.astype(np.float32)
        return X, y
    raise ValueError(f"Unknown dataset: {name}")


# ========= DR helpers =========


def _k_list(D: int) -> List[int]:
    # 2..D-1, 4 values roughly evenly spaced
    if D <= 3:
        return [2] if D >= 3 else [D]
    ks = np.linspace(2, max(2, D - 1), num=4)
    ks = sorted({int(round(k)) for k in ks if 1 < k < D})
    return ks or [min(2, D - 1)]


def _median_heuristic_gamma(X: np.ndarray, max_samples: int = 2000) -> float:
    n = min(len(X), max_samples)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = X[idx]
    # pairwise distances
    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    d = np.sqrt(d2 + 1e-12)
    med = np.median(d[d > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / (2.0 * (med ** 2))


def _fit_transform_umap(X_dr: np.ndarray, Xte_list: List[np.ndarray], y_dr: Optional[np.ndarray], n_components: int,
                        n_neighbors: int, min_dist: float, metric: str, random_state: int) -> List[np.ndarray]:
    try:
        from umap import UMAP  # type: ignore
    except Exception as e:
        raise RuntimeError("umap-learn is not installed. Install optional extra 'umap'.") from e
    um = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    t0 = time.perf_counter()
    Xt = um.fit_transform(X_dr, y=y_dr)
    outs = [um.transform(X) for X in Xte_list]
    vprint(f"  UMAP fit+transform done in {time.perf_counter() - t0:.2f}s (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})")
    return [Xt, *outs]


# ========= Core experiment =========


@dataclass
class ExpConfig:
    datasets: List[str]
    repeats: int = 10
    n_train_list: Tuple[int, ...] = (20, 50, 100, 200, 500)
    test_size: float = 0.2
    base_seed: int = 42

    # DR hyperparameters
    umap_n_neighbors: Tuple[int, ...] = (5, 10, 20)
    umap_min_dist: Tuple[float, ...] = (0.0, 0.1, 0.5)
    umap_metrics: Tuple[str, ...] = ("euclidean", "manhattan")
    kpca_gamma_mode: Tuple[str, ...] = ("med", "med2")  # med: 1/(2*med^2), med2: 1/(med^2)

    # Classifier grids
    lin_svm_C: Tuple[float, ...] = (0.1, 1.0, 10.0)
    rbf_svm_C: Tuple[float, ...] = (0.1, 1.0, 10.0)
    rbf_svm_gamma: Tuple[str, ...] = ("scale", "auto")


def _select_dr_pool(X_pool: np.ndarray, y_pool: np.ndarray, size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    size = int(size)
    if size <= 0:
        return X_pool, y_pool
    n = len(y_pool)
    # If requesting fewer than available, sample stratified; if equal or more, just return all or bootstrap beyond
    if size < n:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
        idx = next(sss.split(X_pool, y_pool))[0]
        return X_pool[idx], y_pool[idx]
    if size == n:
        # Return full pool without sampling to avoid sklearn's constraint (train_size must be < n if int)
        return X_pool, y_pool
    # Do NOT bootstrap: cap at all available
    return X_pool, y_pool


def _build_dr_indices_stratified(y_pool: np.ndarray, include_idx: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    """
    Build a set of indices for the DR pool without replacement:
    - Always include `include_idx` (e.g., training indices)
    - Add as many indices as needed from the remaining pool to reach `target_size`
      using class-proportional quotas (approximately stratified)
    - If total available is smaller than target_size, return all available
    """
    n = len(y_pool)
    target_size = int(min(target_size, n))
    include_idx = np.asarray(include_idx, dtype=int)
    include_idx = np.unique(include_idx)
    if len(include_idx) >= target_size:
        return include_idx[:target_size]

    all_idx = np.arange(n, dtype=int)
    mask_inc = np.zeros(n, dtype=bool)
    mask_inc[include_idx] = True
    rem_idx = all_idx[~mask_inc]

    classes, counts = np.unique(y_pool, return_counts=True)
    c2i = {c: i for i, c in enumerate(classes)}
    total = counts.sum()
    # desired per-class counts (proportional rounding)
    prop = counts / total
    raw_targets = prop * target_size
    base = np.floor(raw_targets).astype(int)
    remainder = target_size - base.sum()
    # distribute remainder to classes with largest fractional parts
    frac = raw_targets - base
    order = np.argsort(-frac)
    for t in order[:remainder]:
        base[t] += 1

    # current counts from include_idx
    y_inc = y_pool[include_idx]
    cur_counts = np.zeros_like(base)
    if len(y_inc) > 0:
        _, cc = np.unique(y_inc, return_counts=True)
        # careful: np.unique returns in sorted class order
        for cls, cnt in zip(*np.unique(y_inc, return_counts=True)):
            cur_counts[c2i[cls]] = cnt

    need = np.maximum(0, base - cur_counts)

    rng = np.random.RandomState(seed)
    chosen = [include_idx]

    # group remaining indices by class
    rem_by_class: Dict[int, np.ndarray] = {}
    for cls in classes:
        rem_idx_c = rem_idx[y_pool[rem_idx] == cls]
        rng.shuffle(rem_idx_c)
        rem_by_class[c2i[cls]] = rem_idx_c

    # take per-class needs
    taken_total = len(include_idx)
    for ci, need_i in enumerate(need):
        if need_i <= 0:
            continue
        pool_c = rem_by_class.get(ci, np.array([], dtype=int))
        take = int(min(need_i, len(pool_c)))
        if take > 0:
            chosen.append(pool_c[:take])
            rem_by_class[ci] = pool_c[take:]
            taken_total += take

    # if still short, fill from remaining in round-robin across classes
    remaining_needed = target_size - taken_total
    if remaining_needed > 0:
        # flatten leftover pools
        leftovers = [arr for arr in rem_by_class.values() if len(arr) > 0]
        if leftovers:
            rest = np.concatenate(leftovers)
            rng.shuffle(rest)
            chosen.append(rest[:remaining_needed])

    return np.unique(np.concatenate(chosen))


def _fit_eval_classifier(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                         linear: bool, cfg: ExpConfig, seed: int) -> Tuple[float, Dict[str, object]]:
    # Small grid with inner CV for selection
    # Determine feasible n_splits based on the least populated class to avoid ValueError
    _, counts = np.unique(ytr, return_counts=True)
    min_class_count = int(np.min(counts)) if len(counts) > 0 else 0
    if min_class_count >= 2:
        n_splits = min(3, min_class_count)
        skf: Optional[StratifiedKFold] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        skf = None  # too few samples per class for CV; fall back to defaults
    if linear:
        params = [{"svc__C": [float(c) for c in cfg.lin_svm_C]}]
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="linear", random_state=seed)),
        ])
    else:
        params = [{"svc__C": [float(c) for c in cfg.rbf_svm_C], "svc__gamma": list(cfg.rbf_svm_gamma)}]
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="rbf", random_state=seed)),
        ])
    best_score = -np.inf
    best_params: Dict[str, object] = {}
    if skf is not None:
        for p in params:
            for C in p.get("svc__C", [1.0]):
                for gamma in p.get("svc__gamma", [None]):
                    model = pipe.set_params(**{"svc__C": C, **({"svc__gamma": gamma} if gamma is not None else {})})
                    cv_scores: List[float] = []
                    for tr_idx, va_idx in skf.split(Xtr, ytr):
                        model.fit(Xtr[tr_idx], ytr[tr_idx])
                        pred = model.predict(Xtr[va_idx])
                        cv_scores.append(accuracy_score(ytr[va_idx], pred))
                    mean_cv = float(np.mean(cv_scores))
                    if mean_cv > best_score:
                        best_score = mean_cv
                        best_params = {"C": C, **({"gamma": gamma} if gamma is not None else {})}
    # retrain on all with best (or default) params
    default_params: Dict[str, object] = {"C": float(cfg.lin_svm_C[1] if linear and len(cfg.lin_svm_C) > 1 else 1.0)}
    if not linear:
        # Prefer 'scale' when available
        default_gamma = cfg.rbf_svm_gamma[0] if len(cfg.rbf_svm_gamma) > 0 else "scale"
        default_params.update({"gamma": default_gamma})
    use_params = {**default_params, **best_params}
    final = pipe.set_params(**{
        "svc__C": use_params.get("C", 1.0),
        **({"svc__gamma": use_params.get("gamma")} if not linear and use_params.get("gamma") is not None else {})
    })
    t0 = time.perf_counter()
    final.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, final.predict(Xte)))
    # Return only classifier-related params
    ret_params = {k: v for k, v in use_params.items() if k in ("C", "gamma")}
    vprint(f"    SVM({'linear' if linear else 'rbf'}) train+eval: {time.perf_counter() - t0:.2f}s, acc={acc:.3f}, params={ret_params}")
    return acc, ret_params


def _fit_transform_linear_dr(name: str, k: int, X_dr: np.ndarray, y_dr: np.ndarray,
                             X_list: List[np.ndarray], n_classes: int) -> List[np.ndarray]:
    if name == "PCA":
        dr = PCA(n_components=k, random_state=0)
        t0 = time.perf_counter()
        dr.fit(X_dr)
        outs = [dr.transform(X) for X in X_list]
        vprint(f"  PCA(k={k}) transform: {time.perf_counter() - t0:.2f}s")
        return outs
    if name == "LDA":
        k_eff = min(k, max(1, n_classes - 1))
        dr = LDA(n_components=k_eff)
        t0 = time.perf_counter()
        dr.fit(X_dr, y_dr)
        outs = [dr.transform(X) for X in X_list]
        vprint(f"  LDA(k={k_eff}) transform: {time.perf_counter() - t0:.2f}s")
        return outs
    if name == "NCA":
        # NCA learns a linear transform; use components=k
        k_eff = min(k, X_dr.shape[1])
        dr = NCA(n_components=k_eff, random_state=0, max_iter=200)
        t0 = time.perf_counter()
        dr.fit(X_dr, y_dr)
        outs = [dr.transform(X) for X in X_list]
        vprint(f"  NCA(k={k_eff}) transform: {time.perf_counter() - t0:.2f}s")
        return outs
    raise ValueError(f"Unknown linear DR: {name}")


def _fit_transform_nonlinear_dr(name: str, k: int, X_dr: np.ndarray, y_dr: Optional[np.ndarray],
                                X_list: List[np.ndarray], seed: int,
                                cfg: ExpConfig) -> List[np.ndarray]:
    if name == "UMAP":
        best = None
        best_score = -np.inf
        # Small unsupervised grid (use labels if provided for SupUMAP notion)
        vprint(f"  [UMAP] start grid search k={k}: {len(cfg.umap_n_neighbors)*len(cfg.umap_min_dist)*len(cfg.umap_metrics)} combos")
        for nn in cfg.umap_n_neighbors:
            for md in cfg.umap_min_dist:
                for metric in cfg.umap_metrics:
                    try:
                        t0 = time.perf_counter()
                        outs = _fit_transform_umap(X_dr, X_list, y_dr, k, nn, md, metric, seed)
                        # outs = [Xt_on_pool, transform(X_list[0]), transform(X_list[1]), ...]
                        Xt, *Xtes = outs
                        # quick neighborhood preservation proxy: use variance of fitted embedding
                        score = float(np.var(Xt))
                        vprint(f"    tried nn={nn}, min_dist={md}, metric={metric} -> score(var)={score:.4f}, {time.perf_counter()-t0:.2f}s")
                        if score > best_score:
                            # Return only transforms corresponding to X_list (e.g., [Z_tr, Z_te])
                            best = Xtes
                            best_score = score
                    except Exception:
                        continue
        if best is None:
            # fallback single run
            vprint("  [UMAP] all combos failed, fallback to first config")
            _, *rest = _fit_transform_umap(
                X_dr, X_list, y_dr, k, cfg.umap_n_neighbors[0], cfg.umap_min_dist[0], cfg.umap_metrics[0], seed
            )
            return rest
        return best
    if name == "KernelPCA":
        # Two gamma settings based on median heuristic
        g_med = _median_heuristic_gamma(X_dr)
        gamma_list = [g_med, 2 * g_med]
        best = None
        best_var = -np.inf
        vprint(f"  [KernelPCA] try gammas={len(gamma_list)} for k={k}")
        for g in gamma_list:
            kp = KernelPCA(n_components=k, kernel="rbf", gamma=g, fit_inverse_transform=False, random_state=seed)
            try:
                t0 = time.perf_counter()
                kp.fit(X_dr)
                outs = [kp.transform(X) for X in X_list]
                score = float(np.var(outs[0]))
                vprint(f"    gamma={g:.3e} -> var={score:.4f}, {time.perf_counter()-t0:.2f}s")
                if score > best_var:
                    best = outs
                    best_var = score
            except Exception:
                continue
        if best is None:
            kp = KernelPCA(n_components=k, kernel="rbf", gamma=g_med, fit_inverse_transform=False, random_state=seed)
            kp.fit(X_dr)
            return [kp.transform(X) for X in X_list]
        return best
    if name == "Isomap":
        # choose n_neighbors from {5,10,20} by simple heuristic
        best = None
        best_var = -np.inf
        vprint(f"  [Isomap] try neighbors in (5,10,20) for k={k}")
        for nn in (5, 10, 20):
            try:
                iso = Isomap(n_neighbors=nn, n_components=k)
                t0 = time.perf_counter()
                iso.fit(X_dr)
                outs = [iso.transform(X) for X in X_list]
                score = float(np.var(outs[0]))
                vprint(f"    n_neighbors={nn} -> var={score:.4f}, {time.perf_counter()-t0:.2f}s")
                if score > best_var:
                    best = outs
                    best_var = score
            except Exception:
                continue
        if best is None:
            iso = Isomap(n_neighbors=10, n_components=k)
            iso.fit(X_dr)
            return [iso.transform(X) for X in X_list]
        return best
    raise ValueError(f"Unknown nonlinear DR: {name}")


def run_experiment(cfg: ExpConfig, out_csv: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def _attach_method_comp_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Attach columns:
        - m1_lt_m3: 1 if best raw < best nonlinear_dr within (dataset, repeat, n_train, clf)
        - m2_lt_m3: 1 if best linear_dr < best nonlinear_dr within the same group
        - both1: 1 if both conditions are 1
        Missing methods in a group yield 0.
        """
        if df.empty:
            return df.assign(m1_lt_m3=pd.Series(dtype=int), m2_lt_m3=pd.Series(dtype=int), both1=pd.Series(dtype=int))
        keys = ["dataset", "repeat", "n_train", "clf"]
        best = df.groupby(keys + ["method"])['acc'].max().unstack("method")
        # Ensure columns exist
        for col in ("raw", "linear_dr", "nonlinear_dr"):
            if col not in best.columns:
                best[col] = np.nan
        def lt(a: pd.Series, b: pd.Series) -> pd.Series:
            return ((a < b) & a.notna() & b.notna()).astype(int)
        flags = pd.DataFrame(index=best.index)
        flags["m1_lt_m3"] = lt(best["raw"], best["nonlinear_dr"])  # 手法1(=raw) < 手法3(=nonlinear)
        flags["m2_lt_m3"] = lt(best["linear_dr"], best["nonlinear_dr"])  # 手法2(=linear) < 手法3(=nonlinear)
        flags["both1"] = ((flags["m1_lt_m3"] == 1) & (flags["m2_lt_m3"] == 1)).astype(int)
        flags = flags.reset_index()
        df2 = df.merge(flags, on=keys, how="left")
        # Fill NaN flags with 0 for groups missing methods
        for c in ("m1_lt_m3", "m2_lt_m3", "both1"):
            if c in df2:
                df2[c] = df2[c].fillna(0).astype(int)
        return df2

    for ds_name in cfg.datasets:
        X, y = load_dataset(ds_name)
        D = X.shape[1]
        # Fixed test split
        X_trp, X_te, y_trp, y_te = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.base_seed
        )
        # usable n_train list
        n_max = len(y_trp)
        n_train_list = [n for n in cfg.n_train_list if n <= n_max and n >= 5]
        if not n_train_list:
            n_train_list = [min(50, n_max)]
        ks = _k_list(D)

        for r in range(cfg.repeats):
            seed = cfg.base_seed + r
            for n_train in n_train_list:
                # sample labeled training set
                if n_train >= len(y_trp):
                    # Use full training pool
                    X_tr, y_tr = X_trp, y_trp
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
                    tr_idx = next(sss.split(X_trp, y_trp))[0]
                    X_tr, y_tr = X_trp[tr_idx], y_trp[tr_idx]
                # DR pool size = up to 10x, using only actual remaining data (no bootstrap)
                dr_size = min(10 * n_train, len(y_trp))
                if n_train >= len(y_trp):
                    dr_idx = np.arange(len(y_trp))
                else:
                    # build DR indices including training indices plus stratified from the remainder
                    # obtain training indices within the training pool
                    # If we sampled via SSS above, we have `tr_idx`; else it's full
                    if n_train >= len(y_trp):
                        tr_idx_local = np.arange(len(y_trp))
                    else:
                        # reconstruct tr_idx by matching X_tr rows in X_trp is costly; keep from SSS above
                        # We saved tr_idx in the branch where we sampled
                        tr_idx_local = tr_idx if 'tr_idx' in locals() else np.arange(len(y_trp))
                    dr_idx = _build_dr_indices_stratified(y_trp, tr_idx_local, dr_size, seed)
                X_dr, y_dr = X_trp[dr_idx], y_trp[dr_idx]

                # Meta info for logging/records
                train_n = int(len(y_tr))
                dr_pool_n = int(len(y_dr))
                test_n = int(len(y_te))
                n_classes = int(len(np.unique(y)))
                dr_ratio = float(dr_pool_n / train_n) if train_n > 0 else np.nan
                vprint(
                    f"[dataset={ds_name}] repeat={r}/{cfg.repeats-1} "
                    f"n_train={train_n} -> dr_pool={dr_pool_n} (ratio={dr_ratio:.2f}), test={test_n}, classes={n_classes}"
                )
                meta = {
                    "train_n": train_n,
                    "dr_pool_n": dr_pool_n,
                    "dr_ratio": dr_ratio,
                    "test_n": test_n,
                    "n_classes": n_classes,
                    # このスクリプトは単一データ（機関数の概念なし）。参考値として1を記録。
                    "n_institution": 1,
                }

                # Method 1: raw (no DR)
                for clf_name, linear in (("LinearSVM", True), ("RBFSVM", False)):
                    vprint(f"  [raw] clf={clf_name} start")
                    acc, bestp = _fit_eval_classifier(X_tr, y_tr, X_te, y_te, linear=linear, cfg=cfg, seed=seed)
                    records.append({
                        "dataset": ds_name, "repeat": r, "n_train": n_train,
                        "method": "raw", "dr": "none", "k": D,
                        "clf": clf_name, "acc": acc,
                        **{f"clf_{k}": v for k, v in bestp.items()},
                        **meta,
                    })

                # Method 2: linear DR (PCA only)
                lin_drs = ["PCA"]  # LDA excluded as requested
                for dr_name in lin_drs:
                    for k in ks:
                        try:
                            vprint(f"  [linear] {dr_name}(k={k}) start")
                            Z_tr, Z_te = _fit_transform_linear_dr(dr_name, k, X_dr, y_dr, [X_tr, X_te], n_classes=len(np.unique(y)))
                        except Exception:
                            continue
                        for clf_name, linear in (("LinearSVM", True), ("RBFSVM", False)):
                            vprint(f"    [linear] clf={clf_name} start")
                            acc, bestp = _fit_eval_classifier(Z_tr, y_tr, Z_te, y_te, linear=linear, cfg=cfg, seed=seed)
                            records.append({
                                "dataset": ds_name, "repeat": r, "n_train": n_train,
                                "method": "linear_dr", "dr": dr_name, "k": k,
                                "clf": clf_name, "acc": acc,
                                **{f"clf_{k}": v for k, v in bestp.items()},
                                **meta,
                            })

                # Method 3: nonlinear DR (UMAP, KernelPCA, Isomap)
                nl_drs = ["UMAP", "KernelPCA", "Isomap"]
                # respect --skip-umap flag
                try:
                    import sys
                    if "--skip-umap" in sys.argv:
                        nl_drs = [d for d in nl_drs if d != "UMAP"]
                except Exception:
                    pass
                for dr_name in nl_drs:
                    for k in ks:
                        try:
                            vprint(f"  [nonlinear] {dr_name}(k={k}) start")
                            Z_tr, Z_te = _fit_transform_nonlinear_dr(dr_name, k, X_dr, None, [X_tr, X_te], seed, cfg)
                        except Exception:
                            continue
                        for clf_name, linear in (("LinearSVM", True), ("RBFSVM", False)):
                            vprint(f"    [nonlinear] clf={clf_name} start")
                            acc, bestp = _fit_eval_classifier(Z_tr, y_tr, Z_te, y_te, linear=linear, cfg=cfg, seed=seed)
                            records.append({
                                "dataset": ds_name, "repeat": r, "n_train": n_train,
                                "method": "nonlinear_dr", "dr": dr_name, "k": k,
                                "clf": clf_name, "acc": acc,
                                **{f"clf_{k}": v for k, v in bestp.items()},
                                **meta,
                            })

        # persist progressively to avoid loss
        df_part = pd.DataFrame(records)
        df_part = _attach_method_comp_flags(df_part)
        df_part.to_csv(out_csv, index=False)
        # Also write aligned per-trial and per-condition means selecting best nonlinear (KernelPCA vs Isomap)
        aligned_df = _write_aligned_views(df_part, out_csv)
        vprint(f"Columns in aligned DataFrame: {list(aligned_df.columns)}")
        # Write aligned summary view with mean, variance, flag-after-mean, and best-k/best-DR info
        _write_summary_views(df_part, aligned_df, out_csv)

    return pd.DataFrame(records)


def _write_aligned_views(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Create an aligned, per-condition summary where raw/PCA/nonlinear-best are on the same row.
    Rules:
    - linear_dr: PCAのみ; kごとのaccを平均して1値
    - nonlinear_dr: KernelPCAとIsomapのうち、k平均が高い方を採用（同点ならKernelPCA優先）
    - raw: そのまま（method=raw の平均、kはDとして1値）
    - その上で m1_lt_m3, m2_lt_m3, both1 を再計算
    出力: <stem>_aligned.csv
    """
    if df.empty:
        empty_path = out_csv.with_name(out_csv.stem + "_aligned.csv")
        empty_path.write_text("", encoding="utf-8")
        return pd.DataFrame(columns=["dataset","repeat","n_train","clf","acc_raw","acc_pca","acc_nl_best","nl_best_dr","m1_lt_m3","m2_lt_m3","both1"])
    keys = ["dataset", "repeat", "n_train", "clf"]
    # raw mean
    raw_mean = (
        df[df["method"] == "raw"]
        .groupby(keys)["acc"].mean()
        .rename("acc_raw")
    )
    # linear: PCA only, mean over k
    lin_pca = (
        df[(df["method"] == "linear_dr") & (df["dr"] == "PCA")]
        .groupby(keys)["acc"].mean()
        .rename("acc_pca")
    )
    # nonlinear: compare KernelPCA vs Isomap using k-mean, pick best (tie-break: KernelPCA)
    nl = df[(df["method"] == "nonlinear_dr") & (df["dr"].isin(["KernelPCA", "Isomap"]))].copy()
    if not nl.empty:
        nl_pv = nl.groupby(keys + ["dr"])['acc'].mean().unstack("dr")
        acc_kpca = nl_pv.get("KernelPCA")
        acc_isomap = nl_pv.get("Isomap")
        # fill for comparison; keep NaN semantics after
        a = acc_kpca.fillna(float("-inf")) if acc_kpca is not None else pd.Series(float("-inf"), index=nl_pv.index)
        b = acc_isomap.fillna(float("-inf")) if acc_isomap is not None else pd.Series(float("-inf"), index=nl_pv.index)
        best_is_kpca = (a >= b)
        acc_nl_best = a.where(best_is_kpca, b)
        nl_best_dr = pd.Series(
            np.where(best_is_kpca, "KernelPCA", "Isomap"), index=nl_pv.index, name="nl_best_dr"
        )
        both_nan = (acc_kpca.isna() if acc_kpca is not None else True) & (acc_isomap.isna() if acc_isomap is not None else True)
        acc_nl_best = acc_nl_best.mask(both_nan, other=np.nan)
        nl_best_dr = nl_best_dr.mask(both_nan, other=np.nan)
        nl_best = pd.DataFrame({"acc_nl_best": acc_nl_best, "nl_best_dr": nl_best_dr})
    else:
        nl_best = pd.DataFrame(columns=["acc_nl_best", "nl_best_dr"]).set_index(pd.MultiIndex.from_arrays([[], [], [], []], names=keys))

    # assemble
    base_index = raw_mean.index.union(lin_pca.index, sort=False).union(nl_best.index, sort=False)
    out = pd.DataFrame(index=base_index)
    out = out.join(raw_mean, how="left").join(lin_pca, how="left").join(nl_best[["acc_nl_best", "nl_best_dr"]], how="left")
    out = out.reset_index()

    # flags
    out["m1_lt_m3"] = ((out["acc_raw"] < out["acc_nl_best"]) & out["acc_raw"].notna() & out["acc_nl_best"].notna()).astype(int)
    out["m2_lt_m3"] = ((out["acc_pca"] < out["acc_nl_best"]) & out["acc_pca"].notna() & out["acc_nl_best"].notna()).astype(int)
    out["both1"] = ((out["m1_lt_m3"] == 1) & (out["m2_lt_m3"] == 1)).astype(int)

    out_path = out_csv.with_name(out_csv.stem + "_aligned.csv")
    out.to_csv(out_path, index=False)
    return out


def _write_summary_views(df_long: pd.DataFrame, aligned_df: pd.DataFrame, out_csv: Path) -> None:
    """Create a summary view with mean, variance, and flag ratios per condition.
    Output: <stem>_summary.csv
    """
    if aligned_df.empty:
        (out_csv.with_name(out_csv.stem + "_aligned_summary.csv")).write_text("", encoding="utf-8")
        return

    keys = ["dataset", "n_train", "clf"]
    # Base summary from aligned (means/vars are already per-repeat values in aligned)
    summary = (
        aligned_df.groupby(keys)
        .agg(
            acc_raw_mean=("acc_raw", "mean"),
            acc_raw_var=("acc_raw", "var"),
            acc_pca_mean=("acc_pca", "mean"),
            acc_pca_var=("acc_pca", "var"),
            acc_nl_best_mean=("acc_nl_best", "mean"),
            acc_nl_best_var=("acc_nl_best", "var"),
        )
        .reset_index()
    )

    # Flags computed AFTER averaging (0/1 per condition)
    def lt_after_mean(a: pd.Series, b: pd.Series) -> pd.Series:
        mask = a.notna() & b.notna()
        return ((a < b) & mask).astype(int)

    summary["m1_lt_m3_ratio"] = lt_after_mean(summary["acc_raw_mean"], summary["acc_nl_best_mean"])  # raw mean < nl_best mean
    summary["m2_lt_m3_ratio"] = lt_after_mean(summary["acc_pca_mean"], summary["acc_nl_best_mean"])  # pca mean < nl_best mean
    summary["both1_ratio"] = ((summary["m1_lt_m3_ratio"] == 1) & (summary["m2_lt_m3_ratio"] == 1)).astype(int)

    # Best-k and best nonlinear DR info computed from the long dataframe
    if not df_long.empty:
        # raw_k (D): take first k per group
        raw = df_long[df_long["method"] == "raw"][keys + ["k"]].drop_duplicates(keys + ["k"])
        raw_k = raw.groupby(keys)["k"].first().rename("raw_k").reset_index()

        # PCA best k by mean acc over repeats
        lin = df_long[(df_long["method"] == "linear_dr") & (df_long["dr"] == "PCA")]
        if not lin.empty:
            lin_mean = lin.groupby(keys + ["k"], as_index=False)["acc"].mean()
            lin_best_idx = lin_mean.groupby(keys)["acc"].idxmax()
            pca_best = lin_mean.loc[lin_best_idx, keys + ["k", "acc"]].rename(columns={"k": "pca_best_k", "acc": "acc_pca_best_mean"})
        else:
            pca_best = pd.DataFrame(columns=keys + ["pca_best_k", "acc_pca_best_mean"])

        # Nonlinear: best per method and then best overall
        nl = df_long[(df_long["method"] == "nonlinear_dr") & (df_long["dr"].isin(["KernelPCA", "Isomap"]))]
        if not nl.empty:
            nl_mean = nl.groupby(keys + ["dr", "k"], as_index=False)["acc"].mean()
            def _best_per_dr(dr_name: str, kcol: str, acccol: str) -> pd.DataFrame:
                sub = nl_mean[nl_mean["dr"] == dr_name]
                if sub.empty:
                    return pd.DataFrame(columns=keys + [kcol, acccol])
                idx = sub.groupby(keys)["acc"].idxmax()
                best = sub.loc[idx, keys + ["k", "acc"]].rename(columns={"k": kcol, "acc": acccol})
                return best
            kpca_best = _best_per_dr("KernelPCA", "kpca_best_k", "acc_kpca_best_mean")
            isomap_best = _best_per_dr("Isomap", "isomap_best_k", "acc_isomap_best_mean")
            best_merge = kpca_best.merge(isomap_best, on=keys, how="outer")
            a = best_merge.get("acc_kpca_best_mean")
            b = best_merge.get("acc_isomap_best_mean")
            a_f = a.fillna(-np.inf) if a is not None else pd.Series(-np.inf, index=best_merge.index)
            b_f = b.fillna(-np.inf) if b is not None else pd.Series(-np.inf, index=best_merge.index)
            is_kpca = a_f >= b_f
            nl_best = pd.DataFrame({
                **{k: best_merge[k] for k in keys},
                "nl_best_dr": np.where(is_kpca, "KernelPCA", "Isomap"),
                "nl_best_k": np.where(is_kpca, best_merge.get("kpca_best_k"), best_merge.get("isomap_best_k")),
                "acc_nl_best_mean_from_bestk": np.where(is_kpca, a_f, b_f),
            })
        else:
            nl_best = pd.DataFrame(columns=keys + ["nl_best_dr", "nl_best_k", "acc_nl_best_mean_from_bestk"])

        # merge best-k info into summary
        summary = summary.merge(raw_k, on=keys, how="left")
        summary = summary.merge(pca_best, on=keys, how="left")
        summary = summary.merge(nl_best, on=keys, how="left")

    out_path = out_csv.with_name(out_csv.stem + "_aligned_summary.csv")
    summary.to_csv(out_path, index=False)


def summarize(csv_path: Path, out_summary: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    grp = df.groupby(["dataset", "method", "dr", "k", "clf"])['acc']
    sumdf = grp.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n"})
    if out_summary is None:
        out_summary = csv_path.with_name(csv_path.stem + "_summary.csv")
    sumdf.to_csv(out_summary, index=False)
    return sumdf


def main():
    parser = argparse.ArgumentParser(description="Small-scale nonlinear vs linear DR classification experiments")
    parser.add_argument("--datasets", nargs="*", default=["iris", "balance", "vowel", "ecoli"], help="Datasets to run")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=str, default=str(Path("output") / "exp_nonlin_small.csv"))
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress and timing information")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP in nonlinear DR to avoid long runs")
    args = parser.parse_args()

    cfg = ExpConfig(datasets=args.datasets, repeats=args.repeats)
    global VERBOSE
    VERBOSE = bool(args.verbose)
    if args.skip_umap:
        # Monkey-patch to remove UMAP from list at runtime
        vprint("[info] Skipping UMAP as requested")
        # We'll wrap run_experiment by temporarily patching the list inside
        pass
    out_csv = Path(args.out)
    df = run_experiment(cfg, out_csv)
    sumdf = summarize(out_csv)
    print(f"Saved results to {out_csv} and summary to {out_csv.with_name(out_csv.stem + '_summary.csv')}")
    print(sumdf.head())


if __name__ == "__main__":
    main()
