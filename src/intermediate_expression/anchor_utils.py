from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier

logger = Optional[object]


def produce_anchor(
    *,
    num_row: int,
    num_col: int,
    seed: int,
    config,
    train_df: pd.DataFrame,
    Xs_train: Sequence[np.ndarray],
    Xs_test: Sequence[np.ndarray],
    ys_train: Sequence[np.ndarray],
    ys_test: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Build anchor samples according to config.anchor_method.
    """
    method = getattr(config, "anchor_method", "gaussian")
    if method == "gaussian":
        np.random.seed(seed=seed)
        return np.random.randn(num_row, num_col)

    if method == "uniform":
        rng = np.random.default_rng(seed)
        y_name = getattr(config, "y_name", "target")
        if y_name in train_df.columns:
            X_df = train_df.drop(columns=[y_name])
        elif Xs_train:
            X_df = pd.DataFrame(np.vstack(Xs_train))
        else:
            return rng.uniform(-1.0, 1.0, size=(num_row, num_col))

        X_vals = X_df.values
        if X_vals.shape[1] < num_col:
            num_col = X_vals.shape[1]
        X_vals = X_vals[:, :num_col]
        col_min = np.nanmin(X_vals, axis=0)
        col_max = np.nanmax(X_vals, axis=0)
        invalid = ~np.isfinite(col_min) | ~np.isfinite(col_max)
        col_min = np.where(invalid, -1.0, col_min)
        col_max = np.where(invalid, 1.0, col_max)
        width = np.clip(col_max - col_min, 0.0, None)
        U = rng.random((num_row, num_col))
        return col_min + U * width

    if method == "smote":
        rng = np.random.default_rng(seed)

        smote_ratio = getattr(config, "smote_ratio", 1.0)
        try:
            smote_ratio = float(smote_ratio)
        except (TypeError, ValueError):
            smote_ratio = 1.0
        smote_ratio = min(max(smote_ratio, 0.0), 1.0)

        n_smote = int(round(num_row * smote_ratio))
        n_gauss = max(0, num_row - n_smote)

        if n_smote > 0 and (not Xs_train or not ys_train):
            raise RuntimeError("SMOTE anchor requires non-empty institutional data.")

        def _generate_smote_samples(target_rows: int, target_cols: int) -> np.ndarray:
            X_test_all = np.vstack(Xs_test) if len(Xs_test) > 1 else Xs_test[0]
            y_test_all = np.hstack(ys_test) if len(ys_test) > 1 else ys_test[0]
            X0 = np.vstack([X_test_all])
            y0 = np.hstack([y_test_all])

            columns = target_cols
            if X0.shape[1] < columns:
                columns = X0.shape[1]
            X0_clip = X0[:, :columns]

            classes, counts = np.unique(y0, return_counts=True)
            N_total = int(len(y0))
            if N_total == 0:
                return rng.normal(size=(target_rows, columns))

            target_counts = []
            allocated = 0
            for i, _ in enumerate(classes):
                if i < len(classes) - 1:
                    n_c = int(round(target_rows * (counts[i] / N_total)))
                    target_counts.append(n_c)
                    allocated += n_c
                else:
                    target_counts.append(target_rows - allocated)

            synthetic_list = []
            for c, n_gen in zip(classes, target_counts):
                if n_gen <= 0:
                    continue
                mask = (y0 == c)
                Xc = X0_clip[mask]
                Nc = Xc.shape[0]
                if Nc == 0:
                    continue

                k = min(6, Nc)
                if Nc == 1:
                    repeated = np.repeat(Xc, repeats=max(n_gen, 1), axis=0)
                    noise = rng.normal(scale=0.01, size=repeated.shape)
                    synthetic_list.append((repeated + noise)[:n_gen])
                    continue

                nbrs = KNeighborsClassifier(n_neighbors=k)
                y_dummy = np.arange(Nc)
                nbrs.fit(Xc, y_dummy)
                neighbors = nbrs.kneighbors(Xc, return_distance=False)

                interpolated = []
                while len(interpolated) < n_gen:
                    idx = rng.integers(0, Nc)
                    nn_idx = rng.choice(neighbors[idx])
                    lam = rng.random()
                    interpolated.append(Xc[idx] + lam * (Xc[nn_idx] - Xc[idx]))
                synthetic_list.append(np.vstack(interpolated[:n_gen]))

            if synthetic_list:
                Xpub_syn = np.vstack(synthetic_list)
            else:
                Xpub_syn = rng.normal(size=(target_rows, columns))

            if Xpub_syn.shape[0] < target_rows:
                deficit = target_rows - Xpub_syn.shape[0]
                extra = rng.normal(size=(deficit, Xpub_syn.shape[1]))
                Xpub_syn = np.vstack([Xpub_syn, extra])
            return Xpub_syn[:target_rows, :columns]

        samples = []
        effective_cols = num_col

        if n_smote > 0:
            smote_samples = _generate_smote_samples(n_smote, num_col)
            effective_cols = smote_samples.shape[1]
            samples.append(smote_samples)

        if n_gauss > 0:
            gaussian_cols = effective_cols if n_smote > 0 else num_col
            samples.append(rng.normal(size=(n_gauss, gaussian_cols)))

        if not samples:
            return rng.normal(size=(num_row, num_col))

        anchor = np.vstack(samples)
        if len(samples) > 1:
            rng.shuffle(anchor)
        return anchor[:num_row, :anchor.shape[1]]

    raise ValueError(f"Unknown anchor_method: {method}")


def assign_anchor_labels(
    *,
    anchor: np.ndarray,
    anchor_test: np.ndarray,
    Xs_train: Sequence[np.ndarray],
    ys_train: Sequence[np.ndarray],
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_all = np.vstack(Xs_train)
    y_train_all = np.hstack(ys_train)

    def _valid_label_mask(y_array: np.ndarray) -> np.ndarray:
        if y_array.dtype.kind in {"f", "c"}:
            return ~np.isnan(y_array)
        return np.array(
            [not (val is None or (isinstance(val, float) and np.isnan(val))) for val in y_array],
            dtype=bool,
        )

    mask_valid = _valid_label_mask(y_train_all)
    if not np.all(mask_valid):
        X_train_all = X_train_all[mask_valid]
        y_train_all = y_train_all[mask_valid]
    if X_train_all.size == 0:
        raise ValueError("Anchor label assignment failed: no valid labeled samples remain.")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_all, y_train_all)
    anchor_y = knn.predict(anchor)

    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train_all, y_train_all)
    anchor_y_test = knn_test.predict(anchor_test)
    return anchor_y, anchor_y_test


def build_laplacians_from_anchor_labels(
    *,
    anchor: np.ndarray,
    anchor_y: np.ndarray,
    gamma: Optional[float] = None,
    logger: logger = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if anchor.size == 0 or anchor_y.size == 0:
        if logger:
            try:
                logger.warning("Anchor Laplacian skipped: empty anchor or labels.")
            except Exception:
                pass
        return None, None

    n_features = anchor.shape[1]
    if gamma is None:
        gamma = 1.0 / n_features

    W = rbf_kernel(anchor, gamma=gamma)
    np.fill_diagonal(W, 0)

    same_label_mask = anchor_y.reshape(-1, 1) == anchor_y.reshape(1, -1)
    diff_label_mask = ~same_label_mask

    W_within = W * same_label_mask
    D_within = np.diag(W_within.sum(axis=1))
    L_within = D_within - W_within

    W_between = W * diff_label_mask
    D_between = np.diag(W_between.sum(axis=1))
    L_between = D_between - W_between

    trace_Lw = np.trace(L_within)
    if trace_Lw > 1e-9:
        L_within = L_within / trace_Lw

    trace_Lb = np.trace(L_between)
    if trace_Lb > 1e-9:
        L_between = L_between / trace_Lb

    if logger:
        try:
            logger.info(f"L_within shape: {L_within.shape}")
            logger.info(f"L_between shape: {L_between.shape}")
        except Exception:
            pass
    return L_within, L_between
