from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
    smote_X: Optional[np.ndarray] = None,
    smote_y: Optional[np.ndarray] = None,
    return_labels: bool = False,
    include_public_anchor: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Build anchor samples according to config.anchor_method.
    """
    method = getattr(config, "anchor_method", "gaussian")
    if method == "gaussian":
        np.random.seed(seed=seed)
        return np.random.randn(num_row, num_col)

    if method == "gaussian_not_iso":
        rng = np.random.default_rng(seed)
        A = rng.standard_normal(size=(num_row, num_col))
        scales = rng.uniform(0.1, 2.0, size=(1, num_col))
        A = A * scales
        if return_labels:
            return A, np.zeros((num_row,), dtype=float)
        return A

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

        # スケール変化の有無に関わらず同じ「標準偏差単位 r」でばらつくようにする
        r_raw = getattr(config, "anchor_uniform_radius", 1.0)
        try:
            r = float(r_raw)
        except (TypeError, ValueError):
            r = 1.0
        if not np.isfinite(r) or r <= 0:
            r = 1.0

        col_mean = np.nanmean(X_vals, axis=0)
        col_std = np.nanstd(X_vals, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        col_std = np.where((~np.isfinite(col_std)) | (col_std == 0), 1.0, col_std)

        U = rng.uniform(-r, r, size=(num_row, num_col))
        return col_mean + U * col_std

    if method == "smote":
        rng = np.random.default_rng(seed)

        smote_ratio = getattr(config, "smote_ratio", 1.0)
        try:
            smote_ratio = float(smote_ratio)
        except (TypeError, ValueError):
            smote_ratio = 1.0
        smote_ratio = min(max(smote_ratio, 0.0), 1.0)

        # Public anchor data used as SMOTE source; optionally included in output anchors.
        n_public = int(smote_X.shape[0]) if smote_X is not None else 0
        y_public = np.asarray(smote_y).ravel() if smote_y is not None else None
        if n_public <= 0 or y_public is None or y_public.size != n_public:
            raise RuntimeError("SMOTE anchor requires explicit smote_X and smote_y data.")

        X0_full = np.asarray(smote_X)
        columns = min(X0_full.shape[1], num_col)
        X0 = X0_full[:, :columns]
        y0 = y_public

        # Decide how many public anchors to include in output.
        n_public_keep = min(n_public, num_row) if include_public_anchor else 0
        if n_public_keep > 0:
            idx_pub = rng.choice(n_public, size=n_public_keep, replace=False)
            X_public_keep = X0[idx_pub]
            y_public_keep = y0[idx_pub]
        else:
            X_public_keep = np.zeros((0, columns))
            y_public_keep = np.zeros((0,))

        remaining = max(0, num_row - n_public_keep)
        n_smote = int(round(remaining * smote_ratio))
        n_smote = min(max(n_smote, 0), remaining)
        n_gauss = max(0, remaining - n_smote)

        classes, counts = np.unique(y0, return_counts=True)
        N_total = int(len(y0))
        if N_total == 0:
            X_rand = rng.normal(size=(num_row, columns))
            y_rand = np.full((num_row,), np.nan)
            if return_labels:
                return X_rand, y_rand
            return X_rand

        # decide same-label vs cross-label fraction within SMOTE portion
        raw = getattr(config, "smote_cross_label_ratio", None)
        if raw in (None, ""):
            cross_ratio = 0.1
        else:
            try:
                cross_ratio = float(raw)
            except (TypeError, ValueError):
                cross_ratio = 0.1
        cross_ratio = max(0.0, min(cross_ratio, 1.0))

        n_cross = int(round(n_smote * cross_ratio)) if len(classes) > 1 else 0
        n_cross = min(max(n_cross, 0), n_smote)
        n_same = max(0, n_smote - n_cross)

        # allocate same-label samples per class
        same_target_counts: list[int] = []
        allocated = 0
        for i, _ in enumerate(classes):
            if i < len(classes) - 1:
                n_c = int(round(n_same * (counts[i] / N_total))) if n_same > 0 else 0
                same_target_counts.append(n_c)
                allocated += n_c
            else:
                same_target_counts.append(max(0, n_same - allocated))

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        # 1) public anchors kept as a contiguous prefix if requested
        X_parts.append(X_public_keep)
        y_parts.append(y_public_keep)

        # 2) same-label SMOTE
        if n_same > 0:
            for c, n_gen in zip(classes, same_target_counts):
                if n_gen <= 0:
                    continue
                mask = (y0 == c)
                Xc = X0[mask]
                Nc = Xc.shape[0]
                if Nc == 0:
                    continue

                k = min(6, Nc)
                if Nc == 1:
                    repeated = np.repeat(Xc, repeats=max(n_gen, 1), axis=0)
                    noise = rng.normal(scale=0.01, size=repeated.shape)
                    X_syn = (repeated + noise)[:n_gen]
                else:
                    nbrs = KNeighborsClassifier(n_neighbors=k)
                    y_dummy = np.arange(Nc)
                    nbrs.fit(Xc, y_dummy)
                    neighbors = nbrs.kneighbors(Xc, return_distance=False)

                    interpolated = []
                    while len(interpolated) < n_gen:
                        idx = rng.integers(0, Nc)
                        nn_idx = rng.choice(neighbors[idx])
                        # Allow mild extrapolation beyond [0, 1]
                        eps_raw = getattr(config, "smote_extrap_eps", 0.05)
                        if eps_raw in (None, ""):
                            eps = 0.0
                        else:
                            try:
                                eps = float(eps_raw)
                            except (TypeError, ValueError):
                                eps = 0.0
                        eps = max(0.0, min(eps, 1.0))
                        lam = rng.uniform(-eps, 1.0 + eps)
                        interpolated.append(Xc[idx] + lam * (Xc[nn_idx] - Xc[idx]))
                    X_syn = np.vstack(interpolated[:n_gen])

                y_syn = np.full((X_syn.shape[0],), c)
                X_parts.append(X_syn)
                y_parts.append(y_syn)

        # 3) cross-label interpolation (within [0, 1] between different labels)
        if n_cross > 0 and len(classes) > 1 and N_total > 1:
            X_all = X0
            y_all = y0
            cross_X: list[np.ndarray] = []
            cross_y: list[Any] = []
            while len(cross_X) < n_cross:
                i = rng.integers(0, N_total)
                j = rng.integers(0, N_total)
                if y_all[i] == y_all[j]:
                    continue
                lam = rng.random()  # [0, 1]
                z = X_all[i] + lam * (X_all[j] - X_all[i])
                y_z = y_all[i] if lam <= 0.5 else y_all[j]
                cross_X.append(z)
                cross_y.append(y_z)
            X_cross = np.vstack(cross_X[:n_cross])
            y_cross = np.asarray(cross_y[:n_cross])
            X_parts.append(X_cross)
            y_parts.append(y_cross)

        # 4) Gaussian fill for any remaining anchors
        if n_gauss > 0:
            Xg = rng.normal(size=(n_gauss, columns))
            if classes.size > 0 and N_total > 0:
                probs = counts / float(N_total)
                yg = rng.choice(classes, size=n_gauss, p=probs)
            else:
                yg = np.full((n_gauss,), np.nan)
            X_parts.append(Xg)
            y_parts.append(yg)

        # Separate public vs synthetic parts to keep public anchors as a prefix.
        X_public = X_parts[0] if X_parts else np.zeros((0, columns))
        y_public = y_parts[0] if y_parts else np.zeros((0,))
        X_other = np.vstack(X_parts[1:]) if len(X_parts) > 1 else np.zeros((0, columns))
        y_other = np.concatenate(y_parts[1:]) if len(y_parts) > 1 else np.zeros((0,))

        anchor = np.vstack([X_public, X_other])
        anchor_labels = np.concatenate([y_public, y_other])

        # Ensure exact num_row length (defensive)
        if anchor.shape[0] > num_row:
            anchor = anchor[:num_row]
            anchor_labels = anchor_labels[:num_row]
        elif anchor.shape[0] < num_row:
            deficit = num_row - anchor.shape[0]
            extra_X = rng.normal(size=(deficit, columns))
            extra_y = np.full((deficit,), np.nan)
            anchor = np.vstack([anchor, extra_X])
            anchor_labels = np.concatenate([anchor_labels, extra_y])

        if return_labels:
            return anchor, anchor_labels
        return anchor

    raise ValueError(f"Unknown anchor_method: {method}")


def _valid_label_mask(y_array: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y_array).ravel()
    if y_array.dtype.kind in {"f", "c"}:
        return ~np.isnan(y_array)
    mask = np.array(
        [not (val is None or (isinstance(val, float) and np.isnan(val))) for val in y_array],
        dtype=bool,
    )
    return mask


def assign_anchor_labels(
    *,
    anchors_inter: Sequence[np.ndarray],
    anchors_test_inter: Sequence[np.ndarray],
    Xs_train_inter: Sequence[np.ndarray],
    ys_train: Sequence[np.ndarray],
    k: int = 10,
    max_neighbor_dist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not anchors_inter:
        raise ValueError("assign_anchor_labels requires at least one institution.")
    num_institutions = len(anchors_inter)
    if not (len(anchors_test_inter) == len(Xs_train_inter) == len(ys_train) == num_institutions):
        raise ValueError("anchors_inter, anchors_test_inter, Xs_train_inter, and ys_train must have the same length.")

    labels_flat: list[np.ndarray] = []
    for y in ys_train:
        y_arr = np.asarray(y).ravel()
        mask = _valid_label_mask(y_arr)
        if np.any(mask):
            labels_flat.append(y_arr[mask])
    if not labels_flat:
        raise ValueError("assign_anchor_labels requires at least one valid training label.")
    unique_labels = np.unique(np.concatenate(labels_flat))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    num_anchor = anchors_inter[0].shape[0] if anchors_inter else 0
    num_anchor_test = anchors_test_inter[0].shape[0] if anchors_test_inter else 0
    counts_anchor = np.zeros((num_anchor, len(unique_labels)), dtype=float)
    counts_anchor_test = np.zeros((num_anchor_test, len(unique_labels)), dtype=float)

    all_min_dists: list[np.ndarray] = []

    for X_proj, anchor_proj, anchor_test_proj, y_raw in zip(
        Xs_train_inter, anchors_inter, anchors_test_inter, ys_train
    ):
        if X_proj is None or len(X_proj) == 0:
            continue
        X_arr = np.asarray(X_proj)
        y_arr = np.asarray(y_raw).ravel()
        mask = _valid_label_mask(y_arr)
        if not np.any(mask):
            continue
        X_arr = X_arr[mask]
        y_arr = y_arr[mask]
        if X_arr.shape[0] == 0:
            continue
        k_eff = max(1, min(int(k), X_arr.shape[0]))
        knn = KNeighborsClassifier(n_neighbors=k_eff)
        knn.fit(X_arr, y_arr)

        anchor_proj = np.asarray(anchor_proj)
        if anchor_proj.size > 0:
            if max_neighbor_dist > 0.0:
                neighbor_dist, neighbor_idx = knn.kneighbors(anchor_proj, return_distance=True)
                min_dists = np.min(neighbor_dist, axis=1)
                all_min_dists.append(min_dists)
                # 近傍データが十分近くにないアンカーは無ラベル候補として扱う
                far_mask = min_dists > float(max_neighbor_dist)
                _accumulate_label_counts(counts_anchor, neighbor_idx, y_arr, label_to_idx)
                if np.any(far_mask):
                    counts_anchor[far_mask, :] = 0.0
            else:
                neighbor_idx = knn.kneighbors(anchor_proj, return_distance=False)
                _accumulate_label_counts(counts_anchor, neighbor_idx, y_arr, label_to_idx)

        if anchor_test_proj is not None:
            anchor_test_arr = np.asarray(anchor_test_proj)
            if anchor_test_arr.size > 0:
                if max_neighbor_dist > 0.0:
                    neighbor_dist_test, neighbor_idx_test = knn.kneighbors(
                        anchor_test_arr, return_distance=True
                    )
                    far_mask_test = np.min(neighbor_dist_test, axis=1) > float(max_neighbor_dist)
                    _accumulate_label_counts(counts_anchor_test, neighbor_idx_test, y_arr, label_to_idx)
                    if np.any(far_mask_test):
                        counts_anchor_test[far_mask_test, :] = 0.0
                else:
                    neighbor_idx_test = knn.kneighbors(anchor_test_arr, return_distance=False)
                    _accumulate_label_counts(counts_anchor_test, neighbor_idx_test, y_arr, label_to_idx)

    # 無ラベル比率と最近傍距離 5% 点を表示（診断用）
    sums_anchor = counts_anchor.sum(axis=1)
    print(f"[assign_anchor_labels] unlabeled anchors: {np.count_nonzero(sums_anchor == 0)}/{sums_anchor.size}")
    if sums_anchor.size > 0:
        num_unlabeled = int(np.count_nonzero(sums_anchor == 0))
        ratio_unlabeled = num_unlabeled / float(sums_anchor.size)
        print(f"[assign_anchor_labels] unlabeled anchors: {num_unlabeled}/{sums_anchor.size} ({ratio_unlabeled:.3f})")
    if all_min_dists:
        all_min_dists_arr = np.concatenate(all_min_dists)
        try:
            p5 = float(np.percentile(all_min_dists_arr, 5))
            print(f"[assign_anchor_labels] 5th percentile of min neighbor distances: {p5}")
        except Exception:
            pass

    fallback_label = unique_labels[np.argmax(counts_anchor.sum(axis=0))]
    use_fallback = max_neighbor_dist <= 0.0
    anchor_labels = _counts_to_labels(
        counts_anchor,
        unique_labels,
        fallback_label,
        use_fallback_for_zeros=use_fallback,
    )
    anchor_test_labels = _counts_to_labels(
        counts_anchor_test,
        unique_labels,
        fallback_label,
        use_fallback_for_zeros=use_fallback,
    )
    return anchor_labels, anchor_test_labels


def _accumulate_label_counts(
    counts_matrix: np.ndarray,
    neighbor_indices: np.ndarray,
    labels: np.ndarray,
    label_to_idx: dict,
) -> None:
    if counts_matrix.size == 0 or neighbor_indices.size == 0:
        return
    max_rows = counts_matrix.shape[0]
    for row_idx, neighbors in enumerate(neighbor_indices):
        if row_idx >= max_rows:
            break
        for neigh in neighbors:
            label = labels[neigh]
            counts_matrix[row_idx, label_to_idx[label]] += 1.0


def _counts_to_labels(
    counts: np.ndarray,
    unique_labels: np.ndarray,
    fallback_label: Any,
    *,
    use_fallback_for_zeros: bool = True,
) -> np.ndarray:
    if counts.size == 0:
        return np.array([], dtype=unique_labels.dtype)
    winners = np.argmax(counts, axis=1)
    labels = unique_labels[winners]
    sums = counts.sum(axis=1)
    zero_mask = sums == 0
    if np.any(zero_mask):
        if use_fallback_for_zeros:
            labels[zero_mask] = fallback_label
        else:
            # 無ラベルを表現：NaN を使う（float 変換されるが後段では _valid_label_mask で除外する）
            labels = labels.astype(float)
            labels[zero_mask] = np.nan
    return labels


def _symmetric_knn_graph(
    points: np.ndarray,
    k_neighbors: int,
    metric: str = "euclidean",
) -> np.ndarray:
    n_samples = points.shape[0]
    if n_samples == 0:
        return np.zeros((0, 0))
    if n_samples == 1:
        return np.zeros((1, 1))
    k_eff = max(1, min(int(k_neighbors), n_samples - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nbrs.fit(points)
    indices = nbrs.kneighbors(points, return_distance=False)
    adjacency = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        adjacency[i, indices[i]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def assign_anchor_labels_cheating(
    *,
    anchor: np.ndarray,
    anchor_test: np.ndarray,
    Xs_train: Sequence[np.ndarray],
    ys_train: Sequence[np.ndarray],
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_all = np.vstack(Xs_train)
    y_train_all = np.hstack(ys_train)

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
    k_neighbors: Optional[int] = None,
    metric: str = "euclidean",
    logger: logger = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if anchor.size == 0 or anchor_y.size == 0:
        if logger:
            try:
                logger.warning("Anchor Laplacian skipped: empty anchor or labels.")
            except Exception:
                pass
        return None, None

    k_val = int(k_neighbors) if k_neighbors is not None else 5
    adjacency = _symmetric_knn_graph(anchor, k_val, metric=metric)

    # 無ラベル（NaN など）と判定されたアンカーは、ラプラシアン構築時には接続を持たないようにする
    valid_mask = _valid_label_mask(anchor_y)
    if not np.all(valid_mask):
        invalid = ~valid_mask
        adjacency[invalid, :] = 0.0
        adjacency[:, invalid] = 0.0

    same_label_mask = anchor_y.reshape(-1, 1) == anchor_y.reshape(1, -1)
    diff_label_mask = ~same_label_mask

    W_within = adjacency * same_label_mask
    D_within = np.diag(W_within.sum(axis=1))
    L_within = D_within - W_within

    W_between = adjacency * diff_label_mask
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


def build_shared_anchor_knn_adjacency(
    *,
    Xs_inter: Sequence[np.ndarray],
    anchors_inter: Sequence[np.ndarray],
    k_neighbors: int = 5,
    metric: str = "euclidean",
    logger: logger = None,
) -> np.ndarray:
    """
    Build a binary adjacency matrix where two samples are adjacent when they share
    at least one common k-NN anchor (anchored in intermediate space).
    """
    if not Xs_inter or not anchors_inter:
        if logger:
            try:
                logger.warning("Adjacency skipped: missing intermediate data or anchors.")
            except Exception:
                pass
        return np.zeros((0, 0))

    if len(Xs_inter) != len(anchors_inter):
        raise ValueError("Xs_inter and anchors_inter must have the same length.")

    num_anchor = anchors_inter[0].shape[0]
    if num_anchor == 0:
        return np.zeros((0, 0))
    for anchor in anchors_inter:
        if anchor.shape[0] != num_anchor:
            raise ValueError("All anchor projections must share the same number of rows.")

    total_samples = sum(X.shape[0] for X in Xs_inter)
    if total_samples == 0:
        return np.zeros((0, 0))

    indicator = np.zeros((total_samples, num_anchor), dtype=bool)
    row_offset = 0
    k_neighbors = max(1, int(k_neighbors))

    for inst_idx, (X_inst, anchor_inst) in enumerate(zip(Xs_inter, anchors_inter)):
        n_samples = X_inst.shape[0]
        if n_samples == 0:
            continue
        k_eff = min(k_neighbors, anchor_inst.shape[0])
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
        nbrs.fit(anchor_inst)
        neighbor_idx = nbrs.kneighbors(X_inst, return_distance=False)
        rows = np.arange(row_offset, row_offset + n_samples)[:, None]
        indicator[rows, neighbor_idx] = True
        row_offset += n_samples

    adjacency_counts = indicator @ indicator.T
    adjacency = (adjacency_counts > 0).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def build_laplacians_from_intermediate_data(
    *,
    Xs_inter: Sequence[np.ndarray],
    anchors_inter: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
    k_neighbors: int = 5,
    metric: str = "euclidean",
    normalize: bool = True,
    logger: logger = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build k-NN adjacency and label-aware Laplacians for actual intermediate data
    (not anchors). Two samples are adjacent when they share at least one anchor
    among their k nearest anchor neighbors.
    """
    adjacency = build_shared_anchor_knn_adjacency(
        Xs_inter=Xs_inter,
        anchors_inter=anchors_inter,
        k_neighbors=k_neighbors,
        metric=metric,
        logger=logger,
    )
    if adjacency.size == 0:
        return adjacency, np.zeros((0, 0)), np.zeros((0, 0))

    labels = np.hstack(ys) if ys else np.array([])
    if labels.size != adjacency.shape[0]:
        raise ValueError("Label count must match the total number of samples in Xs_inter.")

    same_mask = labels.reshape(-1, 1) == labels.reshape(1, -1)
    diff_mask = ~same_mask

    W_within = adjacency * same_mask
    D_within = np.diag(W_within.sum(axis=1))
    L_within = D_within - W_within

    W_between = adjacency * diff_mask
    D_between = np.diag(W_between.sum(axis=1))
    L_between = D_between - W_between

    if normalize:
        trace_w = np.trace(L_within)
        if trace_w > 1e-9:
            L_within = L_within / trace_w
        trace_b = np.trace(L_between)
        if trace_b > 1e-9:
            L_between = L_between / trace_b

    return adjacency, L_within, L_between
