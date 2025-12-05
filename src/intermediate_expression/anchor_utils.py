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
    print(11111234566666666611111)
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
