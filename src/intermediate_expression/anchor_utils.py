from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.task_utils import is_regression_task

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

        # SMOTE/ガウスの割り当て:
        # - 全体 num_row に対して smote_ratio を掛けて SMOTE 枠を決める
        # - 公開アンカーはその SMOTE 枠を優先的に埋める（枠を超える分は捨てる）
        total_smote_target = int(round(num_row * smote_ratio))
        total_smote_target = min(max(total_smote_target, 0), num_row)

        # Decide how many public anchors to include in output (capped by SMOTE枠).
        # smote_ratio が 0 のときは公開アンカーも入れず、全枠をガウスで埋める。
        if include_public_anchor and total_smote_target > 0:
            n_public_keep = min(n_public, num_row, total_smote_target)
        else:
            n_public_keep = 0

        if n_public_keep > 0:
            idx_pub = rng.choice(n_public, size=n_public_keep, replace=False)
            X_public_keep = X0[idx_pub]
            y_public_keep = y0[idx_pub]
        else:
            X_public_keep = np.zeros((0, columns))
            y_public_keep = np.zeros((0,))

        remaining_slots = max(0, num_row - n_public_keep)
        if include_public_anchor:
            smote_quota_after_public = max(total_smote_target - n_public_keep, 0)
            n_smote = min(smote_quota_after_public, remaining_slots)
        else:
            n_smote = min(total_smote_target, remaining_slots)
        n_gauss = max(0, remaining_slots - n_smote)
        n_uniform = max(0, remaining_slots - n_smote)

        classes, counts = np.unique(y0, return_counts=True)
        N_total = int(len(y0))
        if N_total == 0:
            X_rand = rng.normal(size=(num_row, columns))
            y_rand = np.full((num_row,), np.nan)
            if return_labels:
                return X_rand, y_rand
            return X_rand

        if is_regression_task(config, update_config=False):
            X_parts: list[np.ndarray] = [X_public_keep]
            y_parts: list[np.ndarray] = [y_public_keep.astype(float, copy=False)]

            if n_smote > 0:
                if N_total == 1:
                    X_syn = np.repeat(X0, repeats=n_smote, axis=0)
                    X_syn = X_syn + rng.normal(scale=0.01, size=X_syn.shape)
                    y_syn = np.repeat(y0.astype(float), repeats=n_smote)
                else:
                    k = min(10, N_total)
                    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
                    nn.fit(X0)
                    neighbors = nn.kneighbors(X0, return_distance=False)
                    X_syn_rows: list[np.ndarray] = []
                    y_syn_rows: list[float] = []
                    y_float = y0.astype(float)
                    while len(X_syn_rows) < n_smote:
                        i = int(rng.integers(0, N_total))
                        candidates = [int(j) for j in neighbors[i] if int(j) != i]
                        j = int(rng.choice(candidates)) if candidates else int(rng.integers(0, N_total))
                        lam = float(rng.random())
                        X_syn_rows.append(X0[i] + lam * (X0[j] - X0[i]))
                        y_syn_rows.append(float(y_float[i] + lam * (y_float[j] - y_float[i])))
                    X_syn = np.vstack(X_syn_rows)
                    y_syn = np.asarray(y_syn_rows, dtype=float)
                X_parts.append(X_syn)
                y_parts.append(y_syn)

            if n_uniform > 0:
                x_min = X0.min(axis=0)
                x_max = X0.max(axis=0)
                same = x_min == x_max
                x_max = np.where(same, x_min + 1e-6, x_max)
                Xu = rng.uniform(low=x_min, high=x_max, size=(n_uniform, columns))
                nn_y = NearestNeighbors(n_neighbors=min(5, N_total), metric="euclidean")
                nn_y.fit(X0)
                dist_u, idx_u = nn_y.kneighbors(Xu, return_distance=True)
                weights = 1.0 / np.maximum(dist_u, 1e-12)
                y_float = y0.astype(float)
                yu = (weights * y_float[idx_u]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
                X_parts.append(Xu)
                y_parts.append(yu)

            anchor = np.vstack(X_parts)
            anchor_labels = np.concatenate(y_parts)
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

        # decide same-label vs cross-label fraction within SMOTE portion
        raw = getattr(config, "smote_cross_label_ratio", 0)
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

                #k = min(6, Nc)
                k = min(10, Nc)
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
                        #lam = rng.uniform(-eps, 1.0 + eps)
                        lam = rng.uniform(0.0, 1.0 + eps)
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
        # if n_gauss > 0:
        #     if X0.size > 0:
        #         mu = X0.mean(axis=0)
        #         std = X0.std(axis=0)
        #         std = np.where(std == 0, 1e-6, std)  # avoid zero-variance
        #         Xg = rng.normal(loc=mu, scale=std, size=(n_gauss, columns))
        #     else:
        #         Xg = rng.normal(size=(n_gauss, columns))

        #     if classes.size > 0 and N_total > 0:
        #         probs = counts / float(N_total)
        #         yg = rng.choice(classes, size=n_gauss, p=probs)
        #     else:
        #         yg = np.full((n_gauss,), np.nan)
        #     X_parts.append(Xg)
        #     y_parts.append(yg)

        if n_uniform > 0:
            if X0.size > 0:
                # 各特徴ごとに min/max を使う（よりデータ分布に近い）
                x_min = X0.min(axis=0)
                x_max = X0.max(axis=0)

                # 万一 min == max の場合の対策
                same = x_min == x_max
                x_max = np.where(same, x_min + 1e-6, x_max)

                Xu = rng.uniform(low=x_min, high=x_max, size=(n_uniform, columns))
            else:
                # 完全な 0–1 一様分布
                Xu = rng.uniform(0.0, 1.0, size=(n_uniform, columns))

            if classes.size > 0 and N_total > 0:
                probs = counts / float(N_total)
                yu = rng.choice(classes, size=n_uniform, p=probs)
            else:
                yu = np.full((n_uniform,), np.nan)

            X_parts.append(Xu)
            y_parts.append(yu)


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


def assign_anchor_regression_targets(
    *,
    anchors_inter: Sequence[np.ndarray],
    anchors_test_inter: Sequence[np.ndarray],
    Xs_train_inter: Sequence[np.ndarray],
    ys_train: Sequence[np.ndarray],
    k: int = 10,
    max_neighbor_dist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not anchors_inter:
        raise ValueError("assign_anchor_regression_targets requires at least one institution.")
    num_institutions = len(anchors_inter)
    if not (len(anchors_test_inter) == len(Xs_train_inter) == len(ys_train) == num_institutions):
        raise ValueError("anchors_inter, anchors_test_inter, Xs_train_inter, and ys_train must have the same length.")

    num_anchor = anchors_inter[0].shape[0] if anchors_inter else 0
    num_anchor_test = anchors_test_inter[0].shape[0] if anchors_test_inter else 0
    sum_anchor = np.zeros((num_anchor,), dtype=float)
    weight_anchor = np.zeros((num_anchor,), dtype=float)
    sum_anchor_test = np.zeros((num_anchor_test,), dtype=float)
    weight_anchor_test = np.zeros((num_anchor_test,), dtype=float)

    all_min_dists: list[np.ndarray] = []

    def _accumulate(query: np.ndarray, X_arr: np.ndarray, y_arr: np.ndarray, sums: np.ndarray, weights_total: np.ndarray):
        if query is None or np.asarray(query).size == 0:
            return
        query_arr = np.asarray(query)
        k_eff = max(1, min(int(k), X_arr.shape[0]))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(X_arr)
        dists, idx = nn.kneighbors(query_arr, return_distance=True)
        if dists.size:
            all_min_dists.append(np.min(dists, axis=1))
        weights = 1.0 / np.maximum(dists, 1e-12)
        if max_neighbor_dist > 0.0:
            far_mask = np.min(dists, axis=1) > float(max_neighbor_dist)
            weights[far_mask, :] = 0.0
        pred_sum = (weights * y_arr[idx]).sum(axis=1)
        pred_weight = weights.sum(axis=1)
        rows = min(sums.shape[0], pred_sum.shape[0])
        sums[:rows] += pred_sum[:rows]
        weights_total[:rows] += pred_weight[:rows]

    for X_proj, anchor_proj, anchor_test_proj, y_raw in zip(
        Xs_train_inter, anchors_inter, anchors_test_inter, ys_train
    ):
        if X_proj is None or len(X_proj) == 0:
            continue
        X_arr = np.asarray(X_proj)
        y_arr_raw = np.asarray(y_raw).ravel()
        mask = _valid_label_mask(y_arr_raw)
        if not np.any(mask):
            continue
        X_arr = X_arr[mask]
        y_arr = y_arr_raw[mask].astype(float)
        if X_arr.shape[0] == 0:
            continue
        _accumulate(np.asarray(anchor_proj), X_arr, y_arr, sum_anchor, weight_anchor)
        if anchor_test_proj is not None:
            _accumulate(np.asarray(anchor_test_proj), X_arr, y_arr, sum_anchor_test, weight_anchor_test)

    anchor_y = np.full((num_anchor,), np.nan, dtype=float)
    valid_anchor = weight_anchor > 0
    anchor_y[valid_anchor] = sum_anchor[valid_anchor] / weight_anchor[valid_anchor]

    anchor_y_test = np.full((num_anchor_test,), np.nan, dtype=float)
    valid_anchor_test = weight_anchor_test > 0
    anchor_y_test[valid_anchor_test] = sum_anchor_test[valid_anchor_test] / weight_anchor_test[valid_anchor_test]

    print(f"[assign_anchor_regression_targets] unlabeled anchors: {np.count_nonzero(~valid_anchor)}/{valid_anchor.size}")
    if all_min_dists:
        try:
            p5 = float(np.percentile(np.concatenate(all_min_dists), 5))
            print(f"[assign_anchor_regression_targets] 5th percentile of min neighbor distances: {p5}")
        except Exception:
            pass
    return anchor_y, anchor_y_test


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
    """
    Build label-aware anchor Laplacians using the "within-class graph" and
    "penalty graph" constructions (paper-style).

    Within-class graph (L_within):
      w_ij = 1 if j in N_k^in(i) or i in N_k^in(j), else 0,
    where N_k^in(i) are the k nearest neighbors of i restricted to the same class.

    Penalty graph (L_between):
      w'_ij = 1 if (i, j) is among the k smallest cross-class pairs for some class ℓ,
    where cross-class pairs are formed between points in class ℓ and points outside ℓ.

    Notes:
    - `gamma` is currently unused (kept for config compatibility).
    - Both graphs use the same `k_neighbors` value for k1 (within) and k2 (penalty).
    """
    if anchor.size == 0 or anchor_y.size == 0:
        if logger:
            try:
                logger.warning("Anchor Laplacian skipped: empty anchor or labels.")
            except Exception:
                pass
        return None, None

    k_val = int(k_neighbors) if k_neighbors is not None else 5
    k_val = max(1, k_val)

    anchor = np.asarray(anchor)
    anchor_y = np.asarray(anchor_y).ravel()
    n = int(anchor.shape[0])
    if n <= 1:
        return np.zeros((n, n)), np.zeros((n, n))

    # 無ラベル（NaN など）と判定されたアンカーは、グラフ構築対象から除外する
    valid_mask = _valid_label_mask(anchor_y)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return np.zeros((n, n)), np.zeros((n, n))

    labels_valid = anchor_y[valid_idx]
    points_valid = anchor[valid_idx]

    Ww_valid = np.zeros((valid_idx.size, valid_idx.size), dtype=float)
    Wb_valid = np.zeros((valid_idx.size, valid_idx.size), dtype=float)

    unique_labels = np.unique(labels_valid)

    # --- Within-class graph (L_within) ---
    for lab in unique_labels:
        cls_mask = labels_valid == lab
        cls_idx_local = np.where(cls_mask)[0]
        n_cls = int(cls_idx_local.size)
        if n_cls <= 1:
            continue

        pts_cls = points_valid[cls_idx_local]
        k_eff = max(1, min(k_val, n_cls - 1))
        # Request k_eff+1 to ensure we can drop self-neighbor robustly.
        nn = NearestNeighbors(n_neighbors=min(n_cls, k_eff + 1), metric=metric)
        nn.fit(pts_cls)
        neigh = nn.kneighbors(pts_cls, return_distance=False)

        for row_local, neigh_local in enumerate(neigh):
            src = cls_idx_local[row_local]
            # Drop self index if present, then take first k_eff
            neigh_local = [j for j in neigh_local if j != row_local][:k_eff]
            for j_local in neigh_local:
                dst = cls_idx_local[int(j_local)]
                Ww_valid[src, dst] = 1.0

    Ww_valid = np.maximum(Ww_valid, Ww_valid.T)
    np.fill_diagonal(Ww_valid, 0.0)

    # --- Penalty graph (L_between) ---
    # For each class ℓ, pick the k_val smallest cross-class pairs (i in ℓ, j not in ℓ).
    for lab in unique_labels:
        in_mask = labels_valid == lab
        idx_in = np.where(in_mask)[0]
        idx_out = np.where(~in_mask)[0]
        if idx_in.size == 0 or idx_out.size == 0:
            continue

        pts_in = points_valid[idx_in]
        pts_out = points_valid[idx_out]
        k_out = min(k_val, int(pts_out.shape[0]))
        if k_out <= 0:
            continue

        nn_out = NearestNeighbors(n_neighbors=k_out, metric=metric)
        nn_out.fit(pts_out)
        dists, neigh_out = nn_out.kneighbors(pts_in, return_distance=True)

        # Candidate set: for each i in ℓ, its k2 nearest outside neighbors.
        # Selecting the global k2 smallest among these candidates is exact:
        # if a pair (i, j) is in global top-k2, then j is within i's top-k2 outside neighbors.
        candidates_dist: list[float] = []
        candidates_src: list[int] = []
        candidates_dst: list[int] = []
        for row, src_local in enumerate(idx_in):
            for pos, neigh_rel in enumerate(neigh_out[row]):
                candidates_dist.append(float(dists[row, pos]))
                candidates_src.append(int(src_local))
                candidates_dst.append(int(idx_out[int(neigh_rel)]))

        if not candidates_dist:
            continue

        k_pick = min(k_val, len(candidates_dist))
        dist_arr = np.asarray(candidates_dist, dtype=float)
        pick_idx = np.argpartition(dist_arr, k_pick - 1)[:k_pick]
        for t in pick_idx:
            Wb_valid[candidates_src[int(t)], candidates_dst[int(t)]] = 1.0

    Wb_valid = np.maximum(Wb_valid, Wb_valid.T)
    np.fill_diagonal(Wb_valid, 0.0)

    # Embed back to full (including invalid labels as isolated nodes).
    W_within = np.zeros((n, n), dtype=float)
    W_between = np.zeros((n, n), dtype=float)
    W_within[np.ix_(valid_idx, valid_idx)] = Ww_valid
    W_between[np.ix_(valid_idx, valid_idx)] = Wb_valid

    D_within = np.diag(W_within.sum(axis=1))
    L_within = D_within - W_within

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


def build_laplacians_from_anchor_regression_targets(
    *,
    anchor: np.ndarray,
    anchor_y: np.ndarray,
    k_neighbors: Optional[int] = None,
    metric: str = "euclidean",
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    normalize: bool = True,
    logger: logger = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build anchor Laplacians for continuous regression targets.

    Within graph:
      Ww_ij = A_ij exp(-(y_i-y_j)^2 / sigma_y^2)

    Between graph:
      Wb_ij = A_ij (1 - exp(-(y_i-y_j)^2 / sigma_y^2))

    A_ij is a symmetric kNN mask over anchors. sigma_x is accepted for
    backward-compatible config handling but is not used by this weighting.
    """
    if anchor.size == 0 or anchor_y.size == 0:
        if logger:
            try:
                logger.warning("Regression anchor Laplacian skipped: empty anchor or targets.")
            except Exception:
                pass
        return None, None

    anchor = np.asarray(anchor, dtype=float)
    anchor_y = np.asarray(anchor_y).ravel().astype(float)
    n = int(anchor.shape[0])
    if n <= 1:
        return np.zeros((n, n)), np.zeros((n, n))

    valid_mask = _valid_label_mask(anchor_y)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return np.zeros((n, n)), np.zeros((n, n))

    points = anchor[valid_idx]
    y = anchor_y[valid_idx]
    n_valid = points.shape[0]
    if n_valid <= 1:
        return np.zeros((n, n)), np.zeros((n, n))

    k_val = int(k_neighbors) if k_neighbors is not None else 5
    k_eff = max(1, min(k_val, n_valid - 1))
    nn = NearestNeighbors(n_neighbors=min(n_valid, k_eff + 1), metric=metric)
    nn.fit(points)
    neigh = nn.kneighbors(points, return_distance=False)

    Ww_valid = np.zeros((n_valid, n_valid), dtype=float)
    Wb_valid = np.zeros((n_valid, n_valid), dtype=float)

    dy_values: list[float] = []
    edge_pairs: list[tuple[int, int, float]] = []
    for i, neigh_row in enumerate(neigh):
        kept = 0
        for j in neigh_row:
            j = int(j)
            if j == i:
                continue
            dy = float(abs(y[i] - y[j]))
            edge_pairs.append((i, j, dy))
            dy_values.append(dy)
            kept += 1
            if kept >= k_eff:
                break

    if not edge_pairs:
        return np.zeros((n, n)), np.zeros((n, n))

    sy = float(sigma_y) if sigma_y is not None else 0.0
    if not np.isfinite(sy) or sy <= 0:
        positive_dy = np.asarray([v for v in dy_values if v > 0], dtype=float)
        sy = float(np.median(positive_dy)) if positive_dy.size else float(np.std(y))
        if not np.isfinite(sy) or sy <= 0:
            sy = 1.0

    for i, j, dy in edge_pairs:
        wy = float(np.exp(-(dy * dy) / max(sy * sy, 1e-12)))
        Ww_valid[i, j] = max(Ww_valid[i, j], wy)
        Wb_valid[i, j] = max(Wb_valid[i, j], 1.0 - wy)

    Ww_valid = np.maximum(Ww_valid, Ww_valid.T)
    Wb_valid = np.maximum(Wb_valid, Wb_valid.T)
    np.fill_diagonal(Ww_valid, 0.0)
    np.fill_diagonal(Wb_valid, 0.0)

    W_within = np.zeros((n, n), dtype=float)
    W_between = np.zeros((n, n), dtype=float)
    W_within[np.ix_(valid_idx, valid_idx)] = Ww_valid
    W_between[np.ix_(valid_idx, valid_idx)] = Wb_valid

    L_within = np.diag(W_within.sum(axis=1)) - W_within
    L_between = np.diag(W_between.sum(axis=1)) - W_between

    if normalize:
        trace_Lw = np.trace(L_within)
        if trace_Lw > 1e-9:
            L_within = L_within / trace_Lw
        trace_Lb = np.trace(L_between)
        if trace_Lb > 1e-9:
            L_between = L_between / trace_Lb

    if logger:
        try:
            logger.info(f"Regression L_within shape: {L_within.shape}")
            logger.info(f"Regression L_between shape: {L_between.shape}")
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
