from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from config.config import Config
from src import dimensionality_reduction
from src.institution_data_pipeline.load_data import load_data
from src.intermediate_expression import anchor_utils


METHOD_ALIAS_TO_INTERNAL: dict[str, str] = {
    "linregr": "pinv",
    "linreg": "pinv",
    "pinv": "pinv",
    "mlp": "mlp",
    "pinv_not_orth": "pinv_not_orth",
    "pinv+orth": "pinv_not_orth",
    "pinv_notorth": "pinv_not_orth",
    "pinv-not-orth": "pinv_not_orth",
    "pinv-not_orth": "pinv_not_orth",
}
METHOD_INTERNAL_TO_LABEL: dict[str, str] = {
    "pinv": "LinRegr",
    "mlp": "MLP",
    "pinv_not_orth": "PINV",
}
METHOD_LABEL_TO_INTERNAL: dict[str, str] = {
    "LinRegr": "pinv",
    "MLP": "mlp",
    "PINV": "pinv_not_orth",
}
DEFAULT_METHODS: tuple[str, ...] = ("LinRegr", "MLP", "PINV")


@dataclass
class LightReconstructionResult:
    config: Config
    methods: tuple[str, ...]
    selected_labels: list[int]
    selected_indices: dict[int, int]
    selected_indices_by_label: dict[int, list[int]]
    selected_label_sequence: list[int]
    selected_originals: np.ndarray
    selected_reconstructions: dict[str, np.ndarray]
    selected_images: dict[str, dict[int, dict[str, np.ndarray]]]
    selected_images_multi: dict[str, dict[int, list[dict[str, np.ndarray]]]]
    user_X: np.ndarray
    user_y: np.ndarray
    anchors_X: np.ndarray
    anchors_y: np.ndarray
    metrics: dict[str, dict[str, float]]
    anchor_reconstructions: dict[str, np.ndarray]


def _to_config(base_config: Any) -> Config:
    if isinstance(base_config, Config):
        return base_config
    if isinstance(base_config, dict):
        return Config(**base_config)
    return Config(**vars(base_config))


def _means(Z: np.ndarray, X: np.ndarray, center: bool) -> tuple[np.ndarray, np.ndarray]:
    if center:
        return Z.mean(axis=0, keepdims=True), X.mean(axis=0, keepdims=True)
    return np.zeros((1, Z.shape[1])), np.zeros((1, X.shape[1]))


def _reconstruct_pinv_centered(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    Z_mean, X_mean = _means(A_tilde, A, center)
    Zc = A_tilde - Z_mean
    Xc = A - X_mean
    W_c, *_ = np.linalg.lstsq(Zc, Xc, rcond=None)
    return (X_tilde_target - Z_mean) @ W_c + X_mean


def _reconstruct_mlp(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    seed: int,
    center: bool = True,
    hidden_layer_sizes: tuple[int, ...] = (128,),
    max_iter: int = 600,
) -> np.ndarray:
    Z = np.asarray(A_tilde, dtype=float)
    X = np.asarray(A, dtype=float)
    if center:
        Z_mean = Z.mean(axis=0, keepdims=True)
        X_mean = X.mean(axis=0, keepdims=True)
        Z_train = Z - Z_mean
        X_train = X - X_mean
        Z_target = X_tilde_target - Z_mean
    else:
        X_mean = np.zeros((1, X.shape[1]))
        Z_train = Z
        X_train = X
        Z_target = X_tilde_target
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit(Z_train, X_train)
    return mlp.predict(Z_target) + X_mean


def _reconstruct_pinv_not_orth(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    mean_Z, mean_A = _means(A_tilde, A, center)
    A_c = A - mean_A
    Z_c = A_tilde - mean_Z
    F_hat, *_ = np.linalg.lstsq(A_c, Z_c, rcond=None)
    F_hat_pinv = np.linalg.pinv(F_hat)
    mu_hat = (mean_A - mean_Z @ F_hat_pinv).ravel()
    return X_tilde_target @ F_hat_pinv + mu_hat


def _normalize_label_list(labels: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in labels:
        lab = int(item)
        if lab in seen:
            continue
        seen.add(lab)
        out.append(lab)
    if not out:
        raise ValueError("Label list must not be empty.")
    return out


def _normalize_method_keys(methods: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods:
        raw_key = str(method).strip()
        internal = METHOD_LABEL_TO_INTERNAL.get(raw_key)
        if internal is None:
            key = raw_key.lower()
            internal = METHOD_ALIAS_TO_INTERNAL.get(key)
        if internal is None:
            supported = "LinRegr (or pinv), MLP (or mlp), PINV (or pinv_not_orth)"
            raise ValueError(f"Unsupported method '{method}'. Supported: {supported}.")
        if internal in seen:
            continue
        seen.add(internal)
        out.append(internal)
    return tuple(out)


def _select_public_anchor_pool(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    anchor_labels: Sequence[int],
    public_per_label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_idx: list[np.ndarray] = []
    for label in anchor_labels:
        idx = np.where(y_pool == label)[0]
        if len(idx) < public_per_label:
            raise ValueError(
                f"Not enough public samples for label {label}: need {public_per_label}, have {len(idx)}."
            )
        selected_idx.append(rng.choice(idx, size=public_per_label, replace=False))
    idx_all = np.concatenate(selected_idx)
    return idx_all, X_pool[idx_all], y_pool[idx_all]


def _select_user_indices_with_label_minimum(
    y: np.ndarray,
    *,
    n_user: int,
    labels: Sequence[int],
    min_per_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if min_per_label <= 0:
        raise ValueError("min_per_label must be positive.")
    required_total = int(min_per_label * len(labels))
    if n_user < required_total:
        raise ValueError(
            f"num_institution_user is too small: need at least {required_total} "
            f"for {min_per_label} samples per label across {len(labels)} labels, have {n_user}."
        )

    selected_parts: list[np.ndarray] = []
    selected_mask = np.zeros(len(y), dtype=bool)
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < min_per_label:
            raise ValueError(
                f"Not enough total samples for label {label}: need {min_per_label}, have {len(idx)}."
            )
        picked = rng.choice(idx, size=min_per_label, replace=False)
        selected_parts.append(picked)
        selected_mask[picked] = True

    n_fill = n_user - required_total
    if n_fill > 0:
        remaining = np.where(~selected_mask)[0]
        if len(remaining) < n_fill:
            raise ValueError(
                f"Insufficient rows to build user split: need {n_fill} more, have {len(remaining)}."
            )
        selected_parts.append(rng.choice(remaining, size=n_fill, replace=False))

    idx_user = np.concatenate(selected_parts)
    rng.shuffle(idx_user)
    return idx_user


def _select_indices_per_label(
    y: np.ndarray,
    labels: Sequence[int],
    *,
    base_seed: int,
    image_seed_by_label: Optional[Mapping[int, int]],
    samples_per_label: int = 1,
) -> dict[int, list[int]]:
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive.")
    selected: dict[int, list[int]] = {}
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < samples_per_label:
            raise ValueError(
                f"Not enough user samples for label {label}: need {samples_per_label}, have {len(idx)}."
            )
        seed_value = int(image_seed_by_label[label]) if image_seed_by_label and label in image_seed_by_label else int(
            base_seed + label * 1009
        )
        label_rng = np.random.default_rng(seed_value)
        picked = label_rng.choice(idx, size=samples_per_label, replace=False)
        selected[label] = [int(i) for i in np.atleast_1d(picked)]
    return selected


def _infer_image_shape(n_features: int) -> Optional[tuple[int, int]]:
    side = int(round(np.sqrt(n_features)))
    if side * side == n_features:
        return (side, side)
    return None


def _build_selected_image_dict(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out[int(label)] = {"original": np.asarray(orig), "reconstructed": np.asarray(rec)}
    return out


def _build_selected_image_dict_multi(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, list[dict[str, np.ndarray]]]:
    out: dict[int, list[dict[str, np.ndarray]]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out.setdefault(int(label), []).append({"original": np.asarray(orig), "reconstructed": np.asarray(rec)})
    return out


def run_light_anchor_reconstruction(
    base_config: Any,
    *,
    anchor_labels: Sequence[int] = (0, 1, 2),
    display_labels: Optional[Sequence[int]] = None,
    public_per_label: int = 10,
    num_anchor_data: int = 300,
    methods: Sequence[str] = DEFAULT_METHODS,
    image_seed_by_label: Optional[Mapping[int, int]] = None,
    evaluate_classifier_accuracy: bool = False,
    evaluation_classifier: Optional[Any] = None,
    rf_train_size: int = 5000,
    reconstruct_anchor_images: bool = False,
    recon_centering: bool = True,
    smote_cross_label_ratio: float = 0.0,
    smote_ratio: float = 1.0,
    mlp_hidden_layer_sizes: tuple[int, ...] = (128,),
    mlp_max_iter: int = 600,
    samples_per_label: int = 1,
) -> LightReconstructionResult:
    """
    Lightweight reconstruction utility for notebooks.

    - Only one user image is reconstructed per display label.
    - Anchors are generated by SMOTE-style oversampling from a small public pool.
    - Optional classifier accuracy is computed by a provided classifier, or by
      a RandomForest fitted on a disjoint dataset split when no classifier is
      provided.
    """
    cfg = _to_config(base_config)
    seed = int(getattr(cfg, "seed", 0) or 0)
    rng = np.random.default_rng(seed)
    if getattr(cfg, "gamma_ratio", None) is None:
        cfg.gamma_ratio = 1.0

    labels_anchor = _normalize_label_list(anchor_labels)
    labels_display = _normalize_label_list(display_labels if display_labels is not None else labels_anchor)

    method_keys = _normalize_method_keys(methods)

    y_name = getattr(cfg, "y_name", None) or "target"
    df = load_data(cfg)
    if y_name not in df.columns:
        raise ValueError(f"Label column '{y_name}' is not present in loaded dataframe.")

    n_user = int(getattr(cfg, "num_institution_user", 100) or 100)
    if n_user <= 0:
        raise ValueError("num_institution_user must be positive.")

    if num_anchor_data <= 0:
        raise ValueError("num_anchor_data must be positive.")
    if public_per_label <= 0:
        raise ValueError("public_per_label must be positive.")

    arr_y = df[y_name].to_numpy()
    arr_X = df.drop(columns=[y_name]).to_numpy(dtype=np.float32)
    n_total = arr_X.shape[0]
    public_total = int(public_per_label * len(labels_anchor))

    if n_total < n_user + public_total:
        raise ValueError(
            f"Insufficient rows: need at least {n_user + public_total} (user + public), have {n_total}."
        )
    idx_user = _select_user_indices_with_label_minimum(
        arr_y,
        n_user=n_user,
        labels=labels_display,
        min_per_label=samples_per_label,
        rng=rng,
    )
    idx_remaining = np.setdiff1d(np.arange(n_total), idx_user, assume_unique=False)
    idx_remaining = rng.permutation(idx_remaining)

    X_user = arr_X[idx_user]
    y_user = arr_y[idx_user]
    X_rem = arr_X[idx_remaining]
    y_rem = arr_y[idx_remaining]

    idx_public_local, public_X, public_y = _select_public_anchor_pool(
        X_rem,
        y_rem,
        anchor_labels=labels_anchor,
        public_per_label=public_per_label,
        rng=rng,
    )

    # Build anchors from public pool using the existing SMOTE-style utility.
    smote_cfg = Config(**vars(cfg))
    smote_cfg.anchor_method = "smote"
    smote_cfg.smote_ratio = float(smote_ratio)
    smote_cfg.smote_cross_label_ratio = float(smote_cross_label_ratio)
    anchor_X, anchor_y = anchor_utils.produce_anchor(
        num_row=int(num_anchor_data),
        num_col=X_user.shape[1],
        seed=seed,
        config=smote_cfg,
        train_df=df.iloc[:0],
        Xs_train=[],
        Xs_test=[],
        ys_train=[],
        ys_test=[],
        smote_X=public_X,
        smote_y=public_y,
        return_labels=True,
        include_public_anchor=True,
    )

    projector = dimensionality_reduction.build_dimensionality_projector(
        X_user,
        n_components=int(getattr(cfg, "dim_intermediate", 10) or 10),
        F_type=str(getattr(cfg, "F_type", "svd") or "svd"),
        seed=seed,
        config=cfg,
        y=y_user,
    )
    X_user_tilde = projector(X_user)
    anchor_tilde = projector(anchor_X)

    selected_indices_by_label = _select_indices_per_label(
        y_user,
        labels_display,
        base_seed=seed,
        image_seed_by_label=image_seed_by_label,
        samples_per_label=samples_per_label,
    )
    selected_label_sequence: list[int] = []
    selected_rows_list: list[int] = []
    for label in labels_display:
        rows = selected_indices_by_label[label]
        selected_rows_list.extend(rows)
        selected_label_sequence.extend([int(label)] * len(rows))
    selected_rows = np.array(selected_rows_list, dtype=int)
    selected_originals = X_user[selected_rows]
    selected_tilde = X_user_tilde[selected_rows]

    selected_recons: dict[str, np.ndarray] = {}
    anchor_recons: dict[str, np.ndarray] = {}
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        if method == "pinv":
            rec_selected = _reconstruct_pinv_centered(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_centered(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        elif method == "mlp":
            rec_selected = _reconstruct_mlp(
                anchor_tilde,
                anchor_X,
                selected_tilde,
                seed=seed,
                center=recon_centering,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=mlp_max_iter,
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_mlp(
                    anchor_tilde,
                    anchor_X,
                    anchor_tilde,
                    seed=seed,
                    center=recon_centering,
                    hidden_layer_sizes=mlp_hidden_layer_sizes,
                    max_iter=mlp_max_iter,
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        else:
            rec_selected = _reconstruct_pinv_not_orth(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_not_orth(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        selected_recons[method] = rec_selected
        selected_recons[method_label] = rec_selected

    image_shape = _infer_image_shape(selected_originals.shape[1])
    selected_images_multi = {
        method: _build_selected_image_dict_multi(
            selected_label_sequence,
            selected_originals,
            selected_recons[method],
            image_shape=image_shape,
        )
        for method in method_keys
    }
    selected_images = {
        method: {int(label): selected_images_multi[method][int(label)][0] for label in labels_display}
        for method in method_keys
    }
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        selected_images_multi[method_label] = selected_images_multi[method]
        selected_images[method_label] = selected_images[method]

    metrics: dict[str, dict[str, float]] = {}
    if evaluate_classifier_accuracy:
        if evaluation_classifier is None:
            idx_public_global = idx_remaining[idx_public_local]
            all_remaining_idx = np.setdiff1d(
                np.arange(n_total),
                np.concatenate([idx_user, idx_public_global]),
                assume_unique=False,
            )
            if len(all_remaining_idx) < 10:
                raise ValueError("Not enough independent samples for RF training.")
            rf_idx = all_remaining_idx[: min(int(rf_train_size), len(all_remaining_idx))]
            X_rf = arr_X[rf_idx]
            y_rf = arr_y[rf_idx]
            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_rf, y_rf)
        else:
            clf = evaluation_classifier
        y_true_selected = np.array(selected_label_sequence, dtype=y_user.dtype)
        for method in method_keys:
            y_pred = clf.predict(selected_recons[method])
            metric_val = {"classifier_accuracy": float(np.mean(y_pred == y_true_selected))}
            metrics[method] = metric_val
            metrics[METHOD_INTERNAL_TO_LABEL[method]] = metric_val

    return LightReconstructionResult(
        config=cfg,
        methods=tuple(METHOD_INTERNAL_TO_LABEL[m] for m in method_keys),
        selected_labels=list(labels_display),
        selected_indices={k: v[0] for k, v in selected_indices_by_label.items()},
        selected_indices_by_label=selected_indices_by_label,
        selected_label_sequence=selected_label_sequence,
        selected_originals=selected_originals,
        selected_reconstructions=selected_recons,
        selected_images=selected_images,
        selected_images_multi=selected_images_multi,
        user_X=X_user,
        user_y=y_user,
        anchors_X=anchor_X,
        anchors_y=anchor_y,
        metrics=metrics,
        anchor_reconstructions=anchor_recons,
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from config.config import Config
from src import dimensionality_reduction
from src.institution_data_pipeline.load_data import load_data
from src.intermediate_expression import anchor_utils


METHOD_ALIAS_TO_INTERNAL: dict[str, str] = {
    "linregr": "pinv",
    "linreg": "pinv",
    "pinv": "pinv",
    "mlp": "mlp",
    "pinv_not_orth": "pinv_not_orth",
    "pinv+orth": "pinv_not_orth",
    "pinv_notorth": "pinv_not_orth",
    "pinv-not-orth": "pinv_not_orth",
    "pinv-not_orth": "pinv_not_orth",
}
METHOD_INTERNAL_TO_LABEL: dict[str, str] = {
    "pinv": "LinRegr",
    "mlp": "MLP",
    "pinv_not_orth": "PINV",
}
METHOD_LABEL_TO_INTERNAL: dict[str, str] = {
    "LinRegr": "pinv",
    "MLP": "mlp",
    "PINV": "pinv_not_orth",
}
DEFAULT_METHODS: tuple[str, ...] = ("LinRegr", "MLP", "PINV")


@dataclass
class LightReconstructionResult:
    config: Config
    methods: tuple[str, ...]
    selected_labels: list[int]
    selected_indices: dict[int, int]
    selected_indices_by_label: dict[int, list[int]]
    selected_label_sequence: list[int]
    selected_originals: np.ndarray
    selected_reconstructions: dict[str, np.ndarray]
    selected_images: dict[str, dict[int, dict[str, np.ndarray]]]
    selected_images_multi: dict[str, dict[int, list[dict[str, np.ndarray]]]]
    user_X: np.ndarray
    user_y: np.ndarray
    anchors_X: np.ndarray
    anchors_y: np.ndarray
    metrics: dict[str, dict[str, float]]
    anchor_reconstructions: dict[str, np.ndarray]


def _to_config(base_config: Any) -> Config:
    if isinstance(base_config, Config):
        return base_config
    if isinstance(base_config, dict):
        return Config(**base_config)
    return Config(**vars(base_config))


def _means(Z: np.ndarray, X: np.ndarray, center: bool) -> tuple[np.ndarray, np.ndarray]:
    if center:
        return Z.mean(axis=0, keepdims=True), X.mean(axis=0, keepdims=True)
    return np.zeros((1, Z.shape[1])), np.zeros((1, X.shape[1]))


def _reconstruct_pinv_centered(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    Z_mean, X_mean = _means(A_tilde, A, center)
    Zc = A_tilde - Z_mean
    Xc = A - X_mean
    W_c, *_ = np.linalg.lstsq(Zc, Xc, rcond=None)
    return (X_tilde_target - Z_mean) @ W_c + X_mean


def _reconstruct_mlp(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    seed: int,
    center: bool = True,
    hidden_layer_sizes: tuple[int, ...] = (128,),
    max_iter: int = 600,
) -> np.ndarray:
    Z = np.asarray(A_tilde, dtype=float)
    X = np.asarray(A, dtype=float)
    if center:
        Z_mean = Z.mean(axis=0, keepdims=True)
        X_mean = X.mean(axis=0, keepdims=True)
        Z_train = Z - Z_mean
        X_train = X - X_mean
        Z_target = X_tilde_target - Z_mean
    else:
        X_mean = np.zeros((1, X.shape[1]))
        Z_train = Z
        X_train = X
        Z_target = X_tilde_target
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit(Z_train, X_train)
    return mlp.predict(Z_target) + X_mean


def _reconstruct_pinv_not_orth(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    mean_Z, mean_A = _means(A_tilde, A, center)
    A_c = A - mean_A
    Z_c = A_tilde - mean_Z
    F_hat, *_ = np.linalg.lstsq(A_c, Z_c, rcond=None)
    F_hat_pinv = np.linalg.pinv(F_hat)
    mu_hat = (mean_A - mean_Z @ F_hat_pinv).ravel()
    return X_tilde_target @ F_hat_pinv + mu_hat


def _normalize_label_list(labels: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in labels:
        lab = int(item)
        if lab in seen:
            continue
        seen.add(lab)
        out.append(lab)
    if not out:
        raise ValueError("Label list must not be empty.")
    return out


def _normalize_method_keys(methods: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods:
        raw_key = str(method).strip()
        internal = METHOD_LABEL_TO_INTERNAL.get(raw_key)
        if internal is None:
            key = raw_key.lower()
            internal = METHOD_ALIAS_TO_INTERNAL.get(key)
        if internal is None:
            supported = "LinRegr (or pinv), MLP (or mlp), PINV (or pinv_not_orth)"
            raise ValueError(f"Unsupported method '{method}'. Supported: {supported}.")
        if internal in seen:
            continue
        seen.add(internal)
        out.append(internal)
    return tuple(out)


def _select_public_anchor_pool(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    anchor_labels: Sequence[int],
    public_per_label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_idx: list[np.ndarray] = []
    for label in anchor_labels:
        idx = np.where(y_pool == label)[0]
        if len(idx) < public_per_label:
            raise ValueError(
                f"Not enough public samples for label {label}: need {public_per_label}, have {len(idx)}."
            )
        selected_idx.append(rng.choice(idx, size=public_per_label, replace=False))
    idx_all = np.concatenate(selected_idx)
    return idx_all, X_pool[idx_all], y_pool[idx_all]


def _select_user_indices_with_label_minimum(
    y: np.ndarray,
    *,
    n_user: int,
    labels: Sequence[int],
    min_per_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if min_per_label <= 0:
        raise ValueError("min_per_label must be positive.")
    required_total = int(min_per_label * len(labels))
    if n_user < required_total:
        raise ValueError(
            f"num_institution_user is too small: need at least {required_total} "
            f"for {min_per_label} samples per label across {len(labels)} labels, have {n_user}."
        )

    selected_parts: list[np.ndarray] = []
    selected_mask = np.zeros(len(y), dtype=bool)
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < min_per_label:
            raise ValueError(
                f"Not enough total samples for label {label}: need {min_per_label}, have {len(idx)}."
            )
        picked = rng.choice(idx, size=min_per_label, replace=False)
        selected_parts.append(picked)
        selected_mask[picked] = True

    n_fill = n_user - required_total
    if n_fill > 0:
        remaining = np.where(~selected_mask)[0]
        if len(remaining) < n_fill:
            raise ValueError(
                f"Insufficient rows to build user split: need {n_fill} more, have {len(remaining)}."
            )
        selected_parts.append(rng.choice(remaining, size=n_fill, replace=False))

    idx_user = np.concatenate(selected_parts)
    rng.shuffle(idx_user)
    return idx_user


def _select_indices_per_label(
    y: np.ndarray,
    labels: Sequence[int],
    *,
    base_seed: int,
    image_seed_by_label: Optional[Mapping[int, int]],
    samples_per_label: int = 1,
) -> dict[int, list[int]]:
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive.")
    selected: dict[int, list[int]] = {}
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < samples_per_label:
            raise ValueError(
                f"Not enough user samples for label {label}: need {samples_per_label}, have {len(idx)}."
            )
        seed_value = int(image_seed_by_label[label]) if image_seed_by_label and label in image_seed_by_label else int(
            base_seed + label * 1009
        )
        label_rng = np.random.default_rng(seed_value)
        picked = label_rng.choice(idx, size=samples_per_label, replace=False)
        selected[label] = [int(i) for i in np.atleast_1d(picked)]
    return selected


def _infer_image_shape(n_features: int) -> Optional[tuple[int, int]]:
    side = int(round(np.sqrt(n_features)))
    if side * side == n_features:
        return (side, side)
    return None


def _build_selected_image_dict(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out[int(label)] = {"original": np.asarray(orig), "reconstructed": np.asarray(rec)}
    return out


def _build_selected_image_dict_multi(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, list[dict[str, np.ndarray]]]:
    out: dict[int, list[dict[str, np.ndarray]]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out.setdefault(int(label), []).append({"original": np.asarray(orig), "reconstructed": np.asarray(rec)})
    return out


def run_light_anchor_reconstruction(
    base_config: Any,
    *,
    anchor_labels: Sequence[int] = (0, 1, 2),
    display_labels: Optional[Sequence[int]] = None,
    public_per_label: int = 10,
    num_anchor_data: int = 300,
    methods: Sequence[str] = DEFAULT_METHODS,
    image_seed_by_label: Optional[Mapping[int, int]] = None,
    evaluate_classifier_accuracy: bool = False,
    evaluation_classifier: Optional[Any] = None,
    rf_train_size: int = 5000,
    reconstruct_anchor_images: bool = False,
    recon_centering: bool = True,
    smote_cross_label_ratio: float = 0.0,
    smote_ratio: float = 1.0,
    mlp_hidden_layer_sizes: tuple[int, ...] = (128,),
    mlp_max_iter: int = 600,
    samples_per_label: int = 1,
) -> LightReconstructionResult:
    """
    Lightweight reconstruction utility for notebooks.

    - Only one user image is reconstructed per display label.
    - Anchors are generated by SMOTE-style oversampling from a small public pool.
    - Optional classifier accuracy is computed by a provided classifier, or by
      a RandomForest fitted on a disjoint dataset split when no classifier is
      provided.
    """
    cfg = _to_config(base_config)
    seed = int(getattr(cfg, "seed", 0) or 0)
    rng = np.random.default_rng(seed)
    if getattr(cfg, "gamma_ratio", None) is None:
        cfg.gamma_ratio = 1.0

    labels_anchor = _normalize_label_list(anchor_labels)
    labels_display = _normalize_label_list(display_labels if display_labels is not None else labels_anchor)

    method_keys = _normalize_method_keys(methods)

    y_name = getattr(cfg, "y_name", None) or "target"
    df = load_data(cfg)
    if y_name not in df.columns:
        raise ValueError(f"Label column '{y_name}' is not present in loaded dataframe.")

    n_user = int(getattr(cfg, "num_institution_user", 100) or 100)
    if n_user <= 0:
        raise ValueError("num_institution_user must be positive.")

    if num_anchor_data <= 0:
        raise ValueError("num_anchor_data must be positive.")
    if public_per_label <= 0:
        raise ValueError("public_per_label must be positive.")

    arr_y = df[y_name].to_numpy()
    arr_X = df.drop(columns=[y_name]).to_numpy(dtype=np.float32)
    n_total = arr_X.shape[0]
    public_total = int(public_per_label * len(labels_anchor))

    if n_total < n_user + public_total:
        raise ValueError(
            f"Insufficient rows: need at least {n_user + public_total} (user + public), have {n_total}."
        )
    idx_user = _select_user_indices_with_label_minimum(
        arr_y,
        n_user=n_user,
        labels=labels_display,
        min_per_label=samples_per_label,
        rng=rng,
    )
    idx_remaining = np.setdiff1d(np.arange(n_total), idx_user, assume_unique=False)
    idx_remaining = rng.permutation(idx_remaining)

    X_user = arr_X[idx_user]
    y_user = arr_y[idx_user]
    X_rem = arr_X[idx_remaining]
    y_rem = arr_y[idx_remaining]

    idx_public_local, public_X, public_y = _select_public_anchor_pool(
        X_rem,
        y_rem,
        anchor_labels=labels_anchor,
        public_per_label=public_per_label,
        rng=rng,
    )

    # Build anchors from public pool using the existing SMOTE-style utility.
    smote_cfg = Config(**vars(cfg))
    smote_cfg.anchor_method = "smote"
    smote_cfg.smote_ratio = float(smote_ratio)
    smote_cfg.smote_cross_label_ratio = float(smote_cross_label_ratio)
    anchor_X, anchor_y = anchor_utils.produce_anchor(
        num_row=int(num_anchor_data),
        num_col=X_user.shape[1],
        seed=seed,
        config=smote_cfg,
        train_df=df.iloc[:0],
        Xs_train=[],
        Xs_test=[],
        ys_train=[],
        ys_test=[],
        smote_X=public_X,
        smote_y=public_y,
        return_labels=True,
        include_public_anchor=True,
    )

    projector = dimensionality_reduction.build_dimensionality_projector(
        X_user,
        n_components=int(getattr(cfg, "dim_intermediate", 10) or 10),
        F_type=str(getattr(cfg, "F_type", "svd") or "svd"),
        seed=seed,
        config=cfg,
        y=y_user,
    )
    X_user_tilde = projector(X_user)
    anchor_tilde = projector(anchor_X)

    selected_indices_by_label = _select_indices_per_label(
        y_user,
        labels_display,
        base_seed=seed,
        image_seed_by_label=image_seed_by_label,
        samples_per_label=samples_per_label,
    )
    selected_label_sequence: list[int] = []
    selected_rows_list: list[int] = []
    for label in labels_display:
        rows = selected_indices_by_label[label]
        selected_rows_list.extend(rows)
        selected_label_sequence.extend([int(label)] * len(rows))
    selected_rows = np.array(selected_rows_list, dtype=int)
    selected_originals = X_user[selected_rows]
    selected_tilde = X_user_tilde[selected_rows]

    selected_recons: dict[str, np.ndarray] = {}
    anchor_recons: dict[str, np.ndarray] = {}
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        if method == "pinv":
            rec_selected = _reconstruct_pinv_centered(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_centered(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        elif method == "mlp":
            rec_selected = _reconstruct_mlp(
                anchor_tilde,
                anchor_X,
                selected_tilde,
                seed=seed,
                center=recon_centering,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=mlp_max_iter,
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_mlp(
                    anchor_tilde,
                    anchor_X,
                    anchor_tilde,
                    seed=seed,
                    center=recon_centering,
                    hidden_layer_sizes=mlp_hidden_layer_sizes,
                    max_iter=mlp_max_iter,
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        else:
            rec_selected = _reconstruct_pinv_not_orth(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_not_orth(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        selected_recons[method] = rec_selected
        selected_recons[method_label] = rec_selected

    image_shape = _infer_image_shape(selected_originals.shape[1])
    selected_images_multi = {
        method: _build_selected_image_dict_multi(
            selected_label_sequence,
            selected_originals,
            selected_recons[method],
            image_shape=image_shape,
        )
        for method in method_keys
    }
    selected_images = {
        method: {int(label): selected_images_multi[method][int(label)][0] for label in labels_display}
        for method in method_keys
    }
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        selected_images_multi[method_label] = selected_images_multi[method]
        selected_images[method_label] = selected_images[method]

    metrics: dict[str, dict[str, float]] = {}
    if evaluate_classifier_accuracy:
        if evaluation_classifier is None:
            idx_public_global = idx_remaining[idx_public_local]
            all_remaining_idx = np.setdiff1d(
                np.arange(n_total),
                np.concatenate([idx_user, idx_public_global]),
                assume_unique=False,
            )
            if len(all_remaining_idx) < 10:
                raise ValueError("Not enough independent samples for RF training.")
            rf_idx = all_remaining_idx[: min(int(rf_train_size), len(all_remaining_idx))]
            X_rf = arr_X[rf_idx]
            y_rf = arr_y[rf_idx]
            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_rf, y_rf)
        else:
            clf = evaluation_classifier
        y_true_selected = np.array(selected_label_sequence, dtype=y_user.dtype)
        for method in method_keys:
            y_pred = clf.predict(selected_recons[method])
            metric_val = {"classifier_accuracy": float(np.mean(y_pred == y_true_selected))}
            metrics[method] = metric_val
            metrics[METHOD_INTERNAL_TO_LABEL[method]] = metric_val

    return LightReconstructionResult(
        config=cfg,
        methods=tuple(METHOD_INTERNAL_TO_LABEL[m] for m in method_keys),
        selected_labels=list(labels_display),
        selected_indices={k: v[0] for k, v in selected_indices_by_label.items()},
        selected_indices_by_label=selected_indices_by_label,
        selected_label_sequence=selected_label_sequence,
        selected_originals=selected_originals,
        selected_reconstructions=selected_recons,
        selected_images=selected_images,
        selected_images_multi=selected_images_multi,
        user_X=X_user,
        user_y=y_user,
        anchors_X=anchor_X,
        anchors_y=anchor_y,
        metrics=metrics,
        anchor_reconstructions=anchor_recons,
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from config.config import Config
from src import dimensionality_reduction
from src.institution_data_pipeline.load_data import load_data
from src.intermediate_expression import anchor_utils


METHOD_ALIAS_TO_INTERNAL: dict[str, str] = {
    "linregr": "pinv",
    "linreg": "pinv",
    "pinv": "pinv",
    "mlp": "mlp",
    "pinv_not_orth": "pinv_not_orth",
    "pinv+orth": "pinv_not_orth",
    "pinv_notorth": "pinv_not_orth",
    "pinv-not-orth": "pinv_not_orth",
    "pinv-not_orth": "pinv_not_orth",
}
METHOD_INTERNAL_TO_LABEL: dict[str, str] = {
    "pinv": "LinRegr",
    "mlp": "MLP",
    "pinv_not_orth": "PINV",
}
METHOD_LABEL_TO_INTERNAL: dict[str, str] = {
    "LinRegr": "pinv",
    "MLP": "mlp",
    "PINV": "pinv_not_orth",
}
DEFAULT_METHODS: tuple[str, ...] = ("LinRegr", "MLP", "PINV")


@dataclass
class LightReconstructionResult:
    config: Config
    methods: tuple[str, ...]
    selected_labels: list[int]
    selected_indices: dict[int, int]
    selected_indices_by_label: dict[int, list[int]]
    selected_label_sequence: list[int]
    selected_originals: np.ndarray
    selected_reconstructions: dict[str, np.ndarray]
    selected_images: dict[str, dict[int, dict[str, np.ndarray]]]
    selected_images_multi: dict[str, dict[int, list[dict[str, np.ndarray]]]]
    user_X: np.ndarray
    user_y: np.ndarray
    anchors_X: np.ndarray
    anchors_y: np.ndarray
    metrics: dict[str, dict[str, float]]
    anchor_reconstructions: dict[str, np.ndarray]


def _to_config(base_config: Any) -> Config:
    if isinstance(base_config, Config):
        return base_config
    if isinstance(base_config, dict):
        return Config(**base_config)
    return Config(**vars(base_config))


def _means(Z: np.ndarray, X: np.ndarray, center: bool) -> tuple[np.ndarray, np.ndarray]:
    if center:
        return Z.mean(axis=0, keepdims=True), X.mean(axis=0, keepdims=True)
    return np.zeros((1, Z.shape[1])), np.zeros((1, X.shape[1]))


def _reconstruct_pinv_centered(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    Z_mean, X_mean = _means(A_tilde, A, center)
    Zc = A_tilde - Z_mean
    Xc = A - X_mean
    W_c, *_ = np.linalg.lstsq(Zc, Xc, rcond=None)
    return (X_tilde_target - Z_mean) @ W_c + X_mean


def _reconstruct_mlp(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    seed: int,
    center: bool = True,
    hidden_layer_sizes: tuple[int, ...] = (128,),
    max_iter: int = 600,
) -> np.ndarray:
    Z = np.asarray(A_tilde, dtype=float)
    X = np.asarray(A, dtype=float)
    if center:
        Z_mean = Z.mean(axis=0, keepdims=True)
        X_mean = X.mean(axis=0, keepdims=True)
        Z_train = Z - Z_mean
        X_train = X - X_mean
        Z_target = X_tilde_target - Z_mean
    else:
        X_mean = np.zeros((1, X.shape[1]))
        Z_train = Z
        X_train = X
        Z_target = X_tilde_target
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit(Z_train, X_train)
    return mlp.predict(Z_target) + X_mean


def _reconstruct_pinv_not_orth(
    A_tilde: np.ndarray,
    A: np.ndarray,
    X_tilde_target: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    mean_Z, mean_A = _means(A_tilde, A, center)
    A_c = A - mean_A
    Z_c = A_tilde - mean_Z
    F_hat, *_ = np.linalg.lstsq(A_c, Z_c, rcond=None)
    F_hat_pinv = np.linalg.pinv(F_hat)
    mu_hat = (mean_A - mean_Z @ F_hat_pinv).ravel()
    return X_tilde_target @ F_hat_pinv + mu_hat


def _normalize_label_list(labels: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in labels:
        lab = int(item)
        if lab in seen:
            continue
        seen.add(lab)
        out.append(lab)
    if not out:
        raise ValueError("Label list must not be empty.")
    return out


def _normalize_method_keys(methods: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for method in methods:
        raw_key = str(method).strip()
        internal = METHOD_LABEL_TO_INTERNAL.get(raw_key)
        if internal is None:
            key = raw_key.lower()
            internal = METHOD_ALIAS_TO_INTERNAL.get(key)
        if internal is None:
            supported = "LinRegr (or pinv), MLP (or mlp), PINV (or pinv_not_orth)"
            raise ValueError(f"Unsupported method '{method}'. Supported: {supported}.")
        if internal in seen:
            continue
        seen.add(internal)
        out.append(internal)
    return tuple(out)


def _select_public_anchor_pool(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    anchor_labels: Sequence[int],
    public_per_label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_idx: list[np.ndarray] = []
    for label in anchor_labels:
        idx = np.where(y_pool == label)[0]
        if len(idx) < public_per_label:
            raise ValueError(
                f"Not enough public samples for label {label}: need {public_per_label}, have {len(idx)}."
            )
        selected_idx.append(rng.choice(idx, size=public_per_label, replace=False))
    idx_all = np.concatenate(selected_idx)
    return idx_all, X_pool[idx_all], y_pool[idx_all]


def _select_user_indices_with_label_minimum(
    y: np.ndarray,
    *,
    n_user: int,
    labels: Sequence[int],
    min_per_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if min_per_label <= 0:
        raise ValueError("min_per_label must be positive.")
    required_total = int(min_per_label * len(labels))
    if n_user < required_total:
        raise ValueError(
            f"num_institution_user is too small: need at least {required_total} "
            f"for {min_per_label} samples per label across {len(labels)} labels, have {n_user}."
        )

    selected_parts: list[np.ndarray] = []
    selected_mask = np.zeros(len(y), dtype=bool)
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < min_per_label:
            raise ValueError(
                f"Not enough total samples for label {label}: need {min_per_label}, have {len(idx)}."
            )
        picked = rng.choice(idx, size=min_per_label, replace=False)
        selected_parts.append(picked)
        selected_mask[picked] = True

    n_fill = n_user - required_total
    if n_fill > 0:
        remaining = np.where(~selected_mask)[0]
        if len(remaining) < n_fill:
            raise ValueError(
                f"Insufficient rows to build user split: need {n_fill} more, have {len(remaining)}."
            )
        selected_parts.append(rng.choice(remaining, size=n_fill, replace=False))

    idx_user = np.concatenate(selected_parts)
    rng.shuffle(idx_user)
    return idx_user


def _select_indices_per_label(
    y: np.ndarray,
    labels: Sequence[int],
    *,
    base_seed: int,
    image_seed_by_label: Optional[Mapping[int, int]],
    samples_per_label: int = 1,
) -> dict[int, list[int]]:
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive.")
    selected: dict[int, list[int]] = {}
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) < samples_per_label:
            raise ValueError(
                f"Not enough user samples for label {label}: need {samples_per_label}, have {len(idx)}."
            )
        seed_value = int(image_seed_by_label[label]) if image_seed_by_label and label in image_seed_by_label else int(
            base_seed + label * 1009
        )
        label_rng = np.random.default_rng(seed_value)
        picked = label_rng.choice(idx, size=samples_per_label, replace=False)
        selected[label] = [int(i) for i in np.atleast_1d(picked)]
    return selected


def _infer_image_shape(n_features: int) -> Optional[tuple[int, int]]:
    side = int(round(np.sqrt(n_features)))
    if side * side == n_features:
        return (side, side)
    return None


def _build_selected_image_dict(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out[int(label)] = {"original": np.asarray(orig), "reconstructed": np.asarray(rec)}
    return out


def _build_selected_image_dict_multi(
    selected_labels: Sequence[int],
    selected_originals: np.ndarray,
    selected_recons: np.ndarray,
    *,
    image_shape: Optional[tuple[int, int]],
) -> dict[int, list[dict[str, np.ndarray]]]:
    out: dict[int, list[dict[str, np.ndarray]]] = {}
    for i, label in enumerate(selected_labels):
        orig = selected_originals[i]
        rec = selected_recons[i]
        if image_shape is not None:
            orig = orig.reshape(image_shape)
            rec = rec.reshape(image_shape)
        out.setdefault(int(label), []).append({"original": np.asarray(orig), "reconstructed": np.asarray(rec)})
    return out


def run_light_anchor_reconstruction(
    base_config: Any,
    *,
    anchor_labels: Sequence[int] = (0, 1, 2),
    display_labels: Optional[Sequence[int]] = None,
    public_per_label: int = 10,
    num_anchor_data: int = 300,
    methods: Sequence[str] = DEFAULT_METHODS,
    image_seed_by_label: Optional[Mapping[int, int]] = None,
    evaluate_classifier_accuracy: bool = False,
    evaluation_classifier: Optional[Any] = None,
    rf_train_size: int = 5000,
    reconstruct_anchor_images: bool = False,
    recon_centering: bool = True,
    smote_cross_label_ratio: float = 0.0,
    smote_ratio: float = 1.0,
    mlp_hidden_layer_sizes: tuple[int, ...] = (128,),
    mlp_max_iter: int = 600,
    samples_per_label: int = 1,
) -> LightReconstructionResult:
    """
    Lightweight reconstruction utility for notebooks.

    - Only one user image is reconstructed per display label.
    - Anchors are generated by SMOTE-style oversampling from a small public pool.
    - Optional classifier accuracy is computed by a provided classifier, or by
      a RandomForest fitted on a disjoint dataset split when no classifier is
      provided.
    """
    cfg = _to_config(base_config)
    seed = int(getattr(cfg, "seed", 0) or 0)
    rng = np.random.default_rng(seed)
    if getattr(cfg, "gamma_ratio", None) is None:
        cfg.gamma_ratio = 1.0

    labels_anchor = _normalize_label_list(anchor_labels)
    labels_display = _normalize_label_list(display_labels if display_labels is not None else labels_anchor)

    method_keys = _normalize_method_keys(methods)

    y_name = getattr(cfg, "y_name", None) or "target"
    df = load_data(cfg)
    if y_name not in df.columns:
        raise ValueError(f"Label column '{y_name}' is not present in loaded dataframe.")

    n_user = int(getattr(cfg, "num_institution_user", 100) or 100)
    if n_user <= 0:
        raise ValueError("num_institution_user must be positive.")

    if num_anchor_data <= 0:
        raise ValueError("num_anchor_data must be positive.")
    if public_per_label <= 0:
        raise ValueError("public_per_label must be positive.")

    arr_y = df[y_name].to_numpy()
    arr_X = df.drop(columns=[y_name]).to_numpy(dtype=np.float32)
    n_total = arr_X.shape[0]
    public_total = int(public_per_label * len(labels_anchor))

    if n_total < n_user + public_total:
        raise ValueError(
            f"Insufficient rows: need at least {n_user + public_total} (user + public), have {n_total}."
        )
    idx_user = _select_user_indices_with_label_minimum(
        arr_y,
        n_user=n_user,
        labels=labels_display,
        min_per_label=samples_per_label,
        rng=rng,
    )
    idx_remaining = np.setdiff1d(np.arange(n_total), idx_user, assume_unique=False)
    idx_remaining = rng.permutation(idx_remaining)

    X_user = arr_X[idx_user]
    y_user = arr_y[idx_user]
    X_rem = arr_X[idx_remaining]
    y_rem = arr_y[idx_remaining]

    idx_public_local, public_X, public_y = _select_public_anchor_pool(
        X_rem,
        y_rem,
        anchor_labels=labels_anchor,
        public_per_label=public_per_label,
        rng=rng,
    )

    # Build anchors from public pool using the existing SMOTE-style utility.
    smote_cfg = Config(**vars(cfg))
    smote_cfg.anchor_method = "smote"
    smote_cfg.smote_ratio = float(smote_ratio)
    smote_cfg.smote_cross_label_ratio = float(smote_cross_label_ratio)
    anchor_X, anchor_y = anchor_utils.produce_anchor(
        num_row=int(num_anchor_data),
        num_col=X_user.shape[1],
        seed=seed,
        config=smote_cfg,
        train_df=df.iloc[:0],
        Xs_train=[],
        Xs_test=[],
        ys_train=[],
        ys_test=[],
        smote_X=public_X,
        smote_y=public_y,
        return_labels=True,
        include_public_anchor=True,
    )

    projector = dimensionality_reduction.build_dimensionality_projector(
        X_user,
        n_components=int(getattr(cfg, "dim_intermediate", 10) or 10),
        F_type=str(getattr(cfg, "F_type", "svd") or "svd"),
        seed=seed,
        config=cfg,
        y=y_user,
    )
    X_user_tilde = projector(X_user)
    anchor_tilde = projector(anchor_X)

    selected_indices_by_label = _select_indices_per_label(
        y_user,
        labels_display,
        base_seed=seed,
        image_seed_by_label=image_seed_by_label,
        samples_per_label=samples_per_label,
    )
    selected_label_sequence: list[int] = []
    selected_rows_list: list[int] = []
    for label in labels_display:
        rows = selected_indices_by_label[label]
        selected_rows_list.extend(rows)
        selected_label_sequence.extend([int(label)] * len(rows))
    selected_rows = np.array(selected_rows_list, dtype=int)
    selected_originals = X_user[selected_rows]
    selected_tilde = X_user_tilde[selected_rows]

    selected_recons: dict[str, np.ndarray] = {}
    anchor_recons: dict[str, np.ndarray] = {}
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        if method == "pinv":
            rec_selected = _reconstruct_pinv_centered(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_centered(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        elif method == "mlp":
            rec_selected = _reconstruct_mlp(
                anchor_tilde,
                anchor_X,
                selected_tilde,
                seed=seed,
                center=recon_centering,
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                max_iter=mlp_max_iter,
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_mlp(
                    anchor_tilde,
                    anchor_X,
                    anchor_tilde,
                    seed=seed,
                    center=recon_centering,
                    hidden_layer_sizes=mlp_hidden_layer_sizes,
                    max_iter=mlp_max_iter,
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        else:
            rec_selected = _reconstruct_pinv_not_orth(
                anchor_tilde, anchor_X, selected_tilde, center=recon_centering
            )
            if reconstruct_anchor_images:
                rec_anchor = _reconstruct_pinv_not_orth(
                    anchor_tilde, anchor_X, anchor_tilde, center=recon_centering
                )
                anchor_recons[method] = rec_anchor
                anchor_recons[method_label] = rec_anchor
        selected_recons[method] = rec_selected
        selected_recons[method_label] = rec_selected

    image_shape = _infer_image_shape(selected_originals.shape[1])
    selected_images_multi = {
        method: _build_selected_image_dict_multi(
            selected_label_sequence,
            selected_originals,
            selected_recons[method],
            image_shape=image_shape,
        )
        for method in method_keys
    }
    selected_images = {
        method: {int(label): selected_images_multi[method][int(label)][0] for label in labels_display}
        for method in method_keys
    }
    for method in method_keys:
        method_label = METHOD_INTERNAL_TO_LABEL[method]
        selected_images_multi[method_label] = selected_images_multi[method]
        selected_images[method_label] = selected_images[method]

    metrics: dict[str, dict[str, float]] = {}
    if evaluate_classifier_accuracy:
        if evaluation_classifier is None:
            idx_public_global = idx_remaining[idx_public_local]
            all_remaining_idx = np.setdiff1d(
                np.arange(n_total),
                np.concatenate([idx_user, idx_public_global]),
                assume_unique=False,
            )
            if len(all_remaining_idx) < 10:
                raise ValueError("Not enough independent samples for RF training.")
            rf_idx = all_remaining_idx[: min(int(rf_train_size), len(all_remaining_idx))]
            X_rf = arr_X[rf_idx]
            y_rf = arr_y[rf_idx]
            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_rf, y_rf)
        else:
            clf = evaluation_classifier
        y_true_selected = np.array(selected_label_sequence, dtype=y_user.dtype)
        for method in method_keys:
            y_pred = clf.predict(selected_recons[method])
            metric_val = {"classifier_accuracy": float(np.mean(y_pred == y_true_selected))}
            metrics[method] = metric_val
            metrics[METHOD_INTERNAL_TO_LABEL[method]] = metric_val

    return LightReconstructionResult(
        config=cfg,
        methods=tuple(METHOD_INTERNAL_TO_LABEL[m] for m in method_keys),
        selected_labels=list(labels_display),
        selected_indices={k: v[0] for k, v in selected_indices_by_label.items()},
        selected_indices_by_label=selected_indices_by_label,
        selected_label_sequence=selected_label_sequence,
        selected_originals=selected_originals,
        selected_reconstructions=selected_recons,
        selected_images=selected_images,
        selected_images_multi=selected_images_multi,
        user_X=X_user,
        user_y=y_user,
        anchors_X=anchor_X,
        anchors_y=anchor_y,
        metrics=metrics,
        anchor_reconstructions=anchor_recons,
    )
