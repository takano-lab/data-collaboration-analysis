from __future__ import annotations

from typing import Dict, Optional, TypeVar
import copy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.federated_learning import run_federated_learning  # スクラッチ実装をインポート
from src.model import ModelRunner
from src.dimensionality_reduction import build_dimensionality_projector
from src.institution_data_pipeline.builders import InstitutionDatasetBuilder
from src.privacy import compute_membership_auc

logger = TypeVar("logger")


def _features_and_labels(df: pd.DataFrame, y_name: str) -> tuple[np.ndarray, np.ndarray]:
    if y_name not in df.columns:
        raise ValueError(f"target column '{y_name}' not found in DataFrame")
    y = df[y_name].to_numpy(copy=True)
    X = df.drop(columns=[y_name]).to_numpy(copy=True)
    return X, y


# ----------------------------------------------------------------------
# 集中解析
# ----------------------------------------------------------------------
def centralize_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    logger: logger,
    y_name: str,
) -> float:

    X_train, y_train = _features_and_labels(train_df, y_name)
    X_test, y_test = _features_and_labels(test_df, y_name)

    # SVD
    X_tr_svd, X_te_svd = X_train, X_test
    model_runner = ModelRunner(config)
    metrics = model_runner.run(
        X_train=X_tr_svd,
        y_train=y_train,
        X_test=X_te_svd,
        y_test=y_test,
    )

    logger.info(f"集中解析の評価値: {metrics:.4f}")

    return metrics

def centralize_analysis_with_dimension_reduction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    logger: logger,
    y_name: str,
) -> float:
    X_train, y_train = _features_and_labels(train_df, y_name)
    X_test, y_test = _features_and_labels(test_df, y_name)

    projector = build_dimensionality_projector(
        X=X_train,
        n_components=config.dim_integrate,
        y=y_train,
        F_type=getattr(config, "F_type", "svd"),
        seed=getattr(config, "f_seed", None),
        config=config,
    )
    X_tr_svd = projector(X_train)
    X_te_svd = projector(X_test)

    model_runner = ModelRunner(config)
    metrics = model_runner.run(
        X_train=X_tr_svd,
        y_train=y_train,
        X_test=X_te_svd,
        y_test=y_test,
    )
    logger.info(f"集中解析（次元削減）の評価値: {metrics:.4f}")
    return metrics


def centralize_analysis_with_institution_dimension_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_reduction: np.ndarray,
    config: Config,
    logger: logger,
) -> float:
    projector = build_dimensionality_projector(
        X=X_train_reduction,
        n_components=config.dim_intermediate,
        F_type=getattr(config, "F_type", "svd"),
        seed=getattr(config, "f_seed", None),
        config=config,
    )
    X_tr = projector(X_train)
    X_te = projector(X_test)

    model_runner = ModelRunner(config)
    metrics = model_runner.run(
        X_train=X_tr,
        y_train=y_train,
        X_test=X_te,
        y_test=y_test,
    )
    logger.info(f"提案手法の評価値: {metrics:.4f}")
    return metrics

# ----------------------------------------------------------------------
# 個別解析
# ----------------------------------------------------------------------
def individual_analysis(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> float:
    losses: list[float] = []

    model_runner = ModelRunner(config)

    for X_tr, X_te, y_tr, y_te in zip(Xs_train, Xs_test, ys_train, ys_test):
        X_tr_svd, X_te_svd = X_tr, X_te
        metrics = model_runner.run(X_tr_svd, y_tr, X_te_svd, y_te)
        losses.append(metrics)

    mean_score = float(np.mean(losses)) if losses else np.nan
    logger.info(f"個別解析: {mean_score:.4f}")

    return mean_score

def individual_analysis_with_dimension_reduction(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> float:
    losses: list[float] = []
    model_runner = ModelRunner(config)

    base_projector_seed = 0
    base_institution_index = 0
    default_F_type = getattr(config, "F_type", "svd")
    mixed = getattr(config, "True_F_type", None) == "kernel_pca_svd_mixed"

    for inst_offset, (X_tr, X_te, y_tr, y_te) in enumerate(zip(Xs_train, Xs_test, ys_train, ys_test)):
        institution_index = base_institution_index + inst_offset

        unique_labels, label_counts = np.unique(y_tr, return_counts=True)
        if unique_labels.size <= 1:
            logger.warning(
                f"individual_analysis: institution {inst_offset} has only {unique_labels.size} label(s); accuracy=0.0"
            )
            losses.append(0.0)
            continue
        if label_counts.min() < 2:
            logger.warning(
                f"individual_analysis: institution {inst_offset} has rare class count < 2; accuracy=0.0"
            )
            losses.append(0.0)
            continue

        if mixed:
            projector_F_type = "kernel_pca_self_tuning" if institution_index % 2 == 0 else "svd"
        else:
            projector_F_type = default_F_type

        projector = build_dimensionality_projector(
            X=X_tr,
            n_components=config.dim_intermediate,
            y=y_tr,
            F_type=projector_F_type,
            seed=base_projector_seed + inst_offset,
            config=config,
        )

        X_tr_svd = projector(X_tr)
        X_te_svd = projector(X_te)

        metrics = model_runner.run(X_tr_svd, y_tr, X_te_svd, y_te)
        losses.append(metrics)

    mean_score = float(np.mean(losses)) if losses else np.nan
    logger.info(f"個別解析（次元削減）: {mean_score:.4f}")
    return mean_score

# ----------------------------------------------------------------------
# 連合学習 (スクラッチ実装版)
# ----------------------------------------------------------------------
def fl_analysis(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> dict[str, object]:
    """Federated learning baseline evaluated per institution."""

    le = LabelEncoder()
    concat_parts: list[np.ndarray] = []
    if ys_train:
        concat_parts.append(np.concatenate(ys_train))
    if ys_test:
        concat_parts.append(np.concatenate(ys_test))
    if concat_parts:
        le.fit(np.concatenate(concat_parts))
    else:
        le.fit(np.array([], dtype=float))
    n_classes = len(le.classes_)

    clients_y_train_encoded = [le.transform(y) for y in ys_train]
    clients_y_test_encoded = [le.transform(y) for y in ys_test]

    client_stats = [
        {
            'n': X.shape[0],
            'sum': np.sum(X, axis=0),
            'sum_sq': np.sum(X ** 2, axis=0),
        }
        for X in Xs_train
    ]
    total_n = sum(s['n'] for s in client_stats)
    global_mean = sum(s['sum'] for s in client_stats) / total_n
    global_var = (sum(s['sum_sq'] for s in client_stats) / total_n) - (global_mean ** 2)
    global_var = np.maximum(global_var, 0.0)
    global_std = np.sqrt(global_var)
    global_std = np.nan_to_num(global_std, nan=0.0, posinf=0.0, neginf=0.0)
    global_std[global_std == 0] = 1.0

    clients_X_train_std = [
        np.nan_to_num((X - global_mean) / global_std)
        for X in Xs_train
    ]
    clients_X_test_std = [
        np.nan_to_num((X - global_mean) / global_std)
        for X in Xs_test
    ]

    fl_config = {
        "hidden_size": 256,
        "rounds": 10,
        "local_epochs": 5,
        "lr": 0.001,
        "l2": 1e-4,
        "batch_size": 64,
        "seed": config.seed,
        "metrics": config.metrics,
    }

    global_model = run_federated_learning(
        clients_X_train=clients_X_train_std,
        clients_y_train=clients_y_train_encoded,
        n_classes=n_classes,
        config=fl_config,
        logger=logger,
    )

    metric_name = str(getattr(config, "metrics", "auc")).lower()
    institution_metrics: list[float] = []
    institution_sizes: list[int] = []

    for idx, (X_te, y_te_enc) in enumerate(zip(clients_X_test_std, clients_y_test_encoded)):
        institution_sizes.append(len(y_te_enc))
        if X_te.size == 0 or len(y_te_enc) == 0:
            logger.warning(f"FL eval skipped for institution {idx}: no test samples.")
            institution_metrics.append(float("nan"))
            continue

        y_score = global_model.forward(X_te)
        if metric_name == "accuracy":
            y_pred = np.argmax(y_score, axis=1)
            metric_value = float(np.mean(y_pred == y_te_enc))
        else:
            try:
                if n_classes == 2:
                    if len(np.unique(y_te_enc)) < 2:
                        raise ValueError("binary AUC requires at least two classes in y_true.")
                    metric_value = float(roc_auc_score(y_te_enc, y_score[:, 1]))
                else:
                    unique_classes = np.unique(y_te_enc)
                    if len(unique_classes) < 2:
                        raise ValueError("multiclass AUC requires at least two classes in y_true.")
                    metric_value = float(
                        roc_auc_score(
                            y_te_enc,
                            y_score,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            except ValueError as exc:
                logger.warning(f"FL eval for institution {idx} failed: {exc}")
                metric_value = float("nan")

        institution_metrics.append(metric_value)
        logger.info(f"FL metric (institution {idx}): {metric_value}")

    metrics_array = np.array(institution_metrics, dtype=float)
    weights = np.array(institution_sizes, dtype=float)
    valid_mask = (~np.isnan(metrics_array)) & (weights > 0)
    if valid_mask.any():
        weighted_mean = float(np.average(metrics_array[valid_mask], weights=weights[valid_mask]))
    else:
        weighted_mean = float("nan")

    logger.info(f"FL per-institution metrics: {np.round(metrics_array, 4).tolist()}")
    logger.info(f"FL weighted mean metric: {weighted_mean:.4f}")

    return {
        "mean": weighted_mean,
        "per_institution": institution_metrics,
        "weights": institution_sizes,
    }


def _evaluate_classification_from_scores(
    *,
    y_true_enc: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
) -> float:
    if y_true_enc.size == 0 or y_score.size == 0:
        return float("nan")
    if metric_name == "accuracy":
        y_pred = np.argmax(y_score, axis=1)
        return float(np.mean(y_pred == y_true_enc))

    n_classes = int(y_score.shape[1])
    try:
        if n_classes == 2:
            if len(np.unique(y_true_enc)) < 2:
                return float("nan")
            return float(roc_auc_score(y_true_enc, y_score[:, 1]))
        if len(np.unique(y_true_enc)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true_enc, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def one_shot_guha_analysis(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> dict[str, object]:
    """
    One-shot FL (Guha-style) with client-side local training and server-side ensemble.

    Current implementation targets random_forest models:
      - Each client trains one local RF model.
      - Server selects client models by strategy (all/random/data/cv).
      - Global prediction is weighted average of predict_proba.
    """
    h_model = str(getattr(config, "h_model", "random_forest") or "random_forest").lower()
    if h_model != "random_forest":
        raise ValueError(
            "one-shot-guha currently supports h_model='random_forest' only. "
            f"Got: {h_model}"
        )

    # Shared label space for probability alignment and evaluation.
    le = LabelEncoder()
    concat_parts: list[np.ndarray] = []
    if ys_train:
        concat_parts.append(np.concatenate(ys_train))
    if ys_test:
        concat_parts.append(np.concatenate(ys_test))
    if concat_parts:
        le.fit(np.concatenate(concat_parts))
    else:
        le.fit(np.array([], dtype=float))
    global_classes = le.classes_
    n_classes = len(global_classes)
    if n_classes <= 1:
        raise ValueError("one-shot-guha requires at least 2 classes in labels.")

    # Keep same standardization convention as existing FL baseline.
    client_stats = [
        {
            "n": X.shape[0],
            "sum": np.sum(X, axis=0),
            "sum_sq": np.sum(X ** 2, axis=0),
        }
        for X in Xs_train
    ]
    total_n = sum(s["n"] for s in client_stats)
    if total_n <= 0:
        raise ValueError("one-shot-guha requires non-empty training data.")

    global_mean = sum(s["sum"] for s in client_stats) / total_n
    global_var = (sum(s["sum_sq"] for s in client_stats) / total_n) - (global_mean ** 2)
    global_var = np.maximum(global_var, 0.0)
    global_std = np.sqrt(global_var)
    global_std = np.nan_to_num(global_std, nan=0.0, posinf=0.0, neginf=0.0)
    global_std[global_std == 0] = 1.0

    clients_X_train_std = [np.nan_to_num((X - global_mean) / global_std) for X in Xs_train]
    clients_X_test_std = [np.nan_to_num((X - global_mean) / global_std) for X in Xs_test]

    # One-shot settings
    selection = str(getattr(config, "one_shot_selection", "all") or "all").lower()
    top_k_raw = getattr(config, "one_shot_top_k", None)
    top_k = int(top_k_raw) if top_k_raw is not None else None
    min_samples_raw = getattr(config, "one_shot_min_samples", 0)
    min_samples = int(min_samples_raw) if min_samples_raw is not None else 0
    val_ratio_raw = getattr(config, "one_shot_val_ratio", 0.1)
    try:
        val_ratio = float(val_ratio_raw)
    except (TypeError, ValueError):
        val_ratio = 0.1
    val_ratio = min(max(val_ratio, 0.0), 0.5)
    weighting = str(getattr(config, "one_shot_weighting", "n_samples") or "n_samples").lower()
    rng = np.random.default_rng(int(getattr(config, "seed", 0) or 0))

    # Train local models
    local_records: list[dict[str, object]] = []
    for i, (X_tr, y_tr_raw) in enumerate(zip(clients_X_train_std, ys_train)):
        if X_tr.size == 0 or len(y_tr_raw) == 0:
            continue
        if int(X_tr.shape[0]) < max(1, min_samples):
            continue

        y_tr = np.asarray(y_tr_raw)
        if len(np.unique(y_tr)) < 2:
            # Cannot train meaningful probabilistic classifier with a single class.
            continue

        rf = RandomForestClassifier(random_state=int((getattr(config, "seed", 0) or 0) + i))
        rf.fit(X_tr, y_tr)

        classes_local = np.asarray(rf.classes_)
        class_pos = {cls: j for j, cls in enumerate(classes_local)}
        global_idx = [int(np.where(global_classes == cls)[0][0]) for cls in classes_local if cls in class_pos]

        rec: dict[str, object] = {
            "model": rf,
            "global_idx": np.asarray(global_idx, dtype=int),
            "n_samples": int(X_tr.shape[0]),
            "client_idx": i,
            "cv_score": None,
        }

        # CV selection score on held-out local split (optional for selection stage).
        if selection == "cv":
            n = X_tr.shape[0]
            n_val = int(round(n * val_ratio))
            if n_val > 0 and n - n_val >= 2:
                perm = rng.permutation(n)
                tr_idx = perm[: n - n_val]
                va_idx = perm[n - n_val :]
                y_sub = y_tr[tr_idx]
                if len(np.unique(y_sub)) >= 2 and len(np.unique(y_tr[va_idx])) >= 1:
                    rf_cv = RandomForestClassifier(random_state=int((getattr(config, "seed", 0) or 0) + 10_000 + i))
                    rf_cv.fit(X_tr[tr_idx], y_sub)
                    proba_va = rf_cv.predict_proba(X_tr[va_idx])
                    score_va = _evaluate_classification_from_scores(
                        y_true_enc=le.transform(y_tr[va_idx]),
                        y_score=np.asarray(proba_va, dtype=float),
                        metric_name=str(getattr(config, "metrics", "accuracy") or "accuracy").lower(),
                    )
                    rec["cv_score"] = score_va
        local_records.append(rec)

    if not local_records:
        raise ValueError("one-shot-guha: no eligible local models were trained.")

    # Server-side model selection
    selected = list(local_records)
    if selection == "random":
        k = top_k if top_k is not None else len(selected)
        k = max(1, min(int(k), len(selected)))
        idx = rng.choice(len(selected), size=k, replace=False)
        selected = [selected[int(j)] for j in idx]
    elif selection == "data":
        selected = sorted(selected, key=lambda r: int(r["n_samples"]), reverse=True)
        if top_k is not None:
            selected = selected[: max(1, min(int(top_k), len(selected)))]
    elif selection == "cv":
        selected = sorted(
            selected,
            key=lambda r: float(r["cv_score"]) if r["cv_score"] is not None and np.isfinite(float(r["cv_score"])) else float("-inf"),
            reverse=True,
        )
        if top_k is not None:
            selected = selected[: max(1, min(int(top_k), len(selected)))]
        selected = [r for r in selected if r["cv_score"] is not None and np.isfinite(float(r["cv_score"]))]
        if not selected:
            # Fallback if every CV score was invalid.
            selected = sorted(local_records, key=lambda r: int(r["n_samples"]), reverse=True)
            if top_k is not None:
                selected = selected[: max(1, min(int(top_k), len(selected)))]
    elif selection != "all":
        raise ValueError(f"Unknown one_shot_selection: {selection}")

    if not selected:
        raise ValueError("one-shot-guha: no client models selected.")

    # Ensemble weights
    if weighting == "uniform":
        weights = np.full(len(selected), 1.0 / len(selected), dtype=float)
    elif weighting in {"n_samples", "data"}:
        counts = np.array([int(r["n_samples"]) for r in selected], dtype=float)
        s = float(np.sum(counts))
        if s <= 0:
            weights = np.full(len(selected), 1.0 / len(selected), dtype=float)
        else:
            weights = counts / s
    else:
        raise ValueError(f"Unknown one_shot_weighting: {weighting}")

    metric_name = str(getattr(config, "metrics", "accuracy") or "accuracy").lower()
    evaluate_mia = bool(getattr(config, "evaluate_mia", False))
    institution_metrics: list[float] = []
    institution_sizes: list[int] = []
    institution_mia_aucs: list[float] = []

    for idx, (X_te, y_te_raw) in enumerate(zip(clients_X_test_std, ys_test)):
        X_tr = clients_X_train_std[idx] if idx < len(clients_X_train_std) else np.empty((0, 0))
        y_tr_raw = ys_train[idx] if idx < len(ys_train) else np.array([])
        y_te = np.asarray(y_te_raw)
        institution_sizes.append(len(y_te))
        if X_te.size == 0 or len(y_te) == 0:
            logger.warning(f"one-shot-guha eval skipped for institution {idx}: no test samples.")
            institution_metrics.append(float("nan"))
            if evaluate_mia:
                institution_mia_aucs.append(float("nan"))
            continue

        y_score_ens = np.zeros((X_te.shape[0], n_classes), dtype=float)
        for w, rec in zip(weights, selected):
            model = rec["model"]
            global_idx = np.asarray(rec["global_idx"], dtype=int)
            proba_local = np.asarray(model.predict_proba(X_te), dtype=float)
            y_score_ens[:, global_idx] += float(w) * proba_local

        y_te_enc = le.transform(y_te)
        metric_value = _evaluate_classification_from_scores(
            y_true_enc=y_te_enc,
            y_score=y_score_ens,
            metric_name=metric_name,
        )
        institution_metrics.append(metric_value)
        logger.info(f"one-shot-guha metric (institution {idx}): {metric_value}")

        if evaluate_mia:
            try:
                if X_tr.size == 0 or len(y_tr_raw) == 0:
                    mia_auc_i = float("nan")
                else:
                    y_tr = np.asarray(y_tr_raw)
                    y_score_train = np.zeros((X_tr.shape[0], n_classes), dtype=float)
                    for w, rec in zip(weights, selected):
                        model = rec["model"]
                        global_idx = np.asarray(rec["global_idx"], dtype=int)
                        proba_local_tr = np.asarray(model.predict_proba(X_tr), dtype=float)
                        y_score_train[:, global_idx] += float(w) * proba_local_tr
                    mia_auc_i = compute_membership_auc(
                        y_member=y_tr,
                        score_member=y_score_train,
                        y_nonmember=y_te,
                        score_nonmember=y_score_ens,
                        classes=global_classes,
                    )
                institution_mia_aucs.append(mia_auc_i)
                logger.info(f"one-shot-guha MIA AUC (institution {idx}): {mia_auc_i:.4f}")
            except Exception as exc:
                institution_mia_aucs.append(float("nan"))
                logger.warning(f"one-shot-guha MIA evaluation failed (institution {idx}): {exc}")

    metrics_array = np.array(institution_metrics, dtype=float)
    weights_inst = np.array(institution_sizes, dtype=float)
    valid_mask = (~np.isnan(metrics_array)) & (weights_inst > 0)
    if valid_mask.any():
        weighted_mean = float(np.average(metrics_array[valid_mask], weights=weights_inst[valid_mask]))
    else:
        weighted_mean = float("nan")

    logger.info(
        "one-shot-guha selected models=%s/%s, selection=%s, weighting=%s",
        len(selected),
        len(local_records),
        selection,
        weighting,
    )
    logger.info(f"one-shot-guha per-institution metrics: {np.round(metrics_array, 4).tolist()}")
    logger.info(f"one-shot-guha weighted mean metric: {weighted_mean:.4f}")

    mia_auc_mean = float("nan")
    if evaluate_mia:
        mia_arr = np.array(institution_mia_aucs, dtype=float) if institution_mia_aucs else np.array([], dtype=float)
        mia_valid = np.isfinite(mia_arr)
        if mia_valid.any():
            w_mia = weights_inst[mia_valid]
            if np.sum(w_mia) > 0:
                mia_auc_mean = float(np.average(mia_arr[mia_valid], weights=w_mia))
            else:
                mia_auc_mean = float(np.mean(mia_arr[mia_valid]))
        logger.info(f"one-shot-guha MIA AUC (weighted mean): {mia_auc_mean:.4f}")

    return {
        "mean": weighted_mean,
        "per_institution": institution_metrics,
        "weights": institution_sizes,
        "num_models_trained": len(local_records),
        "num_models_selected": len(selected),
        "mia_auc": mia_auc_mean,
        "mia_auc_per_institution": institution_mia_aucs if evaluate_mia else [],
    }

def dca_analysis(
    X_train_integ: np.ndarray,
    X_test_integ: np.ndarray,
    y_train_integ: np.ndarray,
    y_test_integ: np.ndarray,
    config: Config,
    logger: logger,
) -> None:
    model_runner = ModelRunner(config)
    metrics = model_runner.run(
        X_train=X_train_integ,
        y_train=y_train_integ,
        X_test=X_test_integ,
        y_test=y_test_integ,
    )
    logger.info(f"提案手法の評価値: {metrics:.4f}")

    # Optional MIA evaluation (black-box true-label confidence AUC).
    # Kept fully opt-in to avoid changing legacy behavior.
    if bool(getattr(config, "evaluate_mia", False)):
        try:
            X_all = np.vstack([X_train_integ, X_test_integ])
            y_pred_all, y_score_all, classes = model_runner.predict_with_proba(
                X_train=X_train_integ,
                y_train=y_train_integ,
                X_test=X_all,
            )
            n_tr = int(X_train_integ.shape[0])
            score_m = np.asarray(y_score_all[:n_tr], dtype=float)
            score_n = np.asarray(y_score_all[n_tr:], dtype=float)
            mia_auc = compute_membership_auc(
                y_member=np.asarray(y_train_integ),
                score_member=score_m,
                y_nonmember=np.asarray(y_test_integ),
                score_nonmember=score_n,
                classes=np.asarray(classes),
            )
            setattr(config, "mia_auc_last", mia_auc)
            logger.info(f"MIA AUC (current institution): {mia_auc:.4f}")
        except Exception as exc:
            setattr(config, "mia_auc_last", float("nan"))
            logger.warning(f"MIA evaluation failed: {exc}")
    
    return metrics


def jiang_analysis(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> dict[str, object]:
    """
    Jiang et al. (IoTDI'19)-style lightweight PPCL baseline:
      - each institution applies its own private random projection
      - coordinator trains a global model on projected, concatenated data
      - evaluate per institution on projected test data
    """
    if not Xs_train or not ys_train:
        raise ValueError("jiang_analysis requires non-empty institutional training data.")

    proj_dim_raw = getattr(config, "jiang_proj_dim", 10)
    try:
        proj_dim = int(proj_dim_raw)
    except (TypeError, ValueError):
        proj_dim = 10
    proj_dim = max(1, proj_dim)

    seed_base = int(getattr(config, "seed", 0) or 0)
    dist = str(getattr(config, "jiang_proj_dist", "gaussian") or "gaussian").lower()
    normalize_cols = bool(getattr(config, "jiang_normalize_cols", True))

    Zs_train: list[np.ndarray] = []
    Zs_test: list[np.ndarray] = []
    ys_train_clean: list[np.ndarray] = []
    ys_test_clean: list[np.ndarray] = []

    for i, (X_tr, y_tr, X_te, y_te) in enumerate(zip(Xs_train, ys_train, Xs_test, ys_test)):
        X_tr_arr, y_tr_arr = ModelRunner._drop_nan_labels(X_tr, y_tr)
        X_te_arr, y_te_arr = ModelRunner._drop_nan_labels(X_te, y_te)
        if X_tr_arr.size == 0 or y_tr_arr.size == 0:
            continue

        d = int(X_tr_arr.shape[1])
        rng = np.random.default_rng(seed_base + i)
        if dist == "rademacher":
            R = rng.choice([-1.0, 1.0], size=(d, proj_dim))
        else:
            R = rng.standard_normal(size=(d, proj_dim))
        if normalize_cols:
            norms = np.linalg.norm(R, axis=0, keepdims=True)
            norms = np.where(norms <= 1e-12, 1.0, norms)
            R = R / norms

        Z_tr = X_tr_arr @ R
        Z_te = X_te_arr @ R if X_te_arr.size else np.empty((0, proj_dim), dtype=float)

        Zs_train.append(np.asarray(Z_tr, dtype=float))
        Zs_test.append(np.asarray(Z_te, dtype=float))
        ys_train_clean.append(np.asarray(y_tr_arr))
        ys_test_clean.append(np.asarray(y_te_arr))

    if not Zs_train:
        raise ValueError("jiang_analysis produced no valid projected training splits.")

    X_train_all = np.vstack(Zs_train)
    y_train_all = np.concatenate(ys_train_clean)

    model_runner = ModelRunner(config)
    institution_metrics: list[float] = []
    institution_sizes: list[int] = []

    # Optional MIA tracking
    mia_aucs: list[float] = []
    evaluate_mia = bool(getattr(config, "evaluate_mia", False))

    for idx, (Z_te, y_te) in enumerate(zip(Zs_test, ys_test_clean)):
        institution_sizes.append(int(len(y_te)))
        if Z_te.size == 0 or y_te.size == 0:
            institution_metrics.append(float("nan"))
            if evaluate_mia:
                mia_aucs.append(float("nan"))
            continue

        try:
            metric_val = model_runner.run(
                X_train=X_train_all,
                y_train=y_train_all,
                X_test=Z_te,
                y_test=y_te,
            )
        except Exception as exc:
            logger.warning(f"Jiang eval failed for institution {idx}: {exc}")
            metric_val = float("nan")
        institution_metrics.append(float(metric_val))

        if evaluate_mia:
            try:
                Z_train_for_mia = Zs_train[idx]
                y_train_for_mia = ys_train_clean[idx]
                X_all = np.vstack([Z_train_for_mia, Z_te])
                y_pred_all, y_score_all, classes = model_runner.predict_with_proba(
                    X_train=X_train_all,
                    y_train=y_train_all,
                    X_test=X_all,
                )
                n_tr = int(Z_train_for_mia.shape[0])
                score_m = np.asarray(y_score_all[:n_tr], dtype=float)
                score_n = np.asarray(y_score_all[n_tr:], dtype=float)
                mia_auc_i = compute_membership_auc(
                    y_member=np.asarray(y_train_for_mia),
                    score_member=score_m,
                    y_nonmember=np.asarray(y_te),
                    score_nonmember=score_n,
                    classes=np.asarray(classes),
                )
            except Exception as exc:
                logger.warning(f"Jiang MIA failed for institution {idx}: {exc}")
                mia_auc_i = float("nan")
            mia_aucs.append(mia_auc_i)

    metrics_array = np.array(institution_metrics, dtype=float)
    weights = np.array(institution_sizes, dtype=float)
    valid_mask = (~np.isnan(metrics_array)) & (weights > 0)
    if valid_mask.any():
        weighted_mean = float(np.average(metrics_array[valid_mask], weights=weights[valid_mask]))
    else:
        weighted_mean = float("nan")

    out = {
        "mean": weighted_mean,
        "per_institution": institution_metrics,
        "weights": institution_sizes,
    }

    if evaluate_mia:
        mia_arr = np.array(mia_aucs, dtype=float)
        mia_valid = np.isfinite(mia_arr)
        if mia_valid.any():
            w_mia = weights[mia_valid]
            if np.sum(w_mia) > 0:
                mia_mean = float(np.average(mia_arr[mia_valid], weights=w_mia))
            else:
                mia_mean = float(np.mean(mia_arr[mia_valid]))
        else:
            mia_mean = float("nan")
        out["mia_auc"] = mia_mean
        out["mia_auc_per_institution"] = mia_aucs

    logger.info(f"Jiang weighted mean metric: {weighted_mean:.4f}")
    logger.info(f"Jiang per-institution metrics: {np.round(metrics_array, 4).tolist()}")
    return out
