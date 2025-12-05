from __future__ import annotations

from typing import Dict, Optional, TypeVar
import copy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.federated_learning import run_federated_learning  # スクラッチ実装をインポート
from src.model import ModelRunner
from src.dimensionality_reduction import build_dimensionality_projector
from src.institution_data_pipeline.builders import InstitutionDatasetBuilder

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
    all_y_train = np.concatenate(ys_train)
    le.fit(all_y_train)
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
        "lr": 0.01,
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
    
    return metrics

