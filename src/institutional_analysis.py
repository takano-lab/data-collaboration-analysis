from __future__ import annotations

from typing import Optional, TypeVar

import category_encoders as ce
import numpy as np
import pandas as pd

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.model import ModelRunner
#from src.model import h_ml_model, h_models
from src.utils import reduce_dimensions

logger = TypeVar("logger")

def h_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> float:
    """機械学習モデルを実行し、評価値を返す"""
    evaluate_model = h_models[config.h_model]
    metrics = evaluate_model(X_train, y_train, X_test, y_test)
    return metrics
# ----------------------------------------------------------------------
# 集中解析
# ----------------------------------------------------------------------
def centralize_analysis(config: Config, logger: logger, y_name) -> None:
    train_df = pd.read_csv(config.output_path / "train.csv")
    test_df = pd.read_csv(config.output_path / "test.csv")

    # 目的変数と特徴量を分離
    y_train = train_df.pop(y_name).values
    y_test = test_df.pop(y_name).values
    
    X_train = train_df.values
    X_test = test_df.values

    # SVD
    X_tr_svd, X_te_svd = X_train, X_test
    model_runner = ModelRunner(config)
    metrics = model_runner.run(
                    X_train=X_tr_svd,
                    y_train=y_train,
                    X_test=X_te_svd,
                    y_test=y_test
                )
    
    
    logger.info(f"集中解析の評価値: {metrics:.4f}")
    record_value_to_cfg(config, "集中解析", metrics)
    return metrics

def centralize_analysis_with_dimension_reduction(config: Config, logger: logger, y_name) -> None:
    train_df = pd.read_csv(config.output_path / "train.csv")
    test_df = pd.read_csv(config.output_path / "test.csv")

    # 目的変数と特徴量を分離
    y_train = train_df.pop(y_name).values
    y_test = test_df.pop(y_name).values
    
    X_train = train_df.values
    X_test = test_df.values

    # SVD
    #X_tr_svd, X_te_svd = reduce_dimensions(X_train, X_test, n_components=config.dim_integrate)
    X_tr_svd, X_te_svd = X_train, X_test
    metrics = h_ml_model(
        X_train=X_tr_svd,
        y_train=y_train,
        X_test=X_te_svd,
        y_test=y_test,
        config=config,
    )
    
    logger.info(f"集中解析の評価値: {metrics:.4f}")
    record_value_to_cfg(config, "集中解析", metrics)
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
) -> None:
    losses: list[float] = []

    model_runner = ModelRunner(config)

    for X_tr, X_te, y_tr, y_te in zip(Xs_train, Xs_test, ys_train, ys_test):
        #X_tr_svd, X_te_svd = reduce_dimensions_with_svd(X_tr, X_te, n_components=config.dim_intermediate)
        X_tr_svd, X_te_svd = X_tr, X_te
        metrics = model_runner.run(X_tr_svd, y_tr, X_te_svd, y_te)
        losses.append(metrics)
        break
        
    logger.info(f"個別解析の評価値: {np.mean(losses):.4f}")
    record_value_to_cfg(config, "個別解析", np.mean(losses))
    return metrics

# ----------------------------------------------------------------------
# 個別解析
# ----------------------------------------------------------------------
def individual_analysis_with_dimension_reduction(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> None:
    losses: list[float] = []
    
    model_runner = ModelRunner(config)

    for X_tr, X_te, y_tr, y_te in zip(Xs_train, Xs_test, ys_train, ys_test):
        print(y_te)
        X_tr_svd, X_te_svd = reduce_dimensions(X_tr, X_te, n_components=config.dim_intermediate)
        metrics = model_runner.run(X_tr_svd, y_tr, X_te_svd, y_te)
        losses.append(metrics)
        break
        
    logger.info(f"個別解析の評価値: {np.mean(losses):.4f}")
    record_value_to_cfg(config, "個別解析（次元削減）", np.mean(losses))
    return metrics

# ----------------------------------------------------------------------
# データ統合解析（DCA）
# ----------------------------------------------------------------------
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
    record_value_to_cfg(config, "提案手法", metrics)
    return metrics

