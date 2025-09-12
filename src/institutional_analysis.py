from __future__ import annotations

from typing import Optional, TypeVar

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.federated_learning import run_federated_learning  # スクラッチ実装をインポート
from src.model import ModelRunner

#from src.model import h_ml_model, h_models
from src.utils import reduce_dimensions

logger = TypeVar("logger")

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
    #record_value_to_cfg(config, "集中解析", metrics)
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
    X_tr_svd, X_te_svd = reduce_dimensions(X_train, X_test, n_components=config.dim_integrate, seed=config.f_seed)
    model_runner = ModelRunner(config)
    metrics = model_runner.run(
                    X_train=X_tr_svd,
                    y_train=y_train,
                    X_test=X_te_svd,
                    y_test=y_test
                )

    logger.info(f"集中解析（次元削減）の評価値: {metrics:.4f}")
    #record_value_to_cfg(config, "集中解析（次元削減）", metrics)
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
    #record_value_to_cfg(config, "個別解析", np.mean(losses))
    return np.mean(losses)

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

    even_losses = []    
    odd_losses = []
    for i, (X_tr, X_te, y_tr, y_te) in enumerate(zip(Xs_train, Xs_test, ys_train, ys_test)):
        X_tr_svd, X_te_svd = reduce_dimensions(X_tr, X_te, n_components=config.dim_intermediate)
        metrics = model_runner.run(X_tr_svd, y_tr, X_te_svd, y_te)
        losses.append(metrics)
        if i % 2 == 0:
            even_losses.append(metrics)
        else:
            odd_losses.append(metrics)

    config.losses_even_ind = round(sum(even_losses)/len(even_losses), 4)
    config.losses_odd_ind = round(sum(odd_losses)/len(odd_losses), 4)
    config.losses_ind = round(sum(losses)/len(losses), 4)
    logger.info(f"個別解析の評価値: {np.mean(losses):.4f}")
    #record_value_to_cfg(config, "個別解析（次元削減）", np.mean(losses))
    return config.losses_ind


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
) -> float:
    """スクラッチ実装の連合学習を実行します。"""
    
    # 1. LabelEncoderを訓練データのみでfit
    le = LabelEncoder()
    all_y_train = np.concatenate(ys_train)
    le.fit(all_y_train)
    n_classes = len(le.classes_)

    # 訓練・テストデータのラベルを変換
    clients_y_train_encoded = [le.transform(y) for y in ys_train]
    y_test_all_encoded = le.transform(np.concatenate(ys_test))
    
    # 2. 特徴量の標準化 (連合学習の枠組みで)
    client_stats = [{'n': X.shape[0], 'sum': np.sum(X, axis=0), 'sum_sq': np.sum(X**2, axis=0)} for X in Xs_train]
    total_n = sum(s['n'] for s in client_stats)
    global_mean = sum(s['sum'] for s in client_stats) / total_n
    global_var = (sum(s['sum_sq'] for s in client_stats) / total_n) - (global_mean**2)
    global_std = np.sqrt(global_var)
    global_std[global_std == 0] = 1.0

    # 各クライアントのデータを標準化
    clients_X_train_std = [(X - global_mean) / global_std for X in Xs_train]
    X_test_all_std = (np.concatenate(Xs_test) - global_mean) / global_std

    # 3. スクラッチ実装のFL関数を呼び出し
    fl_config = {
        "hidden_size": 256,
        "rounds": 10,
        "local_epochs": 5,
        "lr": 0.01,
        "seed": config.seed
    }

    # 連合学習の実行
    final_auc = run_federated_learning(
        clients_X_train=clients_X_train_std,
        clients_y_train=clients_y_train_encoded,
        X_test=X_test_all_std,
        y_test=y_test_all_encoded,
        n_classes=n_classes,
        config=fl_config,
        logger=logger
    )

    logger.info(f"FL解析 (Scratch) の最終評価値: {final_auc:.4f}")
    #record_value_to_cfg(config, "FL解析", final_auc)
    return final_auc



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
    #record_value_to_cfg(config, "提案手法", metrics)
    return metrics

