from __future__ import annotations

from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as tlgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from surprise import Dataset, KNNBasic, Reader, accuracy

# from pyfm import pylibfm
# from scipy.sparse import csr_matrix


def run_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    categorical_cols: Optional[list[str]] = None,
    use_optuna: bool = False,
    seed: int = 42,
) -> float:
    """
    LightGBMの予測を評価する関数
    """
    if categorical_cols is None:
        categorical_cols = []

    models = []
    oof_train = np.zeros((len(X_train),))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": 0.1,
        "verbosity": -1,
        "random_state": seed,
    }

    train_params = {
        "num_boost_round": 999999,
    }

    for _, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train[train_index, :]
        X_val = X_train[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        if use_optuna:
            lgb_train = tlgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)
            lgb_eval = tlgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols)

            model = tlgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)],
                **train_params,
            )
        else:
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)],
                **train_params,
            )
        oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)

    # testデータでの性能を評価
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


def run_surprise(dataset: str, train_df: pd.DataFrame, test_df: pd.DataFrame, neightbors: int = 10) -> float:
    """
    推薦システムでの性能を評価する関数
    """
    rating_columns = ["uid", "mid", "rating"]  # レーティングのカラム名
    if dataset == "movielens":
        reader = Reader(rating_scale=(1, 5))
    elif dataset == "sushi":
        reader = Reader(rating_scale=(0, 4))

    # surpriseでデータを読み込む
    train_surprise_data = Dataset.load_from_df(train_df[rating_columns], reader)
    test_surprise_data = Dataset.load_from_df(test_df[rating_columns], reader)

    # データセットをビルド
    trainset = train_surprise_data.build_full_trainset()
    testset = test_surprise_data.build_full_trainset()
    testset = testset.build_testset()

    # アルゴリズムを設定
    algo = KNNBasic(k=neightbors)

    # 学習
    algo.fit(trainset)

    # testデータでの性能を評価
    pred = algo.test(testset)
    rmse = accuracy.rmse(pred)

    return rmse


# def run_fm(config, X_train, y_train, X_test, y_test):
#     # csr_matrixに変換
#     X_train = csr_matrix(X_train, dtype=np.float64)
#     X_test = csr_matrix(X_test, dtype=np.float64)

#     fm = pylibfm.FM(
#         num_factors=20,
#         num_iter=100,
#         task="regression",
#         initial_learning_rate=0.001,
#         learning_rate_schedule="optimal",
#         verbose=False,
#     )
#     fm.fit(X_train, y_train)

#     # testデータでの性能を評価
#     y_pred = fm.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#     return rmse
