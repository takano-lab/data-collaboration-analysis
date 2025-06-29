from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold


# ----------------------------------------------------------------------
# 線形回帰 
# ----------------------------------------------------------------------
def run_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """線形回帰で RMSE を返す"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# ----------------------------------------------------------------------
# ランダムフォレスト
# ----------------------------------------------------------------------
def run_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """線形回帰で RMSE を返す"""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# モデル関数を辞書として定義
h_models = {
    "linear_regression": run_linear_regression,
    "random_forest": run_random_forest_classifier,
}