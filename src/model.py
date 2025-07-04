from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


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
    random_state: int = 42,
) -> float:
    """線形回帰で RMSE を返す"""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# ----------------------------------------------------------------------
# SVM（RBF カーネル）分類器
#   - 特徴量スケーリング：StandardScaler
#   - y が文字列でも自動で数値化
#   - バイナリ／多クラスは SVC が勝手に One-Vs-One で処理
# ----------------------------------------------------------------------
def run_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = 1.0,
    random_state: int = 42,
) -> float:
    """
    線形SVM による分類（カーネルなし）を行い、accuracy を返す。

    前処理:
      1. 特徴量を StandardScaler で平均 0・分散 1 にスケーリング
      2. ラベルが非数値の場合は LabelEncoder で数値化
    """
    label_encoder: Optional[LabelEncoder] = None
    if not np.issubdtype(y_train.dtype, np.number):
        label_encoder = LabelEncoder().fit(y_train)
        y_train_enc = label_encoder.transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
    else:
        y_train_enc = y_train
        y_test_enc = y_test
        
    model = make_pipeline(
        StandardScaler(),
        #SVC(kernel="linear", C=C, random_state=random_state)
        SVC(kernel="rbf", C=C, gamma="scale", random_state=random_state)
    )
    # if config.G_type == "GEP_weighted"
        # model = make_pipeline(
        #     SVC(kernel="rbf", C=C, gamma="scale", random_state=random_state)
        # )    

    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)
    return accuracy

# モデル関数を辞書として定義
h_models = {
    "linear_regression": run_linear_regression,
    "random_forest": run_random_forest_classifier,
    "svm_classifier": run_svm_classifier,
}