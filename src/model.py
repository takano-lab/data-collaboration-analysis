from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
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
    auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    return auc

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
    auc = roc_auc_score(y_test_enc, y_pred, multi_class='ovr')
    return auc

def run_svm_linear_classifier(
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
        SVC(kernel="linear", C=C, probability=True, random_state=random_state)
    )  
    
    y_train=y_train_enc
    y_test=y_test_enc
    # 学習
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)
    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")
    if n_classes == 2:
        print("二値分類のため、ROC曲線とAUCを計算・表示します")
        
        # AUCスコアを計算
        auc = roc_auc_score(y_test, y_score[:, 1])
        # print(f"ROC AUC: {auc:.4f}")

        # # ROC曲線のプロットデータを計算
        # # y_score[:, 1] は陽性クラス（クラス1）の確率です
        # print(y_test)
        # print(y_score[:, 1])
        # fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])

        # # グラフの作成
        # plt.figure(figsize=(8, 8))
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # ランダムレベルを示す対角線
        
        # # グラフの装飾
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate (偽陽性率)')
        # plt.ylabel('True Positive Rate (真陽性率)')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.grid(True)
        # plt.show()
    else:
        auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")

    return auc

# モデル関数を辞書として定義
h_models = {
    "linear_regression": run_linear_regression,
    "random_forest": run_random_forest_classifier,
    "svm_classifier": run_svm_classifier,
    "svm_linear_classifier": run_svm_linear_classifier,
}