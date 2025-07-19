from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# --- 前処理用カスタム変換器 ---
class EigenWeightingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eigenvalues):
        self.eigenvalues = np.array(eigenvalues)
        self.weights_ = None

    def fit(self, X, y=None):
        lam = self.eigenvalues
        lam1 = lam[0]
        lamm = lam[-1]
        denom = lamm - lam1 if lamm != lam1 else 1e-8
        self.weights_ = np.exp(-(lam - lam1) / denom)
        return self

    def transform(self, X):
        if self.weights_ is None:
            raise RuntimeError("fit() must be called before transform()")
        return X * self.weights_

# --- 機械学習モデルのクラス設計 ---

class BaseMLModel(ABC):
    """全ての機械学習モデルの基底クラス"""
    def __init__(self, random_state: int = 42, **kwargs: Any):
        self.random_state = random_state
        self.model: Optional[Pipeline | BaseEstimator] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.params = kwargs

    def _preprocess_labels(self, y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ラベルが非数値の場合にエンコードする"""
        if not np.issubdtype(y_train.dtype, np.number):
            self.label_encoder = LabelEncoder().fit(y_train)
            y_train_enc = self.label_encoder.transform(y_train)
            y_test_enc = self.label_encoder.transform(y_test)
            return y_train_enc, y_test_enc
        return y_train, y_test

    @abstractmethod
    def _build_model(self) -> Pipeline | BaseEstimator:
        """モデルのパイプラインを構築する"""
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """モデルを学習させる"""
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """モデルを評価し、指標を返す"""
        pass

    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """学習から評価までの一連の流れを実行する"""
        y_train_proc, y_test_proc = self._preprocess_labels(y_train, y_test)
        self.fit(X_train, y_train_proc)
        return self.evaluate(X_test, y_test_proc)


class SVMClassifier(BaseMLModel):
    """SVM分類器モデル"""
    def _build_model(self) -> Pipeline:
        steps = [StandardScaler()]
        if self.params.get("eigenvalues") is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=self.params["eigenvalues"]))
        
        svc_params = {
            "kernel": self.params.get("kernel", "rbf"),
            "C": self.params.get("C", 1.0),
            "probability": True,
            "random_state": self.random_state,
        }
        if svc_params["kernel"] == "rbf":
            svc_params["gamma"] = "scale"
            
        steps.append(SVC(**svc_params))
        return make_pipeline(*steps)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.model.classes_)
        
        if n_classes == 2:
            return roc_auc_score(y_test, y_score[:, 1])
        else:
            return roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")


class RandomForestClassifierModel(BaseMLModel):
    """ランダムフォレスト分類器モデル"""
    def _build_model(self) -> BaseEstimator:
        return RandomForestClassifier(random_state=self.random_state)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        # predict_probaの方がより一般的にAUCを計算できる
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.model.classes_)

        if n_classes == 2:
            return roc_auc_score(y_test, y_score[:, 1])
        else:
            return roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")


class LinearRegressionModel(BaseMLModel):
    """線形回帰モデル"""
    def _build_model(self) -> BaseEstimator:
        return LinearRegression()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        y_pred = self.model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, y_pred))


# --- モデルファクトリ ---

def get_model(config: Config) -> BaseMLModel:
    """設定に応じてモデルインスタンスを返すファクトリ関数"""
    model_name = config.h_model
    params = {
        "random_state": config.seed,
        "C": config.h_C,
        "kernel": config.h_kernel,
        "eigenvalues": config.get("eigenvalues") # configにeigenvaluesがあれば渡す
    }

    if model_name == "svm":
        return SVMClassifier(**params)
    elif model_name == "random_forest":
        return RandomForestClassifierModel(**params)
    elif model_name == "linear_regression":
        return LinearRegressionModel(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# モデル関数を辞書として定義
h_models = {
    "linear_regression": run_linear_regression,
    "random_forest": run_random_forest_classifier,
    "svm_classifier": run_svm_classifier,
    "svm_linear_classifier": run_svm_linear_classifier,
}

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