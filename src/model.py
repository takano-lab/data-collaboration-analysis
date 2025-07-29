from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
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


# --- 機械学習モデル実行クラス ---

class ModelRunner:
    """
    configに基づいて機械学習モデルの学習と評価を行うクラス。
    """
    def __init__(self, config: Any):
        self.config = config
        # config.h_model の値と実行するメソッドをマッピング
        self._model_map = {
            "linear_regression": self._run_linear_regression,
            "random_forest": self._run_random_forest,
            "svm_classifier": self._run_svm,
            "svm_linear_classifier": self._run_svm_linear,
            "mlp": self._run_mlp,  # MLPを追加
        }

    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        configで指定されたモデルを実行し、評価値を返す。
        """
        model_func = self._model_map.get(self.config.h_model)
        if model_func is None:
            raise ValueError(f"Unknown model name in config: {self.config.h_model}")

        # configにeigenvaluesがあれば、キーワード引数として渡す
        kwargs = {}
        if hasattr(self.config, 'eigenvalues'):
            kwargs['eigenvalues'] = self.config.eigenvalues
        
        return model_func(X_train, y_train, X_test, y_test, **kwargs)

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, n_classes: int) -> float:
        """
        config.metricsに基づいて評価指標を計算する。
        """
        metric = getattr(self.config, 'metrics', 'auc').lower()  # デフォルトはauc

        if metric == 'auc':
            if y_score is None:
                raise ValueError("AUCを計算するには予測確率(y_score)が必要です。")
            if n_classes == 2:
                return roc_auc_score(y_true, y_score[:, 1])
            else:
                return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        
        elif metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        
        else:
            raise ValueError(f"未対応の評価指標です: {self.config.metrics}")

    def _run_linear_regression(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """線形回帰で RMSE を返す"""
        # 注意: linear_regressionは回帰モデルのため、accuracy/aucは適用されません。
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, y_pred))

    def _run_random_forest(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """ランダムフォレストで評価指標を計算する"""
        model = RandomForestClassifier(random_state=self.config.seed)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        n_classes = len(model.classes_)
        
        return self._evaluate(y_test, y_pred, y_score, n_classes)

    def _run_svm(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """RBFカーネルSVMで評価指標を計算する"""
        return self._execute_svm(X_train, y_train, X_test, y_test, kernel="rbf", **kwargs)

    def _run_svm_linear(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """線形カーネルSVMで評価指標を計算する"""
        return self._execute_svm(X_train, y_train, X_test, y_test, kernel="linear", **kwargs)

    def _run_mlp(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """MLPで評価指標を計算する"""
        # ラベルのエンコード
        if not np.issubdtype(y_train.dtype, np.number):
            encoder = LabelEncoder().fit(y_train)
            y_train = encoder.transform(y_train)
            y_test = encoder.transform(y_test)

        # パイプラインの構築
        steps = [StandardScaler()]  # 常にStandardScalerを適用
        eigenvalues = kwargs.get('eigenvalues', None)
        if eigenvalues is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=eigenvalues))

        # MLPモデルの追加
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,  # early_stoppingに必要
            n_iter_no_change=10,      # early_stoppingに必要
            random_state=self.config.seed
        )
        steps.append(mlp_model)

        # パイプラインの作成
        model = make_pipeline(*steps)

        # 学習と評価
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        n_classes = len(model.classes_)

        return self._evaluate(y_test, y_pred, y_score, n_classes)

    def _execute_svm(self, X_train, y_train, X_test, y_test, kernel: str, eigenvalues: Optional[list] = None) -> float:
        """SVMの共通処理"""
        # ラベルのエンコード
        if not np.issubdtype(y_train.dtype, np.number):
            encoder = LabelEncoder().fit(y_train)
            y_train = encoder.transform(y_train)
            y_test = encoder.transform(y_test)

        # パイプラインの構築
        steps = [StandardScaler()]
        if eigenvalues is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=eigenvalues))
        
        c_param = getattr(self.config, 'h_C', 1.0)
        if c_param is None:
            c_param = 1.0

        svc_params = {
            "kernel": kernel,
            "C": c_param,
            "probability": True,
            "random_state": self.config.seed,
        }
        if kernel == "rbf":
            svc_params["gamma"] = "scale"
        
        steps.append(SVC(**svc_params))
        model = make_pipeline(*steps)

        # 学習と評価
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        n_classes = len(model.classes_)
        
        return self._evaluate(y_test, y_pred, y_score, n_classes)


# --- エントリポイント関数 ---

def h_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Any,
) -> float:
    """
    ModelRunnerを介して機械学習モデルを実行し、評価値を返す。
    """
    runner = ModelRunner(config)
    metrics = runner.run(X_train, y_train, X_test, y_test)
    return metrics