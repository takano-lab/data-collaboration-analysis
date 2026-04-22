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
            "softmax": self._run_softmax,
            "logistic_regression": self._run_softmax,
        }
        self._last_train_labels: Optional[np.ndarray] = None

    
    @staticmethod
    def _drop_nan_labels(X, y):
        """Remove samples whose labels are missing (None/NaN)."""
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in {"f", "c"}:
            mask = ~np.isnan(y_arr)
        else:
            def _is_valid(val):
                return not (val is None or (isinstance(val, float) and np.isnan(val)))

            mask = np.array([_is_valid(val) for val in y_arr], dtype=bool)

        X_arr = np.asarray(X)
        if X_arr.shape[0] != mask.size:
            raise ValueError("Feature and label sizes do not match during NaN removal.")
        if mask.all():
            return X_arr, y_arr
        return X_arr[mask], y_arr[mask]

    def run(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        configで指定されたモデルを実行し、評価値を返す。
        """
        X_train, y_train = self._drop_nan_labels(X_train, y_train)
        if len(y_train) == 0:
            raise ValueError("No labeled samples available after dropping NaNs.")
        self._last_train_labels = np.unique(y_train)
        model_func = self._model_map.get(self.config.h_model)
        if model_func is None:
            raise ValueError(f"Unknown model name in config: {self.config.h_model}")

        # configにeigenvaluesがあれば、キーワード引数として渡す
        kwargs = {}
        if hasattr(self.config, 'eigenvalues'):
            kwargs['eigenvalues'] = self.config.eigenvalues
        
        return model_func(X_train, y_train, X_test, y_test, **kwargs)

    def predict_with_proba(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
        """
        学習済みモデルで予測ラベルと確率を返すヘルパー。
        戻り値: y_pred(元ラベル), y_proba(shape=(n_samples, n_classes)), classes(元ラベル順)
        """
        X_train, y_train = self._drop_nan_labels(X_train, y_train)
        if len(y_train) == 0:
            raise ValueError("No labeled samples available after dropping NaNs.")
        self._last_train_labels = np.unique(y_train)
        # ラベルのエンコード（各分類器実装に合わせて統一）
        use_encoder = False
        encoder = None
        y_train_enc = y_train
        h_model = getattr(self.config, 'h_model', 'svm_classifier')
        # SVM/MLP/Softmax は内部でエンコードしているため合わせる
        if h_model in ["svm_classifier", "svm_linear_classifier", "mlp", "softmax", "logistic_regression", "random_forest"]:
            if not np.issubdtype(y_train.dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder().fit(y_train)
                y_train_enc = encoder.transform(y_train)
                use_encoder = True

        steps = [StandardScaler()]

        eigenvalues = getattr(self.config, 'eigenvalues', None)
        if eigenvalues is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=eigenvalues))

        model = None
        if h_model == "svm_classifier" or h_model == "svm_linear_classifier":
            kernel = "rbf" if h_model == "svm_classifier" else "linear"
            c_param = getattr(self.config, 'h_C', 1.0) or 1.0
            svc_params = {
                "kernel": kernel,
                "C": c_param,
                "probability": True,
                "random_state": self.config.seed,
            }
            if kernel == "rbf":
                svc_params["gamma"] = "scale"
            steps.append(SVC(**svc_params))
            pipeline = make_pipeline(*steps)
            pipeline.fit(X_train, y_train_enc if use_encoder else y_train)
            y_pred_enc = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            if use_encoder:
                # pipeline.classes_ は存在しないため、末尾推定器から取得
                classes_enc = pipeline[-1].classes_
                classes = encoder.inverse_transform(classes_enc)
                y_pred = encoder.inverse_transform(y_pred_enc)
            else:
                classes = pipeline[-1].classes_
                y_pred = y_pred_enc

            return y_pred, y_proba, np.array(classes)

        elif h_model == "mlp":
            n_samples = X_train.shape[0]
            base_val_fraction = 0.1
            use_early_stopping = True
            if n_samples * base_val_fraction < 2:
                use_early_stopping = False
            mlp_model = MLPClassifier(
                hidden_layer_sizes=(256,),
                activation='relu',
                solver='adam',
                max_iter=1000,
                early_stopping=use_early_stopping,
                validation_fraction=base_val_fraction,
                n_iter_no_change=10,
                random_state=self.config.seed
            )
            steps_local = steps + [mlp_model]
            pipeline = make_pipeline(*steps_local)
            pipeline.fit(X_train, y_train_enc if use_encoder else y_train)
            y_pred_enc = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            if use_encoder:
                classes_enc = pipeline[-1].classes_
                classes = encoder.inverse_transform(classes_enc)
                y_pred = encoder.inverse_transform(y_pred_enc)
            else:
                classes = pipeline[-1].classes_
                y_pred = y_pred_enc
            return y_pred, y_proba, np.array(classes)

        elif h_model in ["softmax", "logistic_regression"]:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.config.seed
            )
            steps_local = steps + [clf]
            pipeline = make_pipeline(*steps_local)
            pipeline.fit(X_train, y_train_enc if use_encoder else y_train)
            y_pred_enc = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            if use_encoder:
                classes_enc = pipeline[-1].classes_
                classes = encoder.inverse_transform(classes_enc)
                y_pred = encoder.inverse_transform(y_pred_enc)
            else:
                classes = pipeline[-1].classes_
                y_pred = y_pred_enc
            return y_pred, y_proba, np.array(classes)

        elif h_model == "random_forest":
            # RF は本来エンコード無しでも動作するが、一貫性のため encoder を適用済みなら enc を使用
            rf = RandomForestClassifier(random_state=self.config.seed)
            pipeline = rf  # 標準化は RF には不要
            pipeline.fit(X_train, y_train_enc if use_encoder else y_train)
            y_pred_enc = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            if use_encoder:
                classes_enc = pipeline.classes_
                classes = encoder.inverse_transform(classes_enc)
                y_pred = encoder.inverse_transform(y_pred_enc)
            else:
                classes = pipeline.classes_
                y_pred = y_pred_enc
            return y_pred, y_proba, np.array(classes)

        else:
            # フォールバック: SVM RBF と同様に扱う
            c_param = getattr(self.config, 'h_C', 1.0) or 1.0
            svc_params = {
                "kernel": "rbf",
                "C": c_param,
                "probability": True,
                "random_state": self.config.seed,
                "gamma": "scale",
            }
            steps.append(SVC(**svc_params))
            pipeline = make_pipeline(*steps)
            pipeline.fit(X_train, y_train_enc if use_encoder else y_train)
            y_pred_enc = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            if use_encoder:
                classes_enc = pipeline[-1].classes_
                classes = encoder.inverse_transform(classes_enc)
                y_pred = encoder.inverse_transform(y_pred_enc)
            else:
                classes = pipeline[-1].classes_
                y_pred = y_pred_enc
            return y_pred, y_proba, np.array(classes)

    def _evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray],
        classes_pred: Optional[np.ndarray],
    ) -> float:
        """
        config.metrics に応じて評価指標を算出する
        """
        metric = getattr(self.config, 'metrics', 'auc').lower()

        if metric in ['rmse', 'r2']:
            if metric == 'rmse':
                return np.sqrt(mean_squared_error(y_true, y_pred))
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)

        if metric == 'auc':
            if y_score is None or y_score.ndim != 2 or classes_pred is None:
                return np.nan
            classes_pred = np.asarray(classes_pred)
            row_mask = np.isin(y_true, classes_pred)
            if not row_mask.any():
                return np.nan
            y_true = y_true[row_mask]
            y_pred = y_pred[row_mask]
            y_score = y_score[row_mask]

            present_classes = np.unique(y_true)
            if present_classes.size <= 1:
                return np.nan

            col_indices = []
            for cls in present_classes:
                idx = np.where(classes_pred == cls)[0]
                if idx.size == 0:
                    continue
                col_indices.append(int(idx[0]))
            if not col_indices:
                return np.nan
            y_score = y_score[:, col_indices]

            try:
                if len(col_indices) == 2:
                    return roc_auc_score(y_true, y_score[:, 1])
                return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            except ValueError:
                return np.nan

        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)

        raise ValueError(f"未知の評価指標です: {self.config.metrics}")

    def _run_linear_regression(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """線形回帰で評価指標を計算する"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # 線形回帰は回帰問題なので、評価指標はRMSEまたはR2など
        metric = getattr(self.config, 'metrics', 'rmse').lower()
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y_test, y_pred)
        else:
            raise ValueError(f"未対応の回帰評価指標です: {self.config.metrics}")

    def _run_random_forest(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """ランダムフォレストで評価指標を算出する"""
        model = RandomForestClassifier(random_state=self.config.seed)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        classes_pred = getattr(model, "classes_", None)
        
        return self._evaluate(y_test, y_pred, y_score, classes_pred)

    def _run_svm(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """RBFカーネルSVMで評価指標を計算する"""
        return self._execute_svm(X_train, y_train, X_test, y_test, kernel="rbf", **kwargs)

    def _run_svm_linear(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """線形カーネルSVMで評価指標を計算する"""
        return self._execute_svm(X_train, y_train, X_test, y_test, kernel="linear", **kwargs)

    def _run_mlp(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """MLPで評価指標を算出する"""
        # ラベルのエンコード
        if not np.issubdtype(y_train.dtype, np.number):
            encoder = LabelEncoder().fit(y_train)
            y_train = encoder.transform(y_train)
            y_test = encoder.transform(y_test)

        # パイプラインの構成
        steps = [StandardScaler()]  # 先に StandardScaler を適用
        eigenvalues = kwargs.get('eigenvalues', None)
        if eigenvalues is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=eigenvalues))

        # MLP モデルを追加
        n_samples = X_train.shape[0]
        base_val_fraction = 0.1
        use_early_stopping = True
        if n_samples * base_val_fraction < 2:
            use_early_stopping = False
        else:
            _, class_counts = np.unique(y_train, return_counts=True)
            if class_counts.size == 0 or class_counts.min() < 2:
                use_early_stopping = False
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            #alpha=1e-3,  # 少し強めのL2正則化で収束を安定化
            early_stopping=use_early_stopping,
            validation_fraction=base_val_fraction,
            n_iter_no_change=10,
            random_state=self.config.seed
        )
        steps.append(mlp_model)

        # モデルの学習と評価
        model = make_pipeline(*steps)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        classes_pred = getattr(model, "classes_", None)

        return self._evaluate(y_test, y_pred, y_score, classes_pred)

    def _run_softmax(self, X_train, y_train, X_test, y_test, **kwargs) -> float:
        """ロジスティック回帰（多クラスsoftmax）で評価指標を計算する"""
        from sklearn.linear_model import LogisticRegression

        # ラベルのエンコード
        if not np.issubdtype(y_train.dtype, np.number):
            encoder = LabelEncoder().fit(y_train)
            y_train = encoder.transform(y_train)
            y_test = encoder.transform(y_test)

        # パイプラインの構築
        steps = [StandardScaler()]
        eigenvalues = kwargs.get('eigenvalues', None)
        if eigenvalues is not None:
            steps.append(EigenWeightingTransformer(eigenvalues=eigenvalues))

        # ロジスティック回帰（多クラス対応）
        clf = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=self.config.seed
        )
        steps.append(clf)
        model = make_pipeline(*steps)

        # 学習と評価
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        classes_pred = clf.classes_ if hasattr(clf, "classes_") else np.unique(y_train)

        return self._evaluate(y_test, y_pred, y_score, classes_pred)

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
        classes_pred = getattr(model, "classes_", None)

        return self._evaluate(y_test, y_pred, y_score, classes_pred)
