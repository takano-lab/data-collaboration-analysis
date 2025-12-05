from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from config.config import Config


class ScratchMLP:
    """Minimal NumPy-based MLP used for the FL baseline."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42) -> None:
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = self._softmax(self.z2)
        return self.probs

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = X.shape[0]

        d_z2 = self.probs.copy()
        d_z2[np.arange(num_samples), y] -= 1
        d_z2 /= num_samples

        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0)

        d_W1 = X.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        return d_W1, d_b1, d_W2, d_b2

    def update(self, d_W1: np.ndarray, d_b1: np.ndarray, d_W2: np.ndarray, d_b2: np.ndarray, lr: float) -> None:
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

    def get_params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params


def run_federated_learning(
    clients_X_train: list[np.ndarray],
    clients_y_train: list[np.ndarray],
    n_classes: int,
    config: Config | dict,
    logger,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> ScratchMLP:
    """
    Train the scratch MLP with FedAvg. Optionally evaluates on shared test data when provided.
    Returns the final global model so that each institution can perform its own inference.
    """

    def _cfg_get(key: str, default):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _debug_report_nan(name: str, array: Optional[np.ndarray]) -> None:
        if array is None or array.size == 0:
            return
        nan_mask = np.isnan(array)
        if not nan_mask.any():
            return
        total = int(nan_mask.sum())
        msg_total = f"[nan-debug] {name}: detected {total} NaN values."
        try:
            logger.warning(msg_total)
        except Exception:
            logger.error(msg_total)
        per_feature = nan_mask.sum(axis=0)
        bad_cols = np.where(per_feature > 0)[0]
        for col in bad_cols:
            count = int(per_feature[col])
            sample_ids = np.where(nan_mask[:, col])[0][:5]
            msg_col = (
                f"[nan-debug] {name}: feature #{col} has {count} NaNs "
                f"(sample indices: {sample_ids.tolist()})"
            )
            try:
                logger.warning(msg_col)
            except Exception:
                logger.error(msg_col)

    for idx, client_X in enumerate(clients_X_train):
        _debug_report_nan(f"client[{idx}]_X_train", client_X)
    _debug_report_nan("X_test", X_test)

    input_size = clients_X_train[0].shape[1]
    hidden_size = _cfg_get("hidden_size", 256)
    n_rounds = _cfg_get("rounds", 10)
    local_epochs = _cfg_get("local_epochs", 5)
    lr = _cfg_get("lr", 0.01)
    seed = _cfg_get("seed", 42)

    evaluate_model = X_test is not None and y_test is not None
    metric_name = str(_cfg_get("metrics", "auc")).lower()
    final_metric = None

    global_model = ScratchMLP(input_size, hidden_size, n_classes, seed)

    for round_idx in range(n_rounds):
        local_params_list = []

        for client_X, client_y in zip(clients_X_train, clients_y_train):
            local_model = copy.deepcopy(global_model)

            for _ in range(local_epochs):
                local_model.forward(client_X)
                d_W1, d_b1, d_W2, d_b2 = local_model.backward(client_X, client_y)
                local_model.update(d_W1, d_b1, d_W2, d_b2, lr)

            local_params_list.append(local_model.get_params())

        avg_params = [np.mean(np.array(params), axis=0) for params in zip(*local_params_list)]
        global_model.set_params(avg_params)

        if evaluate_model:
            try:
                y_score = global_model.forward(X_test)  # type: ignore[arg-type]
                if metric_name == "accuracy":
                    y_pred = np.argmax(y_score, axis=1)
                    metrics = float(np.mean(y_pred == y_test))  # type: ignore[arg-type]
                elif metric_name == "auc":
                    if n_classes == 2:
                        metrics = roc_auc_score(y_test, y_score[:, 1])  # type: ignore[arg-type]
                    else:
                        metrics = roc_auc_score(  # type: ignore[arg-type]
                            y_test,
                            y_score,
                            multi_class="ovr",
                            average="macro",
                        )
                else:
                    metrics = float("nan")
                final_metric = metrics
                logger.info(
                    f"Round {round_idx + 1}/{n_rounds} | Global Model Test Metrics: {metrics:.4f}"
                )
            except ValueError as exc:
                logger.warning(
                    f"Round {round_idx + 1}/{n_rounds} | Failed to compute metrics: {exc}"
                )

    if evaluate_model and final_metric is None:
        logger.warning("Federated learning finished without a valid evaluation metric.")

    return global_model
