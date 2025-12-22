from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from config.config import Config


class ScratchMLP:
    """
    NumPy-based MLP that mirrors ModelRunner._run_mlp:
    single hidden layer (256 units), ReLU -> softmax, Adam optimizer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
        lr: float = 1e-3,
        l2: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        rng = np.random.default_rng(seed)
        # He initialization for ReLU
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

        # Adam states
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.t = 0

        self.lr = lr
        self.l2 = l2
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(z)
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

        if self.l2 > 0:
            d_W1 += self.l2 * self.W1
            d_W2 += self.l2 * self.W2

        return d_W1, d_b1, d_W2, d_b2

    def update(self, d_W1: np.ndarray, d_b1: np.ndarray, d_W2: np.ndarray, d_b2: np.ndarray) -> None:
        self.t += 1

        def _adam_step(param, grad, m, v):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return param, m, v

        self.W1, self.m_W1, self.v_W1 = _adam_step(self.W1, d_W1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = _adam_step(self.b1, d_b1, self.m_b1, self.v_b1)
        self.W2, self.m_W2, self.v_W2 = _adam_step(self.W2, d_W2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = _adam_step(self.b2, d_b2, self.m_b2, self.v_b2)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> float:
        idx = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        total_loss = 0.0
        n_batches = 0
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            probs = self.forward(xb)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-12, 1.0)
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb]))
            total_loss += loss
            n_batches += 1
            grads = self.backward(xb, yb)
            self.update(*grads)
        return total_loss / max(n_batches, 1)

    def get_params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params
        # Reset optimizer states after aggregation to keep rounds consistent
        self.m_W1.fill(0.0)
        self.v_W1.fill(0.0)
        self.m_b1.fill(0.0)
        self.v_b1.fill(0.0)
        self.m_W2.fill(0.0)
        self.v_W2.fill(0.0)
        self.m_b2.fill(0.0)
        self.v_b2.fill(0.0)
        self.t = 0


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
    lr = _cfg_get("lr", 0.001)
    seed = _cfg_get("seed", 42)
    l2 = _cfg_get("l2", 1e-4)
    batch_size = _cfg_get("batch_size", 64)

    evaluate_model = X_test is not None and y_test is not None
    metric_name = str(_cfg_get("metrics", "auc")).lower()
    final_metric = None

    global_model = ScratchMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=n_classes,
        seed=seed,
        lr=lr,
        l2=l2,
    )

    for round_idx in range(n_rounds):
        local_params_list = []
        sample_counts: list[int] = []

        for client_X, client_y in zip(clients_X_train, clients_y_train):
            local_model = copy.deepcopy(global_model)
            sample_counts.append(client_X.shape[0])
            for _ in range(local_epochs):
                local_model.train_epoch(client_X, client_y, batch_size=batch_size, shuffle=True)
            local_params_list.append(local_model.get_params())

        # FedAvg with sample-size weighting
        total_samples = float(np.sum(sample_counts))
        weights = [cnt / total_samples for cnt in sample_counts]
        avg_params = []
        for params_per_layer in zip(*local_params_list):
            stacked = np.stack(params_per_layer)
            weighted = np.tensordot(weights, stacked, axes=1)
            avg_params.append(weighted)
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
