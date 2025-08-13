import numpy as np
import copy
from sklearn.metrics import roc_auc_score

class ScratchMLP:
    """NumPyベースのシンプルな1隠れ層MLPクラス"""
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = self._softmax(self.z2)
        return self.probs

    def backward(self, X, y):
        num_samples = X.shape[0]
        
        # Cross-Entropy LossとSoftmaxの勾配
        d_z2 = self.probs.copy()
        d_z2[range(num_samples), y] -= 1
        d_z2 /= num_samples

        # 重みとバイアスの勾配
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0) # ReLUの勾配

        d_W1 = X.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        return d_W1, d_b1, d_W2, d_b2

    def update(self, d_W1, d_b1, d_W2, d_b2, lr):
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params[0], params[1], params[2], params[3]

def run_federated_learning(
    clients_X_train: list[np.ndarray],
    clients_y_train: list[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: dict,
    logger
) -> float:
    """スクラッチ実装による連合学習の実行"""
    
    input_size = clients_X_train[0].shape[1]
    hidden_size = config.get("hidden_size", 256)
    n_rounds = config.get("rounds", 10)
    local_epochs = config.get("local_epochs", 5)
    lr = config.get("lr", 0.01)
    seed = config.get("seed", 42)
    
    # グローバルモデルの初期化
    global_model = ScratchMLP(input_size, hidden_size, n_classes, seed)
    
    final_auc = 0
    for round_idx in range(n_rounds):
        local_params_list = []
        
        # クライアントでの学習
        for client_X, client_y in zip(clients_X_train, clients_y_train):
            local_model = copy.deepcopy(global_model)
            
            for epoch in range(local_epochs):
                # ミニバッチなし（全データで学習）
                local_model.forward(client_X)
                d_W1, d_b1, d_W2, d_b2 = local_model.backward(client_X, client_y)
                local_model.update(d_W1, d_b1, d_W2, d_b2, lr)
            
            local_params_list.append(local_model.get_params())
            
        # サーバーでの集約 (FedAvg)
        avg_params = []
        for params in zip(*local_params_list):
            avg_params.append(np.mean(np.array(params), axis=0))
        
        global_model.set_params(avg_params)
        
        # グローバルモデルの評価
        y_score = global_model.forward(X_test)
        
        metric = getattr(config, 'metrics', 'auc').lower()
        
        if metric == "accuracy":
            metrics = global_model.score(X_test, y_test)
        elif metric == "auc":
            if n_classes == 2:
                metrics = roc_auc_score(y_test, y_score[:, 1])
            else:
                metrics = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")

        logger.info(f"Round {round_idx+1}/{n_rounds} | Global Model Test Metrics: {metrics:.4f}")
        final = metrics

    return final