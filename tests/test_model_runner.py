import numpy as np

from config.config import Config
from src.model import ModelRunner


def test_logistic_regression_alias_runs_and_predicts_proba():
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.1],
            [1.2, 0.9],
            [2.0, 2.1],
            [2.2, 1.8],
        ]
    )
    y_train = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    X_test = np.array([[0.05, 0.1], [1.1, 1.0], [2.1, 2.0]])
    y_test = np.array(["a", "b", "c"], dtype=object)

    cfg = Config(h_model="logistic_regression", seed=0, metrics="accuracy")
    runner = ModelRunner(cfg)

    score = runner.run(X_train, y_train, X_test, y_test)
    assert 0.0 <= score <= 1.0

    y_pred, y_proba, classes = runner.predict_with_proba(X_train, y_train, X_test)
    assert y_pred.shape == (3,)
    assert y_proba.shape[0] == 3
    assert classes.shape[0] == y_proba.shape[1]


def test_mlp_uses_regressor_for_regression_task():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 6))
    coef = np.array([1.5, -2.0, 0.7, 0.0, 1.0, -0.3])
    y_train = X_train @ coef + 0.1 * rng.normal(size=80)
    X_test = rng.normal(size=(20, 6))
    y_test = X_test @ coef + 0.1 * rng.normal(size=20)

    cfg = Config(h_model="mlp", seed=0, metrics="rmse", task="regression")
    runner = ModelRunner(cfg)

    score = runner.run(X_train, y_train, X_test, y_test)

    assert np.isfinite(score)
    assert score >= 0.0
