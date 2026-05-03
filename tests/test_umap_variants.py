import sys
import types

import numpy as np

from config.config import Config
from src.dimensionality_reduction import build_dimensionality_projector


class _FakeUMAP:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeUMAP.calls.append(kwargs)

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        return np.zeros((X.shape[0], self.kwargs["n_components"]))


def _install_fake_umap(monkeypatch):
    fake_module = types.ModuleType("umap")
    fake_module.UMAP = _FakeUMAP
    _FakeUMAP.calls = []
    monkeypatch.setitem(sys.modules, "umap", fake_module)


def test_umap_2_uses_narrower_parameter_ranges(monkeypatch):
    _install_fake_umap(monkeypatch)
    X = np.arange(200, dtype=float).reshape(25, 8)
    cfg = Config(seed=0, f_seed=0)

    projector = build_dimensionality_projector(X, 3, F_type="umap_2", seed=0, config=cfg)
    Z = projector(X[:2])

    assert Z.shape == (2, 3)
    params = _FakeUMAP.calls[-1]
    assert params["metric"] in {"correlation", "cosine", "euclidean"}
    assert 8 <= params["n_neighbors"] <= 15
    assert 0.05 <= params["min_dist"] < 0.3


def test_umap_keeps_original_parameter_ranges(monkeypatch):
    _install_fake_umap(monkeypatch)
    X = np.arange(200, dtype=float).reshape(25, 8)
    cfg = Config(seed=0, f_seed=0)

    build_dimensionality_projector(X, 3, F_type="umap", seed=0, config=cfg)

    params = _FakeUMAP.calls[-1]
    assert params["metric"] in {"correlation", "cosine", "euclidean"}
    assert 2 <= params["n_neighbors"] <= 7
    assert 0.0 <= params["min_dist"] < 0.8
