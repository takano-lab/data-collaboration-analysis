import numpy as np

from src.dimensionality_reduction import build_dimensionality_projector


class _Cfg:
    def __init__(self):
        self.seed = 3
        self.f_seed = 5
        self.ae2_epochs = 1
        self.ae2_batch = 8
        self.ae2_lr = 1e-3
        self.ae2_minmax = True


def test_build_projector_ae2_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16))
    cfg = _Cfg()
    proj = build_dimensionality_projector(
        X=X,
        n_components=6,
        F_type="ae2",
        seed=0,
        config=cfg,
    )
    Z = proj(X[:7])
    assert Z.shape == (7, 6)
