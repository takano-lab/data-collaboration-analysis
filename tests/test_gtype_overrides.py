from config.config import Config
from main import _apply_runtime_gtype_overrides


def test_laplacian_nonlinear_tg_override_uses_identity_constraint():
    cfg = Config(G_type="laplacian_nonlinear_tg", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "laplacian_nonlinear_tg"
    assert cfg.G_type == "laplacian_nonlinear_new"
    assert cfg.zerosum is False
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "identity"


def test_laplacian_nonlinear_zero_tg_override_uses_identity_constraint():
    cfg = Config(G_type="laplacian_nonlinear_zero_tg", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "laplacian_nonlinear_zero_tg"
    assert cfg.G_type == "laplacian_nonlinear_new"
    assert cfg.zerosum is True
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "identity"


def test_laplacian_nonlinear_penal_override():
    cfg = Config(G_type="laplacian_nonlinear_penal", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "laplacian_nonlinear_penal"
    assert cfg.G_type == "laplacian_nonlinear_new"
    assert cfg.zerosum is False
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "between"


def test_laplacian_nonlinear_zero_penal_override():
    cfg = Config(G_type="laplacian_nonlinear_zero_penal", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "laplacian_nonlinear_zero_penal"
    assert cfg.G_type == "laplacian_nonlinear_new"
    assert cfg.zerosum is True
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "between"


def test_gtype_override_strips_accidental_trailing_quotes():
    cfg = Config(G_type='laplacian_nonlinear_zero_penal""', target_graph_constraint="identity")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "laplacian_nonlinear_zero_penal"
    assert cfg.G_type == "laplacian_nonlinear_new"
    assert cfg.zerosum is True
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "between"


def test_targetvec_tg_override_uses_targetvec_singular_identity_constraint():
    cfg = Config(G_type="targetvec_tg", regularization="graph", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "targetvec_tg"
    assert cfg.G_type == "targetvec_singular"
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "identity"


def test_targetvec_graph_override_uses_targetvec_singular_identity_constraint():
    cfg = Config(G_type="targetvec_graph", regularization="graph", target_graph_constraint="between")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "targetvec_graph"
    assert cfg.G_type == "targetvec_singular"
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "identity"


def test_targetvec_penal_override_uses_targetvec_singular_between_constraint():
    cfg = Config(G_type="targetvec_penal", regularization="graph", target_graph_constraint="identity")

    original = _apply_runtime_gtype_overrides(cfg)

    assert original == "targetvec_penal"
    assert cfg.G_type == "targetvec_singular"
    assert cfg.regularization == "target-graph"
    assert cfg.target_graph_constraint == "between"
