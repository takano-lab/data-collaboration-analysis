import numpy as np


from src.intermediate_expression.anchor_utils import build_laplacians_from_anchor_labels


def _laplacian_to_adjacency(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    W = -L.copy()
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0
    return (W > 0).astype(int)


def test_build_laplacians_from_anchor_labels_within_and_penalty_photo_style():
    # Two clearly separated classes in 1D so nearest neighbors are deterministic.
    anchor = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # k=1 for within-class neighbors and k=1 for penalty pairs (per class).
    Lw, Lb = build_laplacians_from_anchor_labels(anchor=anchor, anchor_y=y, k_neighbors=1)
    assert Lw is not None and Lb is not None
    assert Lw.shape == (6, 6)
    assert Lb.shape == (6, 6)

    Aw = _laplacian_to_adjacency(Lw)
    Ab = _laplacian_to_adjacency(Lb)

    # Within-class: should only connect within each class (no cross edges).
    assert Aw[:3, 3:].sum() == 0
    assert Aw[3:, :3].sum() == 0
    assert Aw[:3, :3].sum() > 0
    assert Aw[3:, 3:].sum() > 0

    # Penalty: selects the closest cross-class pair(s). Here the closest is (0.2, 10.0).
    # Because k=1 per class, both classes pick that same edge.
    assert Ab[2, 3] == 1
    assert Ab[3, 2] == 1
    # No within-class edges in penalty graph.
    assert Ab[:3, :3].sum() == 0
    assert Ab[3:, 3:].sum() == 0


def test_build_laplacians_from_anchor_labels_ignores_invalid_labels():
    anchor = np.array([[0.0], [0.1], [0.2], [10.0]])
    y = np.array([0, 0, 1, np.nan])

    Lw, Lb = build_laplacians_from_anchor_labels(anchor=anchor, anchor_y=y, k_neighbors=1)
    assert Lw is not None and Lb is not None

    # Invalid label node becomes isolated => zero row/col in Laplacians (up to normalization).
    assert np.allclose(Lw[3, :], 0.0)
    assert np.allclose(Lw[:, 3], 0.0)
    assert np.allclose(Lb[3, :], 0.0)
    assert np.allclose(Lb[:, 3], 0.0)

