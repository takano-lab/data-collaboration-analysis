import numpy as np


from src.intermediate_expression.anchor_utils import (
    assign_anchor_regression_targets,
    build_laplacians_from_anchor_labels,
    build_laplacians_from_anchor_regression_targets,
)


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


def test_assign_anchor_regression_targets_uses_knn_weighted_average():
    X_train = [np.array([[0.0], [1.0], [2.0], [3.0]])]
    y_train = [np.array([0.0, 10.0, 20.0, 30.0])]
    anchors = [np.array([[0.1], [2.9]])]
    anchors_test = [np.array([[1.1]])]

    y_anchor, y_anchor_test = assign_anchor_regression_targets(
        anchors_inter=anchors,
        anchors_test_inter=anchors_test,
        Xs_train_inter=X_train,
        ys_train=y_train,
        k=1,
    )

    assert y_anchor.shape == (2,)
    assert y_anchor_test.shape == (1,)
    assert np.allclose(y_anchor, [0.0, 30.0])
    assert np.allclose(y_anchor_test, [10.0])


def test_build_laplacians_from_anchor_regression_targets_continuous_weights():
    anchor = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0.0, 0.05, 10.0, 10.1])

    Lw, Lb = build_laplacians_from_anchor_regression_targets(
        anchor=anchor,
        anchor_y=y,
        k_neighbors=2,
        sigma_x=1.0,
        sigma_y=1.0,
        normalize=False,
    )

    assert Lw is not None and Lb is not None
    assert Lw.shape == (4, 4)
    assert Lb.shape == (4, 4)
    assert np.allclose(Lw, Lw.T)
    assert np.allclose(Lb, Lb.T)
    assert np.allclose(Lw.sum(axis=1), 0.0)
    assert np.allclose(Lb.sum(axis=1), 0.0)

    Ww = -Lw.copy()
    Wb = -Lb.copy()
    np.fill_diagonal(Ww, 0.0)
    np.fill_diagonal(Wb, 0.0)
    expected_close = np.exp(-((y[0] - y[1]) ** 2))
    expected_far = np.exp(-((y[0] - y[2]) ** 2))
    assert np.isclose(Ww[0, 1], expected_close)
    assert np.isclose(Wb[0, 1], 1.0 - expected_close)
    assert np.isclose(Ww[0, 2], expected_far)
    assert np.isclose(Wb[0, 2], 1.0 - expected_far)
    assert Ww[0, 1] > Ww[0, 2]
    assert Wb[0, 2] > Wb[0, 1]
