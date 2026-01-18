import numpy as np

from src.integrated_expression import runners


def _rand(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape)


def test_build_nonlinear_new_projectors_matches_eigenproblem():
    # Small synthetic setup (no dataset dependency)
    c = 3
    r = 12
    d = 5
    dim = 4
    lam = 0.2

    anchors = [_rand(100 + i, (r, d)) for i in range(c)]
    Xs = [_rand(200 + i, (50, d)) for i in range(c)]

    projs, Z, eigvals, gammas = runners.build_nonlinear_new_projectors(
        anchors_inter=anchors,
        Xs_train_inter=Xs,
        dim_integrate=dim,
        gamma_type="fixed",
        gamma_ratio_krr=1.0,
        kernel_type="linear",
        K_normalization=False,
        nl_lambda=lam,
        zerosum=False,
    )

    assert len(projs) == c
    assert Z.shape == (r, dim)
    assert eigvals.shape == (dim,)
    assert len(gammas) == c

    I = np.eye(r)
    Ss = []
    for A in anchors:
        K = A @ A.T
        Ss.append(np.linalg.inv(K + lam * I))
    M_lambda = lam * sum(Ss)
    M_lambda = (M_lambda + M_lambda.T) * 0.5

    # Z columns should satisfy the eigen-equation approximately.
    residual = np.linalg.norm(M_lambda @ Z - Z @ np.diag(eigvals), ord="fro")
    assert residual < 1e-7

    # Projector output shape sanity
    X = _rand(999, (7, d))
    out = projs[0](X)
    assert out.shape == (7, dim)


def test_build_laplacian_nonlinear_new_projectors_trace_scaled():
    c = 2
    r = 10
    d = 4
    dim = 3
    lam = 0.1

    anchors = [_rand(300 + i, (r, d)) for i in range(c)]
    Xs = [_rand(400 + i, (40, d)) for i in range(c)]

    mu = 1.0
    k = 3
    projs, Z, eigvals, gammas = runners.build_laplacian_nonlinear_new_projectors(
        anchors_inter=anchors,
        Xs_train_inter=Xs,
        anchor=np.vstack(anchors),
        dim_integrate=dim,
        gamma_type="fixed",
        gamma_ratio_krr=1.0,
        nl_lambda=lam,
        kernel_type="linear",
        graph_mu_align=mu,
        laplacian_k=k,
        zerosum=False,
        regularization="graph",
        K_normalization=False,
    )

    assert len(projs) == c
    assert Z.shape == (r, dim)
    assert eigvals.shape == (dim,)
    assert len(gammas) == c

    I = np.eye(r)
    Ss = []
    for A in anchors:
        Kmat = A @ A.T
        Ss.append(np.linalg.inv(Kmat + lam * I))
    M_lambda = lam * sum(Ss)
    M_lambda = (M_lambda + M_lambda.T) * 0.5

    L = runners._build_unlabeled_anchor_laplacian(anchors, k_neighbors=k)
    tr_M = float(np.trace(M_lambda))
    tr_L = float(np.trace(L))
    scale = tr_M / max(tr_L, 1e-12) if tr_L > 0 else 1.0
    A_expected = M_lambda + mu * scale * L
    A_expected = (A_expected + A_expected.T) * 0.5

    residual = np.linalg.norm(A_expected @ Z - Z @ np.diag(eigvals), ord="fro")
    assert residual < 1e-7

