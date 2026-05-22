import numpy as np

from src.integrated_expression import runners


def _rand(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape)


def test_smallest_eigh_matches_full_standard_and_generalized():
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(16, 16))
    A = (X + X.T) * 0.5
    k = 5

    vals, vecs = runners._smallest_eigh(A, k)
    vals_full, _ = runners.eigh(A)

    assert vals.shape == (k,)
    assert vecs.shape == (16, k)
    assert np.allclose(vals, vals_full[:k], atol=1e-10)
    assert np.linalg.norm(A @ vecs - vecs @ np.diag(vals), ord="fro") < 1e-9

    Y = rng.standard_normal(size=(16, 16))
    B = Y.T @ Y + np.eye(16)
    vals_g, vecs_g = runners._smallest_eigh(A, k, B=B)
    vals_g_full, _ = runners.eigh(A, B)

    assert vals_g.shape == (k,)
    assert vecs_g.shape == (16, k)
    assert np.allclose(vals_g, vals_g_full[:k], atol=1e-10)
    assert np.linalg.norm(A @ vecs_g - B @ vecs_g @ np.diag(vals_g), ord="fro") < 1e-9


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


def test_build_laplacian_nonlinear_new_target_graph_requires_l_matrices():
    anchors_inter = [np.array([[0.0], [1.0], [2.0]])]
    Xs_train_inter = [np.array([[0.0], [1.0], [2.0]])]
    anchor = np.array([[0.0], [1.0], [2.0]])

    Lw = np.eye(3)
    Lb = np.eye(3)

    projs, Z, eigvals, gammas = runners.build_laplacian_nonlinear_new_projectors(
        anchors_inter=anchors_inter,
        Xs_train_inter=Xs_train_inter,
        anchor=anchor,
        dim_integrate=2,
        nl_lambda=1.0,
        graph_mu_align=1.0,
        laplacian_k=1,
        regularization="target-graph",
        L_within=Lw,
        L_between=Lb,
    )
    assert len(projs) == 1
    assert Z.shape == (3, 2)
    assert eigvals.shape == (2,)
    assert len(gammas) == 1

    try:
        runners.build_laplacian_nonlinear_new_projectors(
            anchors_inter=anchors_inter,
            Xs_train_inter=Xs_train_inter,
            anchor=anchor,
            dim_integrate=2,
            nl_lambda=1.0,
            graph_mu_align=1.0,
            laplacian_k=1,
            regularization="target-graph",
            L_within=Lw,
            L_between=None,
        )
        assert False, "Expected ValueError when L_between is missing for target-graph"
    except ValueError:
        pass


def test_build_nonlinear_mlp_projectors_uses_common_fixed_u_target():
    c = 2
    r = 8
    d = 4
    dim = 3

    anchors = [_rand(500 + i, (r, d)) for i in range(c)]
    concat = np.hstack(anchors)
    U, S, _ = np.linalg.svd(concat, full_matrices=False)

    projs, Z, eigvals, losses = runners.build_nonlinear_mlp_projectors(
        anchors_inter=anchors,
        dim_integrate=dim,
        hidden_dims=[16],
        mlp_lambda=1e-4,
        epochs=20,
        lr=1e-2,
        seed=7,
    )

    assert len(projs) == c
    assert Z.shape == (r, dim)
    assert eigvals.shape == (dim,)
    assert len(losses) == c
    assert np.allclose(Z, U[:, :dim], atol=1e-6)
    assert np.allclose(eigvals, S[:dim], atol=1e-6)

    out = projs[0](_rand(999, (5, d)))
    assert out.shape == (5, dim)
    row_norms = np.linalg.norm(out, axis=1)
    assert np.allclose(row_norms, np.ones_like(row_norms), atol=1e-5)


def test_build_nonlinear_imakura_z_projectors_fixes_z_by_anchor_svd():
    c = 3
    r = 9
    d = 4
    dim = 5
    lam = 1e-2

    anchors = [_rand(700 + i, (r, d)) for i in range(c)]
    Xs = [_rand(800 + i, (20, d)) for i in range(c)]
    W = np.hstack(anchors)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    U_ref = U[:, :dim]

    projs, Z, eigvals, gammas = runners.build_nonlinear_imakura_Z_projectors(
        anchors_inter=anchors,
        Xs_train_inter=Xs,
        dim_integrate=dim,
        kernel_type="linear",
        nl_lambda=lam,
        gamma_type="fixed",
        gamma_ratio_krr=1.0,
    )

    assert len(projs) == c
    assert len(gammas) == c
    assert Z.shape == (r, dim)
    assert eigvals.shape == (dim,)
    assert np.allclose(eigvals, S[:dim], atol=1e-8)

    # Column signs are ambiguous, so compare subspaces via projection matrices.
    P_ref = U_ref @ U_ref.T
    P = Z @ Z.T
    assert np.allclose(P, P_ref, atol=1e-6)

    out = projs[0](_rand(999, (7, d)))
    assert out.shape == (7, dim)
