from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lobpcg
from sklearn.metrics.pairwise import rbf_kernel

from .runners import (
    _determine_kernel_gammas,
    _build_unlabeled_anchor_laplacian,
    _zerosum_helmert_basis,
    build_laplacian_nonlinear_projectors,
    make_kernel_integrator,
)


def _nystrom_factors_single(
    anchor_inter: np.ndarray,
    *,
    gamma: float,
    rank_nystrom: int,
    kernel_type: str,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Nyström factors U, lambda for a single institution's Gram matrix.

    Returns U (r x r_eff) with approximately orthonormal columns and eigenvalues
    lambda (r_eff,) so that K ≈ U diag(lambda) U^T.
    """
    r, _ = anchor_inter.shape
    if r == 0 or rank_nystrom <= 0:
        return np.zeros((r, 0)), np.zeros((0,))

    rank = min(int(rank_nystrom), r)
    kernel_type_key = (kernel_type or "rbf").lower()

    # If requested rank covers all anchors, fall back to exact Gram eigendecomposition.
    if rank >= r:
        if kernel_type_key == "linear":
            K = anchor_inter @ anchor_inter.T
        else:
            K = rbf_kernel(anchor_inter, anchor_inter, gamma=gamma)
        vals, vecs = np.linalg.eigh(K)
        eps = 1e-12
        vals = np.where(vals > eps, vals, 0.0)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        return vecs, vals

    # Proper Nyström with subsampling when rank < r.
    rng = np.random.default_rng(random_state)
    indices = np.arange(r)
    rng.shuffle(indices)
    landmark_idx = np.sort(indices[:rank])

    X_land = anchor_inter[landmark_idx]

    if kernel_type_key == "linear":
        C = anchor_inter @ X_land.T  # (r, r')
        W = X_land @ X_land.T
    else:
        C = rbf_kernel(anchor_inter, X_land, gamma=gamma)
        W = rbf_kernel(X_land, X_land, gamma=gamma)

    # Eigen-decomposition of the small W (r' x r')
    vals, vecs = np.linalg.eigh(W)
    eps = 1e-12
    mask = vals > eps
    if not np.any(mask):
        return np.zeros((r, 0)), np.zeros((0,))
    vals = vals[mask]
    vecs = vecs[:, mask]

    # Sort by descending eigenvalue and truncate
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    rank_eff = min(len(vals), rank)
    vals = vals[:rank_eff]
    vecs = vecs[:, :rank_eff]

    sqrt_vals = np.sqrt(vals)
    U = C @ vecs @ np.diag(1.0 / sqrt_vals)
    return U, vals


def _build_multi_view_operator(
    U_list: List[np.ndarray],
    D_list: List[np.ndarray],
    *,
    graph_L: Optional[csr_matrix] = None,
    graph_mu_align: float = 0.0,
) -> LinearOperator:
    """Build LinearOperator v -> (M_lambda + mu L) v using Nyström factors."""
    if not U_list:
        return LinearOperator((0, 0), matvec=lambda x: x, dtype=float)

    r = U_list[0].shape[0]
    mu = float(graph_mu_align)
    use_graph = graph_L is not None and mu > 0.0

    def matvec(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X_mat = X.reshape(-1, 1)
        else:
            X_mat = X
        if X_mat.shape[0] != r:
            X_mat = X_mat.T

        R = np.zeros_like(X_mat)
        for U, d in zip(U_list, D_list):
            if U.size == 0 or d.size == 0:
                continue
            # P v ≈ U D U^T v
            UtX = U.T @ X_mat  # (r', b)
            UtX_scaled = d[:, None] * UtX
            Pv = U @ UtX_scaled  # (r, b)

            # P(P v)
            UtPv = U.T @ Pv
            UtPv_scaled = d[:, None] * UtPv
            PPv = U @ UtPv_scaled

            R += X_mat - 2.0 * Pv + PPv

        if use_graph:
            R += mu * (graph_L @ X_mat)

        return R if X.ndim > 1 else R.ravel()

    return LinearOperator((r, r), matvec=matvec, dtype=float)


def _solve_lobpcg_eigenspace(
    A: LinearOperator,
    *,
    dim_integrate: int,
    zerosum: bool,
    tol: float,
    maxiter: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve min tr(Z^T A Z) s.t. Z^T Z = I using LOBPCG."""
    r = A.shape[0]
    if r == 0 or dim_integrate <= 0:
        return np.zeros((0,)), np.zeros((r, 0))

    rng = np.random.default_rng(random_state)
    X0 = rng.standard_normal((r, dim_integrate))

    if zerosum:
        one = np.ones((r, 1))
        proj = (one.T @ X0) / (one.T @ one)
        X0 = X0 - one @ proj

    eigvals, eigvecs = lobpcg(
        A,
        X0,
        largest=False,
        tol=float(tol),
        maxiter=int(maxiter),
    )

    order = np.argsort(eigvals)
    eigvals = np.asarray(eigvals[order], dtype=float)
    eigvecs = np.asarray(eigvecs[:, order], dtype=float)

    # Normalize columns
    for j in range(eigvecs.shape[1]):
        nz = np.linalg.norm(eigvecs[:, j])
        if nz > 0:
            eigvecs[:, j] /= nz

    take = min(dim_integrate, eigvecs.shape[1])
    return eigvals[:take], eigvecs[:, :take]


def build_nonlinear_projectors_faster(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    anchor: np.ndarray,
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    kernel_type: str = "rbf",
    graph_mu_align: float = 0.0,
    laplacian_k: int = 10,
    zerosum: bool = False,
    rank_nystrom: int = 200,
    lobpcg_tol: float = 1e-5,
    lobpcg_maxiter: int = 200,
    use_faiss_graph: bool = False,  # placeholder flag for future FAISS-based k-NN
    use_nystrom: bool = True,
    use_lobpcg: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """Variant of Laplacian-regularized nonlinear projectors with optional approximations.

    Branches (controlled by use_nystrom / use_lobpcg):
      - use_nystrom=True,  use_lobpcg=True  : Nyström + LOBPCG  (fully approximate, default).
      - use_nystrom=True,  use_lobpcg=False : Nyström only (eigh for eigenproblem).
      - use_nystrom=False, use_lobpcg=True  : Exact K/P, eigenproblem only approximated by LOBPCG.
      - use_nystrom=False, use_lobpcg=False : Fully exact; delegates to build_laplacian_nonlinear_projectors.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    # Fully exact branch: delegate to dense implementation.
    if not use_nystrom and not use_lobpcg:
        return build_laplacian_nonlinear_projectors(
            anchors_inter=anchors_inter,
            Xs_train_inter=Xs_train_inter,
            anchor=anchor,
            dim_integrate=dim_integrate,
            gamma_type=gamma_type,
            gamma_ratio_krr=gamma_ratio_krr,
            nl_lambda=nl_lambda,
            kernel_type=kernel_type,
            graph_mu_align=graph_mu_align,
            laplacian_k=laplacian_k,
            zerosum=zerosum,
        )

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    for A in anchors_inter:
        if A.shape[0] != r:
            raise ValueError("All anchor projections must share the same number of rows.")

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()
    I_r = np.eye(r)

    # --- Build either Nyström factors or exact Ks/Ps depending on use_nystrom ---

    U_list: List[np.ndarray] = []
    D_list: List[np.ndarray] = []
    Ks: List[np.ndarray] = []
    Ps: List[np.ndarray] = []

    if use_nystrom:
        for k, anchor_inter_k in enumerate(anchors_inter):
            U_k, lambdas_k = _nystrom_factors_single(
                anchor_inter_k,
                gamma=gammas[k],
                rank_nystrom=rank_nystrom,
                kernel_type=kernel_type_key,
                random_state=random_state,
            )
            if lambdas_k.size == 0:
                U_list.append(np.zeros((r, 0)))
                D_list.append(np.zeros((0,)))
                continue
            D_k = lambdas_k / (lambdas_k + float(nl_lambda))
            U_list.append(U_k)
            D_list.append(D_k)
    else:
        for k, anchor_inter_k in enumerate(anchors_inter):
            if kernel_type_key == "linear":
                K = anchor_inter_k @ anchor_inter_k.T
            else:
                K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[k])
            Ks.append(K)
            Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))

    # --- Dense M and Laplacian for branches that use eigh or exact A ---

    def _build_dense_M_from_nystrom() -> np.ndarray:
        M = np.zeros((r, r), dtype=float)
        for U_k, D_k in zip(U_list, D_list):
            if U_k.size == 0 or D_k.size == 0:
                continue
            U_scaled = U_k * D_k.reshape(1, -1)
            P = U_scaled @ U_k.T
            I_minus_P = I_r - P
            M += I_minus_P.T @ I_minus_P
        M = (M + M.T) * 0.5
        return M

    def _build_dense_M_from_exact() -> np.ndarray:
        M = np.zeros((r, r), dtype=float)
        for P in Ps:
            I_minus_P = I_r - P
            M += I_minus_P.T @ I_minus_P
        M = (M + M.T) * 0.5
        return M

    def _build_dense_A(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if graph_mu_align <= 0.0 or laplacian_k <= 0:
            return M, M
        L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=int(laplacian_k))
        if L_plain.shape != M.shape:
            return M, M
        L_sym = (L_plain + L_plain.T) * 0.5
        eps = 1e-12
        tr_M = float(np.trace(M))
        tr_L = float(np.trace(L_sym))
        scale_L = tr_M / max(tr_L, eps) if tr_L > 0 else 1.0
        A = M + float(graph_mu_align) * scale_L * L_sym
        A = (A + A.T) * 0.5
        return A, L_sym

    # --- Branch 1: Nyström + LOBPCG (original fast path) ---

    if use_nystrom and use_lobpcg:
        # Optional graph Laplacian via sparse operator
        graph_L_csr: Optional[csr_matrix] = None
        scale_L = 1.0
        if graph_mu_align > 0.0 and laplacian_k > 0:
            L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=int(laplacian_k))
            if L_plain.size > 0:
                graph_L_csr = csr_matrix((L_plain + L_plain.T) * 0.5)
                # Compute scale_L so that tr(M_lambda) ~= tr(scale_L * L)
                tr_M = 0.0
                for D_k in D_list:
                    m_k = int(D_k.size)
                    if m_k == 0:
                        tr_M += float(r)
                        continue
                    tr_M += float(np.sum((1.0 - D_k) ** 2) + (r - m_k))
                tr_L = float(graph_L_csr.diagonal().sum())
                eps = 1e-12
                if tr_L > eps and tr_M > 0.0:
                    scale_L = tr_M / tr_L
                else:
                    scale_L = 1.0

        A_op = _build_multi_view_operator(
            U_list,
            D_list,
            graph_L=graph_L_csr,
            graph_mu_align=graph_mu_align * scale_L,
        )

        eigvals, Z_integ = _solve_lobpcg_eigenspace(
            A_op,
            dim_integrate=dim_integrate,
            zerosum=zerosum,
            tol=lobpcg_tol,
            maxiter=lobpcg_maxiter,
            random_state=random_state,
        )

        projs: List[Callable[[np.ndarray], np.ndarray]] = []
        lam = float(nl_lambda)

        for k, (U_k, D_k) in enumerate(zip(U_list, D_list)):
            if U_k.size == 0 or D_k.size == 0:
                B_k = np.zeros((r, Z_integ.shape[1]), dtype=float)
            else:
                # Recover approximate eigenvalues of K from eigenvalues of P:
                # D_k = lambda / (lambda + lam)  =>  lambda = D_k * lam / (1 - D_k)
                eps = 1e-12
                lambdas_k = (D_k * lam) / (1.0 - D_k + eps)
                UtZ = U_k.T @ Z_integ
                inv_diag = 1.0 / (lambdas_k + lam)
                term1 = U_k @ (inv_diag[:, None] * UtZ)
                term2 = (1.0 / lam) * (Z_integ - U_k @ UtZ)
                B_k = term1 + term2

            proj = make_kernel_integrator(
                anchors_inter[k],
                B_k,
                gamma=gammas[k],
                kernel_type=kernel_type_key,
            )
            projs.append(proj)

        return projs, Z_integ, eigvals, gammas

    # --- Branch 2: Nyström only (dense eig) ---

    if use_nystrom and not use_lobpcg:
        M = _build_dense_M_from_nystrom()
        A_dense, _ = _build_dense_A(M)

        if zerosum and r >= 2:
            B_zero = _zerosum_helmert_basis(A_dense.shape[0])
            A_tilde = B_zero.T @ A_dense @ B_zero
            eigvals_raw, eigvecs_sub = eigh(A_tilde, np.eye(A_tilde.shape[0]))
            order = np.argsort(eigvals_raw)
            take = min(dim_integrate, eigvecs_sub.shape[1])
            select = order[:take]
            eigvals = eigvals_raw[select]
            eigvecs = eigvecs_sub[:, select]
            Z_integ = B_zero @ eigvecs
        else:
            eigvals_raw, eigvecs = eigh(A_dense, np.eye(A_dense.shape[0]))
            order = np.argsort(eigvals_raw)
            take = min(dim_integrate, eigvecs.shape[1])
            select = order[:take]
            eigvals = eigvals_raw[select]
            Z_integ = eigvecs[:, select]

        # Normalize columns
        for j in range(Z_integ.shape[1]):
            nz = np.linalg.norm(Z_integ[:, j])
            if nz > 0:
                Z_integ[:, j] /= nz

        projs: List[Callable[[np.ndarray], np.ndarray]] = []
        lam = float(nl_lambda)
        for k, (U_k, D_k) in enumerate(zip(U_list, D_list)):
            if U_k.size == 0 or D_k.size == 0:
                B_k = np.zeros((r, Z_integ.shape[1]), dtype=float)
            else:
                eps = 1e-12
                lambdas_k = (D_k * lam) / (1.0 - D_k + eps)
                UtZ = U_k.T @ Z_integ
                inv_diag = 1.0 / (lambdas_k + lam)
                term1 = U_k @ (inv_diag[:, None] * UtZ)
                term2 = (1.0 / lam) * (Z_integ - U_k @ UtZ)
                B_k = term1 + term2

            proj = make_kernel_integrator(
                anchors_inter[k],
                B_k,
                gamma=gammas[k],
                kernel_type=kernel_type_key,
            )
            projs.append(proj)

        return projs, Z_integ, eigvals, gammas

    # --- Branch 3: Exact K/P + LOBPCG (eigen-solver only approximate) ---

    M = _build_dense_M_from_exact()
    A_dense, _ = _build_dense_A(M)
    A_csr = csr_matrix(A_dense)

    def matvec_A(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (A_csr @ x).astype(float)

    A_op = LinearOperator((r, r), matvec=matvec_A, dtype=float)

    eigvals, Z_integ = _solve_lobpcg_eigenspace(
        A_op,
        dim_integrate=dim_integrate,
        zerosum=zerosum,
        tol=lobpcg_tol,
        maxiter=lobpcg_maxiter,
        random_state=random_state,
    )

    # Projectors as in dense baseline
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for k, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(
            anchors_inter[k],
            B_k,
            gamma=gammas[k],
            kernel_type=kernel_type_key,
        )
        projs.append(proj)

    return projs, Z_integ, eigvals, gammas
