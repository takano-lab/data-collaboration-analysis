from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import block_diag, eigh, solve_triangular
from scipy.sparse.linalg import eigsh, svds
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

from src.dimensionality_reduction import self_tuning_gamma
from src.intermediate_expression.anchor_utils import _symmetric_knn_graph

# --- Basic projector factories ---

def make_linear_integrator(G_k: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return projector X -> X @ G_k (linear right-multiplication)."""
    def projector(X: np.ndarray) -> np.ndarray:
        return X @ G_k
    return projector


def make_centered_linear_integrator(
    G_k: np.ndarray,
    mu_k: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return projector X -> (X - mu_k) @ G_k with row-wise centering."""
    mu_k = np.asarray(mu_k, dtype=float).reshape(1, -1)

    def projector(X: np.ndarray) -> np.ndarray:
        return (X - mu_k) @ G_k

    return projector


def make_kernel_integrator(
    S_train: np.ndarray,
    B_k: np.ndarray,
    *,
    gamma: float,
    normalize: bool = False,
    mu_max: Optional[float] = None,
    kernel_type: str = "rbf",
) -> Callable[[np.ndarray], np.ndarray]:
    """Return kernel projector X -> K(X, S_train) @ B_k with optional spectral normalization.

    kernel_type: "rbf" (default) or "linear".
    """
    kernel_type_key = (kernel_type or "rbf").lower()

    def projector(X: np.ndarray) -> np.ndarray:
        if kernel_type_key == "linear":
            K_x = X @ S_train.T
        else:
            K_x = rbf_kernel(X, S_train, gamma=gamma)
        if normalize and (mu_max is not None) and mu_max > 0:
            K_x = K_x / mu_max
        return K_x @ B_k

    return projector


def _median_heuristic_gamma(X: np.ndarray, *, max_samples: int = 2000) -> float:
    """Compute gamma via median heuristic with optional subsampling for efficiency."""
    n_samples = X.shape[0]
    if n_samples == 0:
        return 1.0
    if n_samples > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_samples, max_samples, replace=False)
        X_ref = X[idx]
    else:
        X_ref = X
    D = pairwise_distances(X_ref, metric="euclidean")
    triu = D[np.triu_indices_from(D, k=1)]
    positive = triu[triu > 0]
    if positive.size == 0:
        return 1.0 / max(X.shape[1], 1)
    med = np.median(positive)
    if not np.isfinite(med) or med <= 0:
        return 1.0 / max(X.shape[1], 1)
    return 1.0 / (2.0 * (med ** 2))


def _determine_kernel_gammas(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    gamma_type: str,
    gamma_ratio_krr: float,
) -> List[float]:
    if not anchors_inter:
        return []
    n_inst = len(anchors_inter)
    gammas: List[float] = []
    gamma_type_key = (gamma_type or "").lower()

    if gamma_type_key == "auto":
        gammas = [1.0 / max(anchor.shape[1], 1) for anchor in anchors_inter]
    elif gamma_type_key == "x_tuning" and len(Xs_train_inter) == n_inst:
        for X_tr in Xs_train_inter:
            gamma = self_tuning_gamma(X_tr, standardize=False, k=3, summary="median")
            gammas.append(float(gamma) * gamma_ratio_krr)
    elif gamma_type_key == "median" and len(Xs_train_inter) == n_inst:
        for X_tr in Xs_train_inter:
            gamma = _median_heuristic_gamma(X_tr)
            gammas.append(float(gamma) * gamma_ratio_krr)
    elif gamma_type_key == "fixed":
        gammas = [float(gamma_ratio_krr)] * n_inst
    else:
        gammas = [1.0 / max(anchor.shape[1], 1) for anchor in anchors_inter]

    if len(gammas) != n_inst:
        gammas = [1.0 / max(anchor.shape[1], 1) for anchor in anchors_inter]
    return gammas


def _validate_anchor_rows(anchors_inter: List[np.ndarray]) -> int:
    if not anchors_inter:
        return 0
    row_count = anchors_inter[0].shape[0]
    for anchor in anchors_inter:
        if anchor.shape[0] != row_count:
            raise ValueError("All anchor projections must share the same number of rows for kernel GEP.")
    return row_count


def _effective_rank(K: np.ndarray, *, eps: float = 1e-12) -> float:
    """Compute entropy-based effective rank of a positive semidefinite matrix.

    r_eff = exp( - sum p_i log p_i ),  p_i = lambda_i / sum(lambda_i).
    """
    if K.size == 0:
        return 0.0
    vals = np.linalg.eigvalsh(K)
    vals = np.maximum(vals, 0.0)
    s = float(np.sum(vals))
    if s <= eps:
        return 0.0
    p = vals / s
    # Avoid log(0)
    p = np.clip(p, eps, 1.0)
    H = float(-np.sum(p * np.log(p)))
    return float(np.exp(H))


def _top_svd(
    M: np.ndarray,
    rank: int,
    *,
    truncated: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute leading singular triplets of M.

    When `truncated` is True and rank < min(m, n), use scipy.sparse.linalg.svds
    to obtain the leading `rank` singular vectors; otherwise fall back to full
    SVD via numpy.linalg.svd.
    """
    m, n = M.shape
    k = min(rank, m, n)
    if k <= 0 or M.size == 0:
        return np.zeros((m, 0)), np.zeros((0,)), np.zeros((0, n))

    if not truncated or k >= min(m, n):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        return U[:, :k], S[:k], Vt[:k, :]

    U, S, Vt = svds(M, k=k)
    order = np.argsort(S)[::-1]
    S = S[order]
    U = U[:, order]
    Vt = Vt[order, :]
    return U, S, Vt


def _top_eig_symmetric(
    A: np.ndarray,
    k: int,
    *,
    smallest: bool = True,
    truncated: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k eigenpairs of a symmetric matrix A.

    When `truncated` is True and k < n, use eigsh; otherwise fall back to full
    eigh. Returns (eigvals, eigvecs), where eigvals are sorted ascending if
    `smallest` is True, descending otherwise.
    """
    n = A.shape[0]
    k_eff = min(max(k, 0), n)
    if k_eff == 0:
        return np.zeros((0,)), np.zeros((n, 0))

    if not truncated or k_eff >= n:
        eigvals, eigvecs = np.linalg.eigh(A)
        if smallest:
            idx = slice(0, k_eff)
        else:
            idx = slice(n - k_eff, n)
        eigvals_sel = eigvals[idx]
        eigvecs_sel = eigvecs[:, idx]
        return eigvals_sel, eigvecs_sel

    which = "SA" if smallest else "LA"
    eigvals_sel, eigvecs_sel = eigsh(A, k=k_eff, which=which)
    order = np.argsort(eigvals_sel)
    if not smallest:
        order = order[::-1]
    eigvals_sel = eigvals_sel[order]
    eigvecs_sel = eigvecs_sel[:, order]
    return eigvals_sel, eigvecs_sel


def _smallest_eigh(
    A: np.ndarray,
    k: int,
    *,
    B: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k smallest eigenpairs of a dense symmetric problem.

    scipy.linalg.eigh can ask LAPACK for only a selected index range. That keeps
    the nonlinear integration step from computing every eigenvector when only
    dim_integrate columns are used.
    """
    n = A.shape[0]
    k_eff = min(max(int(k), 0), n)
    if k_eff == 0:
        return np.zeros((0,)), np.zeros((n, 0))

    kwargs: Dict[str, Any] = {"check_finite": False}
    if k_eff < n:
        kwargs["subset_by_index"] = [0, k_eff - 1]

    try:
        if B is None:
            eigvals, eigvecs = eigh(A, **kwargs)
        else:
            eigvals, eigvecs = eigh(A, B, **kwargs)
    except (TypeError, ValueError):
        # Older SciPy builds may not support subset_by_index for the selected
        # driver. Fall back to the full decomposition for compatibility.
        if B is None:
            eigvals, eigvecs = eigh(A, check_finite=False)
        else:
            eigvals, eigvecs = eigh(A, B, check_finite=False)
        order = np.argsort(eigvals)[:k_eff]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    return eigvals[:k_eff], eigvecs[:, :k_eff]


def _build_anchor_alignment_terms(Ks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not Ks:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    C_blocks = [K @ K for K in Ks]
    C = block_diag(*C_blocks)
    C_H = block_diag(*Ks)
    T_rows = []
    for k, Kk in enumerate(Ks):
        row_blocks = [Kk @ Ks[kp] for kp in range(len(Ks))]
        T_rows.append(np.hstack(row_blocks))
    T = np.vstack(T_rows)
    return C, C_H, T


def _build_phi_matrix(
    Xs_inter: List[np.ndarray],
    anchors_inter: List[np.ndarray],
    gammas: List[float],
) -> np.ndarray:
    if not Xs_inter or not anchors_inter:
        return np.zeros((0, 0))
    if len(Xs_inter) != len(anchors_inter):
        raise ValueError("Xs_inter and anchors_inter must have the same length.")
    blocks = []
    for X_inst, anchor_inst, gamma in zip(Xs_inter, anchors_inter, gammas):
        if X_inst.size == 0:
            blocks.append(np.zeros((0, anchor_inst.shape[0])))
            continue
        blocks.append(rbf_kernel(X_inst, anchor_inst, gamma=gamma))
    if not blocks:
        return np.zeros((0, 0))
    return block_diag(*blocks)


# --- Per-method integrator builders (return projector and the raw matrix when applicable) ---

def compute_linear_integrator_from_Z_anchor(
    Z_integ: np.ndarray,
    anchor_inter_k: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """TargetVec-style: build (right-mult) integrator from Z_integ and anchor_inter_k.
    Returns (projector, G_k).
    """
    G_k = pinv(anchor_inter_k) @ Z_integ
    return make_linear_integrator(G_k), G_k


# ================================
# Projector builders per method
# ================================

def build_imakura_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, float]:
    """
    SVD(Imakura) based projector builders.
    Returns (projs_per_institution, Z_integ (r�~m_inter), g_abs_sum).
    """
    centralized_anchor = np.hstack(anchors_inter)  # r �~ sum d_k
    U, _, _ = np.linalg.svd(centralized_anchor)
    U = U[:, :dim_integrate]
    Z_integ = U  # r �~ m_inter (retain for Z_integ)
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    g_abs_sum = 0.0
    for anchor_inter_k in anchors_inter:
        # Build per-institution projector from Z_integ and anchor_inter_k (right-mult form)
        proj, integrate_function = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        g_abs_sum += float(np.sum(np.abs(integrate_function)))
        projs.append(proj)
    return projs, Z_integ, g_abs_sum

def build_targetvec_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    zerosum: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray]:
    """
    TargetVec-based projector builders.
    Returns (projs_per_institution, Z_integ (r�~m_inter)).
    """
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)
    C_tildeS = c * I_r
    for anchor_inter_k in anchors_inter:
        C_tildeS -= anchor_inter_k @ pinv(anchor_inter_k)

    if zerosum:
        B = _zerosum_helmert_basis(r)
        M_tilde = (B.T @ C_tildeS @ B + (B.T @ C_tildeS @ B).T) * 0.5
        eigvals, eigvecs_sub = np.linalg.eigh(M_tilde)
        order = np.argsort(eigvals)
        take = min(dim_integrate, eigvecs_sub.shape[1])
        select = order[:take]
        eigvecs = eigvecs_sub[:, select]
        Z_integ = B @ eigvecs
    else:
        eigvals, eigvecs = np.linalg.eigh(C_tildeS)
        eigvals[eigvals < 0] = 0.0
        Z_integ = eigvecs[:, :dim_integrate]

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_inter_k in anchors_inter:
        proj, _Gk = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        projs.append(proj)
    return projs, Z_integ


def build_targetvec_singular_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    zerosum: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray]:
    """
    Rank-deficient tolerant TargetVec variant using QR+SVD (Theorem 3.15 style).

    Steps:
      - Thin QR: A_k = Q_k R_k with rank r_k (columns of Q_k truncated to r_k).
      - Stack W_Q = [Q_1 ... Q_c], take SVD W_Q = U Σ V^T.
      - Z = U(:, 1:t) with t = min(dim_integrate, rank(W_Q)).
      - G_k = A_k^T Z.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,))

    row_dim = anchors_inter[0].shape[0]
    Q_blocks: list[np.ndarray] = []
    ranks: list[int] = []
    for A in anchors_inter:
        if A.shape[0] != row_dim:
            raise ValueError("targetvec_singular requires all anchor matrices to share the same number of rows.")
        r_k = int(np.linalg.matrix_rank(A))
        if r_k <= 0:
            Q_blocks.append(np.zeros((row_dim, 0)))
            ranks.append(0)
            continue
        # Choose an orthonormal basis of Col(A): full rank -> QR, otherwise SVD-based.
        if r_k == min(A.shape):
            Q_k, _ = np.linalg.qr(A, mode="reduced")
            Q_blocks.append(Q_k[:, :r_k])
        else:
            U_k, _, _ = np.linalg.svd(A, full_matrices=False)
            Q_blocks.append(U_k[:, :r_k])
        ranks.append(r_k)

    W_Q = np.hstack(Q_blocks) if Q_blocks else np.zeros((row_dim, 0))
    if W_Q.size == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    if zerosum:
        B = _zerosum_helmert_basis(row_dim)
        W_use = B.T @ W_Q
    else:
        W_use = W_Q

    U, S, _ = np.linalg.svd(W_use, full_matrices=False)
    t = min(dim_integrate, U.shape[1])
    if t <= 0:
        return [], np.zeros((row_dim, 0)), S

    U_d = U[:, :t]
    if zerosum:
        Z_integ = B @ U_d
    else:
        Z_integ = U_d

    projs: list[Callable[[np.ndarray], np.ndarray]] = []
    for A_k in anchors_inter:
        G_k = pinv(A_k) @ Z_integ
        projs.append(make_linear_integrator(G_k))

    eigvals = S[:t]
    return projs, Z_integ, eigvals


def build_laplacian_targetvec_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    graph_mu_align: float = 0.0,
    laplacian_k: int = 10,
    zerosum: bool = False,
    regularization: str = "graph",
    constraint_matrix: Optional[np.ndarray] = None,
    constraint_eps: float = 1e-9,
    L_within: Optional[np.ndarray] = None,
    L_between: Optional[np.ndarray] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray]:
    """
    TargetVec eigenproblem regularization:
      - regularization="graph":      A = M + mu * L_plain
      - regularization="target-graph": A = M + mu * L_within, with optional mass matrix
    where M is the original TargetVec matrix.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,))

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    I_r = np.eye(r)
    M = c * I_r
    for anchor_inter_k in anchors_inter:
        M -= anchor_inter_k @ pinv(anchor_inter_k)
    M = (M + M.T) * 0.5

    reg_key = str(regularization or "graph").lower()
    eps = 1e-12
    A = M
    if float(graph_mu_align) != 0.0:
        tr_M = float(np.trace(M))
        if reg_key == "graph":
            L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=laplacian_k)
            if L_plain.shape == M.shape:
                L_plain = (L_plain + L_plain.T) * 0.5
                tr_L = float(np.trace(L_plain))
                scale_L = tr_M / max(tr_L, eps) if tr_L > 0 else 1.0
                A = M + float(graph_mu_align) * scale_L * L_plain
        elif reg_key in {"target-graph", "target_graph"}:
            if L_within is None:
                raise ValueError("target-graph regularization requires L_within.")
            Lw = np.asarray(L_within, dtype=float)
            if Lw.shape != M.shape:
                raise ValueError(f"target-graph requires L_within shape {M.shape} but got {Lw.shape}")
            Lw = (Lw + Lw.T) * 0.5
            tr_Lw = float(np.trace(Lw))
            if tr_M > eps:
                M_use = M / tr_M
            else:
                M_use = M
            if tr_Lw > eps:
                Lw_use = Lw / tr_Lw
            else:
                Lw_use = Lw
            A = M_use + float(graph_mu_align) * Lw_use
    A = (A + A.T) * 0.5

    mass = None
    if reg_key in {"target-graph", "target_graph"} and constraint_matrix is None and L_between is not None:
        Lb = np.asarray(L_between, dtype=float)
        if Lb.shape == (r, r):
            Lb = (Lb + Lb.T) * 0.5
            mass = Lb
    if constraint_matrix is not None:
        mass = np.asarray(constraint_matrix, dtype=float)
        if mass.shape != (r, r):
            raise ValueError(f"constraint_matrix must be shape {(r, r)} but got {mass.shape}")

    if zerosum:
        B_zero = _zerosum_helmert_basis(A.shape[0])
        A_tilde = (B_zero.T @ A @ B_zero + (B_zero.T @ A @ B_zero).T) * 0.5
        if mass is None:
            eigvals_raw, eigvecs_sub = np.linalg.eigh(A_tilde)
        else:
            B_tilde = (B_zero.T @ mass @ B_zero + (B_zero.T @ mass @ B_zero).T) * 0.5
            B_tilde = B_tilde + float(constraint_eps) * np.eye(B_tilde.shape[0])
            eigvals_raw, eigvecs_sub = eigh(A_tilde, B_tilde)
        eigvecs_full = B_zero @ eigvecs_sub
    else:
        if mass is None:
            eigvals_raw, eigvecs_full = np.linalg.eigh(A)
        else:
            B_mass = (mass + mass.T) * 0.5 + float(constraint_eps) * np.eye(mass.shape[0])
            eigvals_raw, eigvecs_full = eigh(A, B_mass)

    order = np.argsort(eigvals_raw)
    take = min(dim_integrate, eigvecs_full.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    Z_integ = eigvecs_full[:, select]

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_inter_k in anchors_inter:
        proj, _Gk = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        projs.append(proj)
    return projs, Z_integ, eigvals_selected


def build_linear_nonridge_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    nl_lambda: float = 1e-2,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray]:
    """
    Linear-kernel variant matching the provided formulation.
    Builds M_lambda = sum_k A_k (A_k^T A_k + λ I)^{-1} A_k^T (or sum_k A_k A_k^T if λ=∞),
    takes its leading eigenvectors for Z, then obtains G_k via least squares (no ridge).
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,))

    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    lam_raw = nl_lambda
    lam_is_infinite = False
    lam_value: Optional[float] = None
    if isinstance(lam_raw, str):
        lam_str = lam_raw.strip()
        if lam_str == "∞" or lam_str.lower() in {"inf", "infinity"}:
            lam_is_infinite = True
        else:
            lam_value = float(lam_str)
    else:
        lam_value = float(lam_raw)
        if math.isinf(lam_value):
            lam_is_infinite = True
            lam_value = None

    M_terms: List[np.ndarray] = []
    for A_k in anchors_inter:
        if lam_is_infinite:
            term = A_k @ A_k.T
        else:
            lam = float(lam_value if lam_value is not None else 0.0)
            I_d = np.eye(A_k.shape[1])
            if lam == 0:
                try:
                    inv_block = np.linalg.inv(A_k.T @ A_k)
                except np.linalg.LinAlgError:
                    inv_block = np.linalg.pinv(A_k.T @ A_k)
            else:
                try:
                    inv_block = np.linalg.inv(A_k.T @ A_k + lam * I_d)
                except np.linalg.LinAlgError:
                    inv_block = np.linalg.pinv(A_k.T @ A_k + lam * I_d)
            term = (1.0 + lam) * A_k @ (inv_block @ A_k.T)
        M_terms.append(term)

    M = sum(M_terms)
    M = (M + M.T) * 0.5

    eigvals_raw, eigvecs = eigh(M)
    order = np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    Z_integ = eigvecs[:, select]

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_inter_k in anchors_inter:
        proj, _Gk = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        projs.append(proj)

    return projs, Z_integ, eigvals_selected


def build_imakura_new_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    truncated: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, float]:
    """
    Imakura (shared subspace maximization) with optional truncated SVD.
    Uses SVD of W_A = [A_1, ..., A_c] to obtain Z* and then
    G_k* = A_k^† Z* for each institution k.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), 0.0
    centralized_anchor = np.hstack(anchors_inter)
    U, _, _ = _top_svd(centralized_anchor, dim_integrate, truncated=truncated)
    Z_integ = U
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    g_abs_sum = 0.0
    for anchor_inter_k in anchors_inter:
        proj, G_k = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        g_abs_sum += float(np.sum(np.abs(G_k)))
        projs.append(proj)
    return projs, Z_integ, g_abs_sum


def build_targetvec_new_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    truncated: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray]:
    """
    Target matrix optimization (TargetVec) based on QR+SVD formulation.

    Each A_k is decomposed as A_k = Q_k R_k (thin QR), W_Q = [Q_1, ..., Q_c]
    is formed, and Z* is given by the leading left singular vectors of W_Q.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0))

    Q_list: List[np.ndarray] = []
    for A in anchors_inter:
        Q_k, _ = np.linalg.qr(A, mode="reduced")
        Q_list.append(Q_k)

    W_Q = np.hstack(Q_list)
    U, _, _ = _top_svd(W_Q, dim_integrate, truncated=truncated)
    Z_integ = U

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_inter_k in anchors_inter:
        proj, _Gk = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        projs.append(proj)
    return projs, Z_integ


def build_multi_cca_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    stability_eps: float = 1e-6,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Multi-view CCA (SUMCOR) based projector builders.

    Given per-institution anchor representations S~^(k) (rows: anchor samples,
    columns: institution-specific features), this implementation:
      1. Centers each S~^(k) across rows to obtain S̄^(k).
      2. Builds within-view covariances Σ_kk and cross-covariances Σ_kℓ.
      3. Forms the block matrix M = D^{-1/2} R D^{-1/2} where D is blockdiag(Σ_kk)
         and R has off-diagonal blocks Σ_kℓ (diagonal blocks are zero).
      4. Solves the eigenproblem M u_j = λ_j u_j and takes the top
         `dim_integrate` eigenvectors.
      5. Computes per-view projection matrices W^(k) = Σ_kk^{-1/2} U^(k),
         where U^(k) is the block of U corresponding to view k.

    The returned projectors apply X -> X @ W^(k) for each institution k.
    """
    if not anchors_inter:
        return [], {"eigvals": np.array([]), "W_list": [], "Z_integ": None}

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    for anchor in anchors_inter:
        if anchor.shape[0] != r:
            raise ValueError("All anchor matrices must share the same number of rows for multi-view CCA.")

    # Step 1: center each anchor representation
    centered: List[np.ndarray] = []
    means: List[np.ndarray] = []
    for S in anchors_inter:
        mu = np.mean(S, axis=0, keepdims=True)
        means.append(mu)
        centered.append(S - mu)

    # Step 2: build within- and cross-view covariance blocks
    factor = 1.0 / max(r - 1, 1)
    dims = [S.shape[1] for S in centered]
    p_total = int(sum(dims))

    Sigma_kk: List[np.ndarray] = []
    for S in centered:
        cov = factor * (S.T @ S)
        cov = _nearest_spd(cov, min_eig=stability_eps)
        Sigma_kk.append(cov)

    Sigma_kl: List[List[Optional[np.ndarray]]] = [[None for _ in range(c)] for _ in range(c)]
    for k in range(c):
        for ell in range(k + 1, c):
            cov_kell = factor * (centered[k].T @ centered[ell])
            Sigma_kl[k][ell] = cov_kell
            Sigma_kl[ell][k] = cov_kell.T

    # Precompute Σ_kk^{-1/2} via eigen-decomposition
    Sigma_kk_inv_sqrt: List[np.ndarray] = []
    for cov in Sigma_kk:
        eigvals_cov, eigvecs_cov = np.linalg.eigh(cov)
        eigvals_cov = np.maximum(eigvals_cov, stability_eps)
        inv_sqrt_vals = 1.0 / np.sqrt(eigvals_cov)
        cov_inv_sqrt = eigvecs_cov @ np.diag(inv_sqrt_vals) @ eigvecs_cov.T
        Sigma_kk_inv_sqrt.append(cov_inv_sqrt)

    # Step 3: build M = D^{-1/2} R D^{-1/2} in block form
    M = np.zeros((p_total, p_total), dtype=float)
    cum_dims = np.cumsum([0] + dims)
    for k in range(c):
        i0, i1 = cum_dims[k], cum_dims[k + 1]
        for ell in range(c):
            if k == ell:
                continue
            j0, j1 = cum_dims[ell], cum_dims[ell + 1]
            cov_kell = Sigma_kl[k][ell]
            if cov_kell is None:
                continue
            block = Sigma_kk_inv_sqrt[k] @ cov_kell @ Sigma_kk_inv_sqrt[ell]
            M[i0:i1, j0:j1] = block

    # Ensure symmetry
    M = (M + M.T) * 0.5

    # Step 4: eigen-decomposition of M (SUMCOR -> use largest eigenvalues)
    eigvals_raw, eigvecs = np.linalg.eigh(M)
    if eigvals_raw.ndim == 0:
        eigvals_raw = np.asarray([eigvals_raw])
        eigvecs = eigvecs.reshape(-1, 1)
    order = np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    U = eigvecs[:, select]

    # Step 5: per-view projection matrices W^(k) = Σ_kk^{-1/2} U^(k)
    W_list: List[np.ndarray] = []
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for k in range(c):
        i0, i1 = cum_dims[k], cum_dims[k + 1]
        U_k = U[i0:i1, :]
        W_k = Sigma_kk_inv_sqrt[k] @ U_k
        W_list.append(W_k)
        projs.append(make_centered_linear_integrator(W_k, means[k]))

    metrics: Dict[str, Any] = {
        "eigvals": eigvals_selected,
        "W_list": W_list,
        "means": means,
        # There is no single canonical Z_integ; projected anchors per view are:
        # [centered[k] @ W_list[k] for k in range(c)]
        "Z_integ": None,
    }
    return projs, metrics


def _solve_gep_standard(A: np.ndarray, B: np.ndarray, orth_ver: bool) -> Tuple[np.ndarray, np.ndarray]:
    if orth_ver:
        return eigh(A)
    return eigh(A, B)


def _solve_gep_regularized(
    A: np.ndarray,
    B: np.ndarray,
    orth_ver: bool,
    *,
    base_eps: float = 1e-6,
    attempts: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = base_eps
    last_error: Exception | None = None
    for _ in range(max(attempts, 1)):
        try:
            if orth_ver:
                return eigh(A + eps * np.eye(A.shape[0]))
            return eigh(A, B + eps * np.eye(B.shape[0]))
        except np.linalg.LinAlgError as exc:
            last_error = exc
            eps *= 10.0
    if last_error is not None:
        raise last_error
    raise np.linalg.LinAlgError("Failed to solve generalized eigen problem.")


def _nearest_spd(matrix: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _compute_gep_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    lambda_gen: float,
    orth_ver: bool,
    solver: Callable[[np.ndarray, np.ndarray, bool], Tuple[np.ndarray, np.ndarray]],
    regularize_B: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    # Build W and B
    W_s_tilde = np.hstack(anchors_inter)
    blocks = [anchor_inter_k.T @ anchor_inter_k for anchor_inter_k in anchors_inter]
    epsilon = 0 #1e-6
    B_s_tilde = blocks[0]
    for b in blocks[1:]:
        B_s_tilde = block_diag(B_s_tilde, b)
    B_s_tilde = B_s_tilde + epsilon * np.eye(B_s_tilde.shape[0])

    A_s_tilde = 2 * c * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde) + lambda_gen * np.eye(W_s_tilde.shape[1])

    B_for_solver = _nearest_spd(B_s_tilde) if regularize_B else B_s_tilde
    eigvals, eigvecs = solver(A_s_tilde, B_for_solver, orth_ver)
    order = np.argsort(eigvals)
    lambdas = eigvals[order][:dim_integrate]
    V_sel = eigvecs[:, order[:dim_integrate]]

    cum_dims = np.cumsum([0] + [anchor_inter_k.shape[1] for anchor_inter_k in anchors_inter])

    # Compute diagnostics analogous to original implementation
    jreg_val = 0.0
    for j in range(dim_integrate):
        gj = V_sel[:, j]
        term1 = 0.0
        sum_Sgj = np.zeros(r)
        for k in range(c):
            gjk = gj[cum_dims[k]:cum_dims[k+1]]
            anchor_inter_k = anchors_inter[k]
            term1 += gjk.T @ (anchor_inter_k.T @ anchor_inter_k) @ gjk
            sum_Sgj += anchor_inter_k @ gjk
        jreg_val += (2.0 * c * term1 - 2.0 * (sum_Sgj @ sum_Sgj))

    norm_val_sum = 0.0
    for j in range(dim_integrate):
        gj = V_sel[:, j]
        for k in range(c):
            gjk = gj[cum_dims[k]:cum_dims[k+1]]
            anchor_inter_k = anchors_inter[k]
            norm_vec = anchor_inter_k @ gjk
            norm_val_sum += norm_vec @ norm_vec
    avg_norm_val = norm_val_sum / dim_integrate if dim_integrate > 0 else 0.0

    g_abs_sum = float(np.sum(np.abs(V_sel)))
    mean_vars = []
    for k in range(len(anchors_inter)):
        V_k = V_sel[cum_dims[k]:cum_dims[k + 1], :]
        var_k = np.var(V_k, axis=0)
        mean_vars.append(np.mean(var_k))
    g_mean_var = float(np.mean(mean_vars)) if mean_vars else 0.0
    lambda_min, lambda_max = lambdas[0], lambdas[-1]
    cond_number = float(lambda_max / lambda_min) if lambda_min > 0 else float("inf")

    # Build projectors per institution
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for k in range(len(anchors_inter)):
        G_k = V_sel[cum_dims[k]:cum_dims[k + 1], :]
        proj = make_linear_integrator(G_k)
        projs.append(proj)

    metrics: Dict[str, Any] = {
        "V_sel": V_sel,
        "lambdas": lambdas,
        "jreg_gep": jreg_val,
        "g_norm_val_gep": avg_norm_val,
        "sum_objective_function": float(np.sum(lambdas)),
        "g_abs_sum": g_abs_sum,
        "g_mean_var": g_mean_var,
        "g_condition_number": cond_number,
    }
    return projs, metrics


def build_gep_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    lambda_gen: float = 0.0,
    orth_ver: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Legacy (test-aligned) GEP projector builder.
    """
    return _compute_gep_projectors(
        anchors_inter,
        dim_integrate,
        lambda_gen=lambda_gen,
        orth_ver=orth_ver,
        solver=_solve_gep_standard,
        regularize_B=False,
    )


def build_gep2_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    lambda_gen: float = 0.0,
    orth_ver: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Enhanced (regularized) GEP projector builder with fallback eigen solver.
    """
    return _compute_gep_projectors(
        anchors_inter,
        dim_integrate,
        lambda_gen=lambda_gen,
        orth_ver=orth_ver,
        solver=_solve_gep_regularized,
        regularize_B=True,
    )


def build_faster_gep_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    QR+SVD based fast GEP projector builder (Kawakami-DC style).

    For each institution i, let A_i be the anchor intermediate representation
    (rows: anchor samples, columns: local features). This routine:
      1. Computes thin QR factorizations A_i = Q_i R_i.
      2. Forms W_Q = [Q_1 ... Q_c] and computes its SVD W_Q = U Σ V^T.
      3. Takes the leading `t = min(dim_integrate, d)` right singular vectors,
         partitions them into c blocks, and obtains \hat g_{i,k}.
      4. Sets G_i = R_i^{-1} [\hat g_{i,1}, ..., \hat g_{i,t}] and additionally
         rescales G_i <- G_i / sqrt(c) for numerical stability (c = #institutions).

    Returns (projectors_per_institution, metrics), where each projector applies
    X -> X @ G_i.
    """
    if not anchors_inter:
        return [], {"G_list": [], "R_list": [], "singular_values": np.array([])}

    c = len(anchors_inter)
    # All anchor matrices must share the same column dimension for block partitioning.
    d = anchors_inter[0].shape[1]
    for A in anchors_inter:
        if A.shape[1] != d:
            raise ValueError("faster_gep requires all anchor matrices to share the same column dimension.")
        if A.shape[0] < d:
            raise ValueError("faster_gep requires each anchor matrix to have rows >= columns for thin QR.")

    # Step 1: thin QR factorizations A_i = Q_i R_i
    Q_list: List[np.ndarray] = []
    R_list: List[np.ndarray] = []
    for A in anchors_inter:
        Q_i, R_i = np.linalg.qr(A, mode="reduced")
        Q_list.append(Q_i)
        R_list.append(R_i)

    # Step 2: SVD of concatenated Q blocks
    W_Q = np.hstack(Q_list)  # shape: (r, c * d)
    if W_Q.size == 0:
        return [], {"G_list": [], "R_list": R_list, "singular_values": np.array([])}

    U, S, Vt = np.linalg.svd(W_Q, full_matrices=False)
    V = Vt.T  # shape: (c * d, rank)

    # Number of integrated dimensions (t in the paper, capped by d)
    t = min(dim_integrate, d, V.shape[1])
    if t <= 0:
        return [], {"G_list": [], "R_list": R_list, "singular_values": S}

    V_t = V[:, :t]  # (c * d, t)

    # Step 3: partition right singular vectors into c blocks of length d
    try:
        V_blocks = V_t.reshape(c, d, t)  # V_blocks[i] = \hat G_i (d x t)
    except ValueError as exc:
        raise ValueError(
            f"faster_gep reshape failed; expected total columns c*d = {c*d}, "
            f"got {V_t.shape[0]}"
        ) from exc

    # Step 4: G_i = R_i^{-1} \hat G_i with additional 1/sqrt(c) scaling
    scale = 1.0 / np.sqrt(float(c)) if c > 0 else 1.0
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    G_list: List[np.ndarray] = []

    for i in range(c):
        R_i = R_list[i]  # (d, d), upper-triangular and invertible under assumptions
        hat_G_i = V_blocks[i]  # (d, t)
        # Solve R_i G_i = hat_G_i for G_i
        G_i = np.linalg.solve(R_i, hat_G_i)
        G_i = scale * G_i
        G_list.append(G_i)
        projs.append(make_linear_integrator(G_i))

    metrics: Dict[str, Any] = {
        "G_list": G_list,
        "R_list": R_list,
        "singular_values": S,
        "output_dim": t,
    }
    return projs, metrics


def build_gep_new_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    truncated: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray]:
    """
    Pairwise (GEP-like) method based on the QR+SVD formulation.

    With anchor matrices A_k ∈ R^{a×p̃} (rows = anchor samples, columns =
    intermediate features) we compute thin QR factorizations A_k = Q_k R_k,
    stack W_Q = [Q_1, …, Q_c], and obtain the leading p̂ singular triplets
    W_Q = U Σ V^T. The optimal values from Theorem 3.12 are
        Z* = (1/√c) U(:,1:p̂) Σ(1:p̂,1:p̂),
        G*_k = R_k^{-1} V_k,
    where V_k is the block of V(:,1:p̂) corresponding to institution k.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0))

    c = len(anchors_inter)
    row_dim = anchors_inter[0].shape[0]
    col_dim = anchors_inter[0].shape[1]

    Q_list: List[np.ndarray] = []
    R_list: List[np.ndarray] = []
    for idx, A in enumerate(anchors_inter):
        if A.shape[0] != row_dim:
            raise ValueError("gep_new requires all anchor matrices to share the same number of rows (anchor samples).")
        if A.shape[1] != col_dim:
            raise ValueError("gep_new requires all anchor matrices to share the same number of columns (intermediate features).")
        if A.shape[0] < A.shape[1]:
            raise ValueError("gep_new expects each anchor matrix to satisfy (#rows >= #cols) for thin QR.")
        Q_k, R_k = np.linalg.qr(A, mode="reduced")
        Q_list.append(Q_k)
        R_list.append(R_k)

    W_Q = np.hstack(Q_list)
    U, S, Vt = _top_svd(W_Q, dim_integrate, truncated=truncated)
    output_dim = U.shape[1]
    scale = 1.0 / math.sqrt(float(c)) if c > 0 else 1.0
    if output_dim > 0:
        Z_integ = scale * (U @ np.diag(S[:output_dim]))
    else:
        Z_integ = np.zeros((row_dim, 0))

    if output_dim == 0:
        zero_proj = make_linear_integrator(np.zeros((col_dim, 0)))
        return [zero_proj for _ in anchors_inter], Z_integ

    V = Vt.T  # shape: (c * col_dim, output_dim)
    try:
        V_blocks = V.reshape(c, col_dim, output_dim)
    except ValueError as exc:
        expected = c * col_dim
        raise ValueError(
            f"gep_new expected right singular vectors of length {expected}, got {V.shape[0]}."
        ) from exc

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for idx in range(c):
        R_k = R_list[idx]
        hat_G_k = V_blocks[idx]
        try:
            G_k = solve_triangular(R_k, hat_G_k, lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            G_k = np.linalg.pinv(R_k) @ hat_G_k
        projs.append(make_linear_integrator(G_k))

    return projs, Z_integ


def build_gep_singular_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray]:
    """
    QR+SVD based solution tolerant to rank-deficient anchors.

    For each A_k (r x d_k), take thin QR: A_k = Q_k R_k (r_k = rank(A_k)).
    Stack W_Q = [Q_1 ... Q_c], compute SVD W_Q = U Σ V^T, take top t components:
        Z = (1/√c) U(:,1:t) Σ(1:t),
        G_k = √c * pinv(R_k) @ V_k
    where V_k is the block of V(:,1:t) corresponding to institution k.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,))

    c = len(anchors_inter)
    row_dim = anchors_inter[0].shape[0]
    Q_blocks: List[np.ndarray] = []
    R_blocks: List[np.ndarray] = []
    ranks: List[int] = []

    for A in anchors_inter:
        if A.shape[0] != row_dim:
            raise ValueError("gep_singular requires all anchor matrices to share the same number of rows.")
        r_k = int(np.linalg.matrix_rank(A))
        if r_k <= 0:
            Q_blocks.append(np.zeros((row_dim, 0)))
            R_blocks.append(np.zeros((0, A.shape[1])))
            ranks.append(0)
            continue
        # Orthonormal basis of Col(A); if rank deficient, use SVD-based basis.
        if r_k == min(A.shape):
            Q_k, _ = np.linalg.qr(A, mode="reduced")
            Q_k = Q_k[:, :r_k]
        else:
            U_k, _, _ = np.linalg.svd(A, full_matrices=False)
            Q_k = U_k[:, :r_k]
        R_k = Q_k.T @ A  # rank-revealing upper block
        Q_blocks.append(Q_k)
        R_blocks.append(R_k)
        ranks.append(r_k)

    W_Q = np.hstack(Q_blocks) if Q_blocks else np.zeros((row_dim, 0))
    if W_Q.size == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    U, S, Vt = np.linalg.svd(W_Q, full_matrices=False)
    t = min(dim_integrate, U.shape[1])
    if t <= 0:
        return [], np.zeros((0, 0)), S

    U_d = U[:, :t]
    S_d = S[:t]
    V_d = Vt.T[:, :t]  # shape (sum r_k, t)

    Z_integ = (U_d * S_d) / np.sqrt(float(c))

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    offset = 0
    for R_k, r_k, A_k in zip(R_blocks, ranks, anchors_inter):
        V_k = V_d[offset : offset + r_k, :]
        offset += r_k
        if V_k.size == 0:
            G_k = np.zeros((A_k.shape[1], t))
        else:
            G_k = np.sqrt(float(c)) * pinv(R_k) @ V_k
        projs.append(make_linear_integrator(G_k))

    eigvals = S_d
    return projs, Z_integ, eigvals


def build_gep_singular_2_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray]:
    """
    QR+SVD based solution tolerant to rank-deficient anchors (alternative closed-form).

    This implements the closed-form given by (with Q = I):
        G_k^* = sqrt(c) * A_k^† U_{Qd} Σ_{Qd}^†
    where W_Q = [Q_1 ... Q_c] and W_Q = U Σ V^T is the SVD (truncated to d=t).

    Notes:
    - Q_k is an orthonormal basis of Col(A_k) with r_k = rank(A_k).
    - Z_integ is returned as (1/sqrt(c)) U_{Qd} Σ_{Qd}.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,))

    c = len(anchors_inter)
    row_dim = anchors_inter[0].shape[0]
    Q_blocks: List[np.ndarray] = []

    for A in anchors_inter:
        if A.shape[0] != row_dim:
            raise ValueError("gep_singular_2 requires all anchor matrices to share the same number of rows.")
        r_k = int(np.linalg.matrix_rank(A))
        if r_k <= 0:
            Q_blocks.append(np.zeros((row_dim, 0)))
            continue
        if r_k == min(A.shape):
            Q_k, _ = np.linalg.qr(A, mode="reduced")
            Q_k = Q_k[:, :r_k]
        else:
            U_k, _, _ = np.linalg.svd(A, full_matrices=False)
            Q_k = U_k[:, :r_k]
        Q_blocks.append(Q_k)

    W_Q = np.hstack(Q_blocks) if Q_blocks else np.zeros((row_dim, 0))
    if W_Q.size == 0:
        return [], np.zeros((0, 0)), np.zeros((0,))

    U, S, _ = np.linalg.svd(W_Q, full_matrices=False)
    t = min(dim_integrate, U.shape[1])
    if t <= 0:
        return [], np.zeros((0, 0)), S

    U_d = U[:, :t]
    S_d = S[:t]

    Z_integ = (U_d * S_d) / np.sqrt(float(c))

    # Build U Σ^† robustly (diagonal pseudoinverse).
    eps = np.finfo(float).eps
    tol = float(max(W_Q.shape)) * eps * (float(S_d[0]) if S_d.size else 0.0)
    inv_S = np.zeros_like(S_d)
    mask = S_d > tol
    inv_S[mask] = 1.0 / S_d[mask]
    U_S_dagger = U_d * inv_S  # equals U_d @ diag(inv_S)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    scale = np.sqrt(float(c))
    for A_k in anchors_inter:
        G_k = scale * (np.linalg.pinv(A_k) @ U_S_dagger)
        projs.append(make_linear_integrator(G_k))

    eigvals = S_d
    return projs, Z_integ, eigvals


def build_kernel_gep_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Kernelized GEP projector builders based on the RKHS formulation.
    """
    if not anchors_inter:
        return [], {"gammas": [], "eigvals": np.array([]), "alphas": np.array([])}
    r = _validate_anchor_rows(anchors_inter)
    c = len(anchors_inter)
    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    Ks = [rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[idx]) for idx, anchor_inter_k in enumerate(anchors_inter)]
    C, C_H, T = _build_anchor_alignment_terms(Ks)

    A = 2 * c * C - 2 * T + nl_lambda * C_H
    ridge = 1e-6
    B = C + ridge * np.eye(C.shape[0])
    eigvals, eigvecs = _solve_gep_regularized(A, B, orth_ver=False)
    take = min(dim_integrate, eigvecs.shape[1])
    order = np.argsort(eigvals)
    select = order[:take]
    eigvals_selected = eigvals[select]
    Alpha_stack = eigvecs[:, select]

    for j in range(Alpha_stack.shape[1]):
        vec = Alpha_stack[:, j]
        denom = float(vec.T @ (C @ vec))
        if denom > 0:
            Alpha_stack[:, j] = vec / np.sqrt(denom)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for k in range(c):
        start = k * r
        end = start + r
        Alpha_k = Alpha_stack[start:end, :]
        proj = make_kernel_integrator(anchors_inter[k], Alpha_k, gamma=gammas[k])
        projs.append(proj)

    metrics: Dict[str, Any] = {
        "alphas": Alpha_stack,
        "eigvals": eigvals_selected,
        "gammas": gammas,
    }
    return projs, metrics


def build_kernel_graph_gep_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    L_within_data: Optional[np.ndarray],
    L_between_data: Optional[np.ndarray],
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    mu_align: float = 1.0,
    lambda_rkhs: float = 1e-2,
    stability_eps: float = 1e-6,
    g_type: Optional[str] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Kernel GEP with graph Laplacian terms derived from intermediate representations.
    """
    if not anchors_inter:
        return [], {"gammas": [], "eigvals": np.array([]), "alphas": np.array([])}
    if L_within_data is None or L_between_data is None:
        raise ValueError("kernel_graph_gep requires both L_within_data and L_between_data.")

    r = _validate_anchor_rows(anchors_inter)
    c = len(anchors_inter)
    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    Ks = [rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[idx]) for idx, anchor_inter_k in enumerate(anchors_inter)]
    C, C_H, T = _build_anchor_alignment_terms(Ks)
    Phi = _build_phi_matrix(Xs_train_inter, anchors_inter, gammas)
    if Phi.size == 0:
        raise ValueError("kernel_graph_gep requires non-empty intermediate representations.")
    n_samples = Phi.shape[0]
    if L_within_data.shape != (n_samples, n_samples) or L_between_data.shape != (n_samples, n_samples):
        raise ValueError("Graph Laplacian shapes must match total sample size in Xs_train_inter.")

    Lw = np.asarray(L_within_data)
    Lb = np.asarray(L_between_data)
    A_b = Phi.T @ Lb @ Phi
    A_w = Phi.T @ Lw @ Phi
    A_b = (A_b + A_b.T) * 0.5
    A_w = (A_w + A_w.T) * 0.5

    A_align = 2 * c * C - 2 * T
    # Prepare both formulations
    # minimize: (A_w + μ A_align + λ C_H) a = γ (A_b + ε I) a
    A_min = (A_w + mu_align * A_align + lambda_rkhs * C_H)
    A_min = (A_min + A_min.T) * 0.5 + (mu_align * 1e-9) * np.eye(A_min.shape[0])
    B_min = (A_b + stability_eps * np.eye(A_b.shape[0]))
    B_min = (B_min + B_min.T) * 0.5

    # maximize: A_b a = γ (A_w + μ A_align + λ C_H + ε I) a
    A_max = A_b
    A_max = (A_max + A_max.T) * 0.5
    B_max = (A_w + mu_align * A_align + lambda_rkhs * C_H + stability_eps * np.eye(A_b.shape[0]))
    B_max = (B_max + B_max.T) * 0.5

    mode = (g_type or "").lower()
    use_max = ("maximize" in mode)

    A_use, B_use = (A_max, B_max) if use_max else (A_min, B_min)
    # Ensure B is SPD to avoid eigh failures on some datasets
    B_use = _nearest_spd(B_use, min_eig=max(stability_eps, 1e-9))
    eigvals, eigvecs = _solve_gep_regularized(A_use, B_use, orth_ver=False)
    take = min(dim_integrate, eigvecs.shape[1])
    # Select eigenvalues: minimize -> ascending, maximize -> descending
    order = np.argsort(eigvals) if not use_max else np.argsort(eigvals)[::-1]
    select = order[:take]
    eigvals_selected = eigvals[select]
    Alpha_stack = eigvecs[:, select]

    for j in range(Alpha_stack.shape[1]):
        vec = Alpha_stack[:, j]
        denom = float(vec.T @ (B_use @ vec))
        if denom > 0:
            Alpha_stack[:, j] = vec / np.sqrt(denom)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for k in range(c):
        start = k * r
        end = start + r
        Alpha_k = Alpha_stack[start:end, :]
        proj = make_kernel_integrator(anchors_inter[k], Alpha_k, gamma=gammas[k])
        projs.append(proj)

    metrics: Dict[str, Any] = {
        "alphas": Alpha_stack,
        "eigvals": eigvals_selected,
        "gammas": gammas,
        "mode": "maximize" if use_max else "minimize",
    }
    return projs, metrics


def build_kernel_graph_gep_projectors_maximize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    L_within_data: Optional[np.ndarray],
    L_between_data: Optional[np.ndarray],
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    mu_align: float = 1.0,
    lambda_rkhs: float = 1e-2,
    stability_eps: float = 1e-6,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    return build_kernel_graph_gep_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        L_within_data=L_within_data,
        L_between_data=L_between_data,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        mu_align=mu_align,
        lambda_rkhs=lambda_rkhs,
        stability_eps=stability_eps,
        g_type="kernel_graph_gep_maximize",
    )


def build_kernel_graph_gep_projectors_minimize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    L_within_data: Optional[np.ndarray],
    L_between_data: Optional[np.ndarray],
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    mu_align: float = 1.0,
    lambda_rkhs: float = 1e-2,
    stability_eps: float = 1e-6,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    return build_kernel_graph_gep_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        L_within_data=L_within_data,
        L_between_data=L_between_data,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        mu_align=mu_align,
        lambda_rkhs=lambda_rkhs,
        stability_eps=stability_eps,
        g_type="kernel_graph_gep_minimize",
    )


def build_odc_projectors(
    anchors_inter: List[np.ndarray],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray]:
    """
    Orthogonal Procrustes based projectors. Returns (projs, anchor_1-as-Z_integ)
    """
    if not anchors_inter:
        return [], np.array([])
    anchor_1 = anchors_inter[0]
    Z_integ = anchor_1
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_k in anchors_inter:
        M_k = anchor_k.T @ Z_integ
        U_k, _, Vh_k = np.linalg.svd(M_k, full_matrices=False)
        G_k = U_k @ Vh_k
        projs.append(make_linear_integrator(G_k))
    return projs, Z_integ


def _zerosum_helmert_basis(n: int) -> np.ndarray:
    """
    Construct an orthonormal basis of the zero-sum subspace in R^n (Helmert type).

    Columns of the returned matrix B (shape: n x (n-1)) satisfy:
        1^T b_k = 0,  ||b_k|| = 1,  and they are mutually orthogonal.
    """
    if n <= 1:
        raise ValueError("zerosum basis requires n >= 2.")
    B = np.zeros((n, n - 1), dtype=float)
    for k in range(1, n):
        denom = np.sqrt(k * (k + 1))
        B[:k, k - 1] = 1.0 / denom
        B[k, k - 1] = -k / denom
    return B


def build_nonlinear_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    kernel_type: str = "rbf",
    K_normalization: bool = False,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 0.0,
    L_within: Optional[np.ndarray] = None,
    L_between: Optional[np.ndarray] = None,
    zerosum: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Kernel (nonlinear) based projector builders.
    Returns (projs_per_institution, Z_integ (r�~m_inter), eigvals (ascending)).
    """
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    Ks, Ps, mu_max_list = [], [], []
    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        if K_normalization:
            mu_max = max(np.linalg.eigvalsh(K).max(), 1e-12)
            mu_max_list.append(mu_max)
            K = K / mu_max
        else:
            mu_max_list.append(None)
        Ks.append(K)
        Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))

    M = sum((P - I_r).T @ (P - I_r) for P in Ps)
    Msym = (M + M.T) * 0.5
    
    print("float(graph_mu_align)", float(graph_mu_align))
    if graph_mu_align == 0.0 or L_within is None or L_between is None:
        Q = Msym
    else:
        eps = 1e-12
        tr_M = float(np.trace(Msym))
        tr_Lw = float(np.trace(L_within))
        tr_Lb = float(np.trace(L_between))
        scale_Lw = tr_M / max(tr_Lw, eps) if tr_Lw > 0 else 1.0
        n = L_between.shape[0]
        scale_Lb = n / max(tr_Lb, eps) if tr_Lb > 0 else 1.0
        Q = Msym + float(graph_mu_align) * (scale_Lw * L_within - scale_Lb * L_between)
        
        print("float(graph_mu_align)", float(graph_mu_align))
        print("scale_Lw", scale_Lw)

    if zerosum:
        B = _zerosum_helmert_basis(Q.shape[0])
        Q_tilde = B.T @ Q @ B
        I_sub = np.eye(Q_tilde.shape[0])
        eigvals_raw, eigvecs_sub = eigh(Q_tilde, I_sub)
    else:
        I_full = np.eye(Q.shape[0])
        eigvals_raw, eigvecs_sub = eigh(Q, I_full)
    order = np.argsort(eigvals_raw)
    eigvals_selected = eigvals_raw[order[:dim_integrate]]
    eigvecs = eigvecs_sub[:, order[:dim_integrate]]
    if zerosum:
        Z_integ = B @ eigvecs
    else:
        Z_integ = eigvecs
    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(
            anchors_inter[i],
            B_k,
            gamma=gammas[i],
            normalize=K_normalization,
            mu_max=mu_max_list[i],
            kernel_type=kernel_type_key,
        )
        projs.append(proj)
            
    # Z_integ の各列の総和を計算
    col_sums = np.sum(Z_integ, axis=0)
    # print で表示
    #print("Column sums:", col_sums)

    return projs, Z_integ, eigvals_selected, gammas


def build_nonlinear_new_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    kernel_type: str = "rbf",
    K_normalization: bool = False,
    nl_lambda: float = 1e-2,
    zerosum: bool = False,
    constraint_matrix: Optional[np.ndarray] = None,
    constraint_eps: float = 1e-9,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Paper-matching variant of nonlinear integration.

    Uses:
      S^(k) = (K^(k) + λ I)^(-1)
      M_λ = λ * sum_k S^(k)
      Z = argmin_{Z^T Z = I} tr(Z^T M_λ Z)  => smallest eigenvectors of M_λ
      C^(k)* = S^(k) Z
      g^(k)(x) = κ_k(x) C^(k)  (kernel expansion over anchors)
    """
    def _sym(A: np.ndarray) -> np.ndarray:
        return (A + A.T) * 0.5

    def _solve_eig(
        A: np.ndarray,
        dim: int,
        *,
        zerosum_enabled: bool,
        B_mass: Optional[np.ndarray],
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        A = _sym(A)
        take = min(int(dim), A.shape[0])
        if take <= 0:
            return np.zeros((A.shape[0], 0)), np.zeros((0,))

        if zerosum_enabled and A.shape[0] >= 2:
            B0 = _zerosum_helmert_basis(A.shape[0])
            A_tilde = _sym(B0.T @ A @ B0)
            take_tilde = min(take, A_tilde.shape[0])
            if B_mass is None:
                eigvals, eigvecs = _smallest_eigh(A_tilde, take_tilde)
            else:
                B_mass = _sym(B_mass)
                B_tilde = _sym(B0.T @ B_mass @ B0)
                # Add stability term once in the reduced (zero-sum) subspace.
                B_tilde = B_tilde + float(eps) * np.eye(B_tilde.shape[0])
                eigvals, eigvecs = _smallest_eigh(A_tilde, take_tilde, B=B_tilde)
            Z = B0 @ eigvecs
            return Z, eigvals

        if B_mass is None:
            eigvals, eigvecs = _smallest_eigh(A, take)
        else:
            B_mass = _sym(B_mass)
            # Add stability term once in the original space.
            B_mass = B_mass + float(eps) * np.eye(B_mass.shape[0])
            eigvals, eigvecs = _smallest_eigh(A, take, B=B_mass)
        Z = eigvecs
        return Z, eigvals

    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []
    I_r = np.eye(r)
    eps = float(constraint_eps) if constraint_eps is not None else 1e-9
    eps = max(eps, 1e-12)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    Ks: List[np.ndarray] = []
    Ss: List[np.ndarray] = []
    mu_max_list: List[Optional[float]] = []
    lam = float(nl_lambda)

    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])

        if K_normalization:
            mu_max = max(float(np.linalg.eigvalsh(K).max()), 1e-12)
            mu_max_list.append(mu_max)
            K = K / mu_max
        else:
            mu_max_list.append(None)

        Ks.append(K)
        try:
            S = np.linalg.inv(K + lam * I_r)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(K + lam * I_r)
        Ss.append(S)

    M_lambda = _sym(lam * sum(Ss))

    mass = None
    if constraint_matrix is not None:
        mass = np.asarray(constraint_matrix, dtype=float)
        if mass.shape != (r, r):
            raise ValueError(f"constraint_matrix must be shape {(r, r)} but got {mass.shape}")

    Z_integ, eigvals_selected = _solve_eig(
        M_lambda,
        dim_integrate,
        zerosum_enabled=bool(zerosum),
        B_mass=mass,
        eps=eps,
    )

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, (S, anchor_inter_k) in enumerate(zip(Ss, anchors_inter)):
        C_k = S @ Z_integ
        proj = make_kernel_integrator(
            anchor_inter_k,
            C_k,
            gamma=gammas[i],
            normalize=K_normalization,
            mu_max=mu_max_list[i],
            kernel_type=kernel_type_key,
        )
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_nonlinear_imakura_Z_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    kernel_type: str = "rbf",
    K_normalization: bool = False,
    nl_lambda: float = 1e-2,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Nonlinear integration with Imakura-fixed target representation Z.

    Z is fixed to the leading left singular vectors of concatenated anchors:
        W = [A_1, A_2, ..., A_c],  Z = U[:, :dim_integrate]  where W = U Σ V^T.

    Then each institution-specific nonlinear projector is obtained by
        C_k = (K_k + λI)^(-1) Z,
        g_k(x) = k(x, A_k) C_k.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    for anchor_inter_k in anchors_inter:
        if anchor_inter_k.shape[0] != r:
            raise ValueError("nonlinear_imakura_Z requires all anchor matrices to share the same number of rows.")

    # Fix Z to Imakura-style target representation.
    W_concat = np.hstack(anchors_inter)
    U, S, _ = np.linalg.svd(W_concat, full_matrices=False)
    take = min(int(dim_integrate), U.shape[1])
    if take <= 0:
        return [], np.zeros((r, 0)), np.zeros((0,)), []
    Z_integ = U[:, :take]
    eigvals_selected = S[:take]

    I_r = np.eye(r)
    lam = float(nl_lambda)
    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    Ss: List[np.ndarray] = []
    mu_max_list: List[Optional[float]] = []
    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])

        if K_normalization:
            mu_max = max(float(np.linalg.eigvalsh(K).max()), 1e-12)
            mu_max_list.append(mu_max)
            K = K / mu_max
        else:
            mu_max_list.append(None)

        try:
            S_k = np.linalg.inv(K + lam * I_r)
        except np.linalg.LinAlgError:
            S_k = np.linalg.pinv(K + lam * I_r)
        Ss.append(S_k)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, (S_k, anchor_inter_k) in enumerate(zip(Ss, anchors_inter)):
        C_k = S_k @ Z_integ
        proj = make_kernel_integrator(
            anchor_inter_k,
            C_k,
            gamma=gammas[i],
            normalize=K_normalization,
            mu_max=mu_max_list[i],
            kernel_type=kernel_type_key,
        )
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_nonlinear_mlp_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    hidden_dims: Optional[List[int]] = None,
    mlp_lambda: float = 1e-3,
    nl_lambda: Optional[float] = None,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: Optional[int] = None,
    seed: int = 0,
    zerosum: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    MLP-based nonlinear integration with a common fixed target representation Z.

    Steps:
      1. Stack anchors W = [A_1 ... A_c] and compute W = U Σ V^T.
      2. Fix Z_integ = U_{:, 1:t} (or B U in the zero-sum subspace).
      3. Train institution-specific MLPs g_k so that g_k(A_k) ~= Z_integ.
      4. Apply row-wise L2 normalization at the output layer.

    Returns:
      - per-institution projectors backed by trained MLPs,
      - the common fixed Z_integ,
      - leading singular values of the concatenated anchor matrix,
      - an empty gamma list for API compatibility.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("nonlinear_mlp を利用するには 'torch' が必要です") from e

    row_dim = anchors_inter[0].shape[0]
    if row_dim == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []
    if any(anchor.shape[0] != row_dim for anchor in anchors_inter):
        raise ValueError("nonlinear_mlp requires all anchor matrices to share the same number of rows.")

    hidden_dims = list(hidden_dims) if hidden_dims is not None else [500, 200]
    hidden_dims = [int(h) for h in hidden_dims if int(h) > 0]
    take = min(int(dim_integrate), row_dim)
    if take <= 0:
        return [], np.zeros((row_dim, 0)), np.zeros((0,)), []

    W_concat = np.hstack(anchors_inter)
    if zerosum and row_dim >= 2:
        B_zero = _zerosum_helmert_basis(row_dim)
        U_sub, S, _ = np.linalg.svd(B_zero.T @ W_concat, full_matrices=False)
        Z_integ = B_zero @ U_sub[:, :take]
    else:
        U, S, _ = np.linalg.svd(W_concat, full_matrices=False)
        Z_integ = U[:, :take]
    eigvals = S[:take]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = int(seed)
    # Backward compatibility: nl_lambda is accepted for legacy callers,
    # but nonlinear_mlp should use mlp_lambda going forward.
    lam_cfg = mlp_lambda if mlp_lambda is not None else nl_lambda
    lam = float(max(lam_cfg if lam_cfg is not None else 1e-3, 0.0))
    train_batch_size = None if batch_size is None else max(1, int(batch_size))

    class _ProjectorMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, widths: List[int]) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = int(input_dim)
            for width in widths:
                layers.append(nn.Linear(prev_dim, int(width)))
                layers.append(nn.ReLU())
                prev_dim = int(width)
            self.hidden = nn.Sequential(*layers)
            self.out = nn.Linear(prev_dim, int(output_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.hidden(x)
            y = self.out(h)
            return F.normalize(y, p=2, dim=1, eps=1e-12)

    target_tensor = torch.from_numpy(np.asarray(Z_integ, dtype=np.float32)).to(device)
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    losses: List[float] = []

    for inst_idx, anchor_inter_k in enumerate(anchors_inter):
        torch.manual_seed(base_seed + inst_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + inst_idx)

        X_anchor = np.asarray(anchor_inter_k, dtype=np.float32)
        input_dim = int(X_anchor.shape[1])
        model = _ProjectorMLP(input_dim=input_dim, output_dim=take, widths=hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        data_tensor = torch.from_numpy(X_anchor).to(device)

        effective_batch = data_tensor.shape[0] if train_batch_size is None else min(train_batch_size, data_tensor.shape[0])
        indices_full = torch.arange(data_tensor.shape[0], device=device)

        for epoch_idx in range(max(1, int(epochs))):
            model.train()
            if effective_batch >= data_tensor.shape[0]:
                batch_slices = [indices_full]
            else:
                perm = torch.randperm(data_tensor.shape[0], device=device)
                batch_slices = [perm[start:start + effective_batch] for start in range(0, data_tensor.shape[0], effective_batch)]

            for batch_idx in batch_slices:
                pred = model(data_tensor[batch_idx])
                target_batch = target_tensor[batch_idx]
                data_loss = torch.sum((pred - target_batch) ** 2)
                reg_loss = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    if "weight" in name:
                        reg_loss = reg_loss + torch.sum(param ** 2)
                loss = data_loss + lam * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch_idx == max(1, int(epochs)) - 1:
                losses.append(float(loss.detach().cpu().item()))

        model.eval()

        def _make_proj(model_ref: nn.Module):
            def projector(X: np.ndarray) -> np.ndarray:
                arr = np.asarray(X, dtype=np.float32)
                with torch.no_grad():
                    tensor = torch.from_numpy(arr).to(device)
                    out = model_ref(tensor).detach().cpu().numpy()
                return out

            return projector

        projs.append(_make_proj(model))

    return projs, Z_integ, eigvals, losses


def build_graph_nonlinear_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    L_within: Optional[np.ndarray],
    L_between: Optional[np.ndarray],
    g_type: Optional[str] = None,
    zerosum: bool = False,
    kernel_type: str = "rbf",
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    if L_within is None or L_between is None:
        raise ValueError("graph_nonlinear requires both L_within and L_between.")

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    Ks, Ps, mu_max_list = [], [], []
    kernel_type_key = (kernel_type or "rbf").lower()
    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        mu_max = None
        Ks.append(K)
        Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))
        mu_max_list.append(mu_max)

    M = sum((P - I_r).T @ (P - I_r) for P in Ps)
    M = (M + M.T) * 0.5
    
    eps = 1e-12
    tr_M  = np.trace(M)
    tr_Lw = np.trace(L_within)
    scale_Lw = tr_M / max(tr_Lw, eps)  # L_within のスケール
    
    tr_Lb = np.trace(L_between)
    n = L_between.shape[0]
    scale_Lb = n / max(tr_Lb, eps)  # tr(scale_Lb * L_between) ≒ tr(I) = n
    
    # Prepare both formulations
    # minimize: (M + μ L_within) z = γ (L_between + ε I) z
    
    A_min = M + graph_mu_align * scale_Lw * L_within
    B_min = scale_Lb * L_between + constraint_eps * np.eye(L_between.shape[0])
    A_min = (A_min + A_min.T) * 0.5
    B_min = (B_min + B_min.T) * 0.5
    print(" constraint_eps :", constraint_eps)
    print(" scale_Lw  :", scale_Lw)
    print(" scale_Lb  :", scale_Lb)
    print("graph_mu_align:", graph_mu_align)
    # maximize: L_between z = γ (M + μ L_within + ε I) z
    A_max = (scale_Lb * L_between + 0.0)
    B_max = (M + graph_mu_align  * scale_Lw * L_within + constraint_eps * np.eye(L_between.shape[0]))
    A_max = (A_max + A_max.T) * 0.5
    B_max = (B_max + B_max.T) * 0.5

    mode = (g_type or "").lower()
    use_max = ("maximize" in mode)
    print("use_max", use_max)
    A_use, B_use = (A_max, B_max) if use_max else (A_min, B_min)

    if zerosum:
        B_zero = _zerosum_helmert_basis(A_use.shape[0])
        A_use_tilde = B_zero.T @ A_use @ B_zero
        B_use_tilde = B_zero.T @ B_use @ B_zero
        eigvals_raw, eigvecs_sub = eigh(A_use_tilde, B_use_tilde)
    else:
        eigvals_raw, eigvecs_sub = eigh(A_use, B_use)
    # Select eigenvalues by mode
    order = np.argsort(eigvals_raw) if not use_max else np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs_sub.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    eigvecs = eigvecs_sub[:, select]
    if zerosum:
        Z_integ = B_zero @ eigvecs
    else:
        Z_integ = eigvecs

    for j in range(Z_integ.shape[1]):
        denom = float(Z_integ[:, j].T @ (B_use @ Z_integ[:, j]))
        if denom > 0:
            Z_integ[:, j] /= np.sqrt(denom)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(anchors_inter[i], B_k, gamma=gammas[i], kernel_type=kernel_type_key)
        projs.append(proj)

    # Z_integ の各列の総和を計算
    col_sums = np.sum(Z_integ, axis=0)
    # print で表示
    #print("Column sums:", col_sums)


    return projs, Z_integ, eigvals_selected, gammas


def _build_unlabeled_anchor_laplacian(
    anchors_inter: List[np.ndarray],
    k_neighbors: int,
) -> np.ndarray:
    """
    Build a label-agnostic Laplacian on anchors using an averaged k-NN graph
    over per-institution anchor intermediate representations.
    """
    if not anchors_inter:
        return np.zeros((0, 0))
    r = anchors_inter[0].shape[0]
    if r == 0:
        return np.zeros((0, 0))
    for A in anchors_inter:
        if A.shape[0] != r:
            raise ValueError("All anchor projections must share the same number of rows.")

    k_eff = max(1, min(int(k_neighbors), r - 1))
    if k_eff <= 0:
        return np.zeros((r, r))

    W_sum = np.zeros((r, r), dtype=float)
    for anchor_inst in anchors_inter:
        adjacency = _symmetric_knn_graph(anchor_inst, k_eff, metric="euclidean")
        if adjacency.size == 0:
            continue
        W_sum += adjacency

    if not np.any(W_sum):
        return np.zeros((r, r))

    W = W_sum / float(len(anchors_inter))
    d = W.sum(axis=1)
    L = np.diag(d) - W
    return L


def build_laplacian_nonlinear_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    anchor: np.ndarray,
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    kernel_type: str = "rbf",
    graph_mu_align: float = 1.0,
    laplacian_k: int = 10,
    zerosum: bool = False,
    regularization: str = "graph"
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Laplacian-regularized kernel (nonlinear) projectors with label-agnostic Laplacian.
    Uses a k-NN Laplacian over anchors scaled by graph_mu_align.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    Ks: List[np.ndarray] = []
    Ps: List[np.ndarray] = []
    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        Ks.append(K)
        Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))

    M = sum((P - I_r).T @ (P - I_r) for P in Ps)
    M = (M + M.T) * 0.5

    if regularization.lower() == "identity":
        print("Using identity regularization.")
        I = np.eye(M.shape[0])
        tr_M = float(np.trace(M))
        tr_I = float(np.trace(I))
        A = M + float(graph_mu_align) * (tr_M / max(tr_I, 1e-12)) * I
    elif regularization.lower() == "graph":
        L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=laplacian_k)
        if L_plain.shape != M.shape:
            A = M
        else:
            eps = 1e-12
            tr_M = float(np.trace(M))
            tr_L = float(np.trace(L_plain))
            scale_L = tr_M / max(tr_L, eps) if tr_L > 0 else 1.0
            A = M + float(graph_mu_align) * scale_L * L_plain
    else:
        A = M
    A = (A + A.T) * 0.5

    if zerosum:
        B_zero = _zerosum_helmert_basis(A.shape[0])
        A_tilde = B_zero.T @ A @ B_zero
        eigvals_raw, eigvecs_sub = eigh(A_tilde, np.eye(A_tilde.shape[0]))
    else:
        eigvals_raw, eigvecs_sub = eigh(A, np.eye(A.shape[0]))

    order = np.argsort(eigvals_raw)
    take = min(dim_integrate, eigvecs_sub.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    eigvecs = eigvecs_sub[:, select]

    if zerosum:
        Z_integ = B_zero @ eigvecs
    else:
        Z_integ = eigvecs

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(anchors_inter[i], B_k, gamma=gammas[i], kernel_type=kernel_type_key)
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_laplacian_nonlinear_new_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    anchor: np.ndarray,
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    kernel_type: str = "rbf",
    graph_mu_align: float = 1.0,
    laplacian_k: int = 10,
    zerosum: bool = False,
    regularization: str = "graph",
    K_normalization: bool = False,
    constraint_matrix: Optional[np.ndarray] = None,
    constraint_eps: float = 1e-9,
    L_within: Optional[np.ndarray] = None,
    L_between: Optional[np.ndarray] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Paper-matching Laplacian-regularized nonlinear integration.

    Builds:
      M_λ = λ * sum_k (K^(k) + λ I)^(-1)
      A = M_λ + μ * scale * L

    where L is an unlabeled k-NN Laplacian over anchors (graph), and scale is chosen
    so that tr(scale * L) ~= tr(M_λ).

    regularization options:
      - "identity": A = M_λ + μI (trace-matched)
      - "graph": A = M_λ + μL
      - "target-graph": A = M_λ + μL_within, with L_between used as mass (default)
      - "penal-target-graph": A = M_λ + μ(L_within - L_between)
    """
    def _sym(A: np.ndarray) -> np.ndarray:
        return (A + A.T) * 0.5

    def _solve_eig(
        A: np.ndarray,
        dim: int,
        *,
        zerosum_enabled: bool,
        B_mass: Optional[np.ndarray],
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        A = _sym(A)
        take = min(int(dim), A.shape[0])
        if take <= 0:
            return np.zeros((A.shape[0], 0)), np.zeros((0,))

        if zerosum_enabled and A.shape[0] >= 2:
            B0 = _zerosum_helmert_basis(A.shape[0])
            A_tilde = _sym(B0.T @ A @ B0)
            take_tilde = min(take, A_tilde.shape[0])
            if B_mass is None:
                eigvals, eigvecs = _smallest_eigh(A_tilde, take_tilde)
            else:
                B_mass = _sym(B_mass)
                B_tilde = _sym(B0.T @ B_mass @ B0)
                # Add stability term once in the reduced (zero-sum) subspace.
                B_tilde = B_tilde + float(eps) * np.eye(B_tilde.shape[0])
                eigvals, eigvecs = _smallest_eigh(A_tilde, take_tilde, B=B_tilde)
            Z = B0 @ eigvecs
            return Z, eigvals

        if B_mass is None:
            eigvals, eigvecs = _smallest_eigh(A, take)
        else:
            B_mass = _sym(B_mass)
            # Add stability term once in the original space.
            B_mass = B_mass + float(eps) * np.eye(B_mass.shape[0])
            eigvals, eigvecs = _smallest_eigh(A, take, B=B_mass)
        Z = eigvecs
        return Z, eigvals

    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []
    I_r = np.eye(r)
    lam = float(nl_lambda)
    eps = float(constraint_eps) if constraint_eps is not None else 1e-9
    eps = max(eps, 1e-12)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    Ss: List[np.ndarray] = []
    mu_max_list: List[Optional[float]] = []

    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])

        if K_normalization:
            mu_max = max(float(np.linalg.eigvalsh(K).max()), 1e-12)
            mu_max_list.append(mu_max)
            K = K / mu_max
        else:
            mu_max_list.append(None)

        try:
            S = np.linalg.inv(K + lam * I_r)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(K + lam * I_r)
        Ss.append(S)

    M_lambda = _sym(lam * sum(Ss))
    A = M_lambda
    reg_key = str(regularization or "graph").lower()
    if float(graph_mu_align) != 0.0:
        eps = 1e-12
        tr_M = float(np.trace(M_lambda))
        if reg_key == "identity":
            I = np.eye(M_lambda.shape[0])
            tr_I = float(np.trace(I))
            scale = tr_M / max(tr_I, eps) if tr_I > 0 else 1.0
            A = M_lambda + float(graph_mu_align) * scale * I
        elif reg_key == "graph":
            # Label-agnostic graph: always build an unlabeled anchor Laplacian.
            L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=int(laplacian_k))
            if L_plain.shape == M_lambda.shape:
                L_plain = _sym(L_plain)
                tr_L = float(np.trace(L_plain))
                scale = tr_M / max(tr_L, eps) if tr_L > 0 else 1.0
                A = M_lambda + float(graph_mu_align) * scale * L_plain
        elif reg_key in {"target-graph", "target_graph"}:
            if L_within is None:
                raise ValueError("target-graph regularization requires L_within.")
            Lw = np.asarray(L_within, dtype=float)
            if Lw.shape != M_lambda.shape:
                raise ValueError(f"target-graph requires L_within shape {M_lambda.shape} but got {Lw.shape}")
            Lw = _sym(Lw)
            tr_Lw = float(np.trace(Lw))
            if tr_M > eps:
                M_use = M_lambda / tr_M
            else:
                M_use = M_lambda
            if tr_Lw > eps:
                Lw_use = Lw / tr_Lw
            else:
                Lw_use = Lw
            A = M_use + float(graph_mu_align) * Lw_use
        elif reg_key in {"penal-target-graph", "penal_target_graph"}:
            if L_within is None or L_between is None:
                raise ValueError("penal-target-graph regularization requires both L_within and L_between.")
            Lw = np.asarray(L_within, dtype=float)
            Lb = np.asarray(L_between, dtype=float)
            if Lw.shape != M_lambda.shape or Lb.shape != M_lambda.shape:
                raise ValueError(
                    f"penal-target-graph requires L_within/L_between shape {M_lambda.shape} but got {Lw.shape}/{Lb.shape}"
                )
            Lw = _sym(Lw)
            Lb = _sym(Lb)
            tr_Lw = float(np.trace(Lw))
            tr_Lb = float(np.trace(Lb))
            scale_w = tr_M / max(tr_Lw, eps) if tr_Lw > 0 else 1.0
            scale_b = tr_M / max(tr_Lb, eps) if tr_Lb > 0 else 1.0
            A = M_lambda + float(graph_mu_align) * (scale_w * Lw - scale_b * Lb)

    A = _sym(A)

    mass = None
    if reg_key in {"target-graph", "target_graph"} and constraint_matrix is None:
        # Use L_between as the mass matrix (constraint) by default, scaled so that tr(B) ~= n.
        if L_between is None:
            raise ValueError("target-graph regularization requires L_between.")
        Lb = np.asarray(L_between, dtype=float)
        if Lb.shape != (r, r):
            raise ValueError(f"L_between must be shape {(r, r)} but got {Lb.shape}")
        Lb = _sym(Lb)
        mass = Lb
    if constraint_matrix is not None:
        mass = np.asarray(constraint_matrix, dtype=float)
        if mass.shape != (r, r):
            raise ValueError(f"constraint_matrix must be shape {(r, r)} but got {mass.shape}")

    Z_integ, eigvals_selected = _solve_eig(
        A,
        dim_integrate,
        zerosum_enabled=bool(zerosum),
        B_mass=mass,
        eps=eps,
    )

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, (S, anchor_inter_k) in enumerate(zip(Ss, anchors_inter)):
        C_k = S @ Z_integ
        proj = make_kernel_integrator(
            anchor_inter_k,
            C_k,
            gamma=gammas[i],
            normalize=K_normalization,
            mu_max=mu_max_list[i],
            kernel_type=kernel_type_key,
        )
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_laplacian_nonlinear_nonridge_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    anchor: np.ndarray,
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    kernel_type: str = "rbf",
    graph_mu_align: float = 1.0,
    laplacian_k: int = 10,
    zerosum: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Laplacian-regularized kernel (nonlinear) projectors sharing the same
    graph construction and scaling as build_laplacian_nonlinear_projectors,
    but fitting per-institution mappings without ridge (K^+ Z instead of
    (K + λI)^{-1} Z). λ is still used only for determining Z via M_λ.
    """
    if not anchors_inter:
        return [], np.zeros((0, 0)), np.zeros((0,)), []

    r = anchors_inter[0].shape[0]
    if r == 0:
        return [], np.zeros((0, 0)), np.zeros((0,)), []
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)
    kernel_type_key = (kernel_type or "rbf").lower()

    lam_raw = nl_lambda
    lam_is_infinite = False
    lam_value: Optional[float] = None
    if isinstance(lam_raw, str):
        lam_str = lam_raw.strip()
        if lam_str == "∞" or lam_str.lower() in {"inf", "infinity"}:
            lam_is_infinite = True
        else:
            lam_value = float(lam_str)
    else:
        lam_value = float(lam_raw)
        if math.isinf(lam_value):
            lam_is_infinite = True
            lam_value = None

    Ks: List[np.ndarray] = []
    M_terms: List[np.ndarray] = []
    for i, anchor_inter_k in enumerate(anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        Ks.append(K)
        if lam_is_infinite:
            term = K
        else:
            lam = float(lam_value if lam_value is not None else 0.0)
            if lam == 0:
                # For constructing M_λ we allow nl_lambda=0; use a
                # pseudo-inverse fallback to avoid singular-matrix failures.
                K_reg_inv = np.linalg.pinv(K)
            else:
                try:
                    K_reg_inv = np.linalg.inv(K + lam * I_r)
                except np.linalg.LinAlgError:
                    K_reg_inv = np.linalg.pinv(K + lam * I_r)
            term = K @ K_reg_inv
        M_terms.append(term)

    M = sum(M_terms)
    M = (M + M.T) * 0.5

    L_plain = _build_unlabeled_anchor_laplacian(anchors_inter, k_neighbors=laplacian_k)
    if L_plain.shape != M.shape:
        A = M
    else:
        eps = 1e-12
        tr_M = float(np.trace(M))
        tr_L = float(np.trace(L_plain))
        scale_L = tr_M / max(tr_L, eps) if tr_L > 0 else 1.0
        A = M + float(graph_mu_align) * scale_L * L_plain
    A = (A + A.T) * 0.5

    if zerosum:
        B_zero = _zerosum_helmert_basis(A.shape[0])
        A_tilde = B_zero.T @ A @ B_zero
        eigvals_raw, eigvecs_sub = eigh(A_tilde, np.eye(A_tilde.shape[0]))
    else:
        eigvals_raw, eigvecs_sub = eigh(A, np.eye(A.shape[0]))

    order = np.argsort(eigvals_raw)
    take = min(dim_integrate, eigvecs_sub.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    eigvecs = eigvecs_sub[:, select]

    if zerosum:
        Z_integ = B_zero @ eigvecs
    else:
        Z_integ = eigvecs

    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        try:
            K_pinv = np.linalg.pinv(K)
        except np.linalg.LinAlgError:
            K_pinv = np.linalg.pinv(K + 1e-8 * np.eye(K.shape[0]))
        B_k = K_pinv @ Z_integ
        proj = make_kernel_integrator(anchors_inter[i], B_k, gamma=gammas[i], kernel_type=kernel_type_key)
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_nonlinear_projectors_maximize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    K_normalization: bool = False,
    nl_lambda: float = 1e-2,
    lw_alpha: float = 0.0,
    L_within: Optional[np.ndarray] = None,
    L_between: Optional[np.ndarray] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Maximize-mode variant of build_nonlinear_projectors.
    Selects largest eigenvalues of the same symmetric objective.
    """
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    Ks, Ps, mu_max_list = [], [], []
    for i, anchor_inter_k in enumerate(anchors_inter):
        K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        if K_normalization:
            mu_max = max(np.linalg.eigvalsh(K).max(), 1e-12)
            mu_max_list.append(mu_max)
            K = K / mu_max
        else:
            mu_max_list.append(None)
        Ks.append(K)
        Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))

    M = sum((P - I_r).T @ (P - I_r) for P in Ps)
    trace_M = np.trace(M)
    if trace_M > 1e-9:
        M = M / trace_M

    if lw_alpha == 0:
        Q = (M + M.T) * 0.5
    else:
        if L_within is None or L_between is None:
            raise ValueError("Non-zero lw_alpha requires both L_within and L_between Laplacians.")
        Q = (M + M.T) * 0.5 + lw_alpha * (L_within - L_between)

    # Maximize mode: solve I z = λ Q z and take descending λ
    # This selects directions with small eigenvalues of Q but expressed via
    # generalized eigenproblem for numerical symmetry with other routines.
    eps = 1e-9
    A_use = np.eye(Q.shape[0])
    B_use = (Q + Q.T) * 0.5 + eps * np.eye(Q.shape[0])
    eigvals_raw, eigvecs = _solve_gep_regularized(A_use, B_use, orth_ver=False)
    order = np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    Z_integ = eigvecs[:, select]
    # Normalize by B metric for stability
    for j in range(Z_integ.shape[1]):
        denom = float(Z_integ[:, j].T @ (B_use @ Z_integ[:, j]))
        if denom > 0:
            Z_integ[:, j] /= np.sqrt(denom)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(anchors_inter[i], B_k, gamma=gammas[i], kernel_type=kernel_type_key)
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


def build_graph_nonlinear_X_projectors(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    graph_L_within: Optional[np.ndarray],
    graph_L_between: Optional[np.ndarray],
    g_type: Optional[str] = None,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    """
    Graph-regularized nonlinear integration using the Schur complement on
    (anchor + data) blocks per institution.

    Theory mapping (variables use repo naming):
    - For each k, build kernel blocks K_SS, K_SX, K_XS, K_XX with the same RBF
      kernel parameter gamma_k.
    - Hat-matrix approximation using only K_SS inversion:
        P_SS = K_SS (K_SS + λ I_r)^{-1}
        P_SX = P_SS K_SX,  P_XS = K_XS (K_SS + λ I_r)^{-1}
        P_XX = K_XS (K_SS + λ I_r)^{-1} K_SX
    - Block components of M_k = (I - H_k)^2:
        A_k = (I_r - P_SS)^2 + P_SX P_XS
        B_k = - (I_r - P_SS) P_SX - P_SX (I_{n_k} - P_XX)
        C_k = P_XS P_SX + (I_{n_k} - P_XX)^2
    - Aggregate A = Σ A_k, B = [B_1 ... B_c], C = blkdiag(C_k) and
      Ψ = C - B^T A^{-1} B.
    - Solve the GEP with graph Laplacians over actual data samples X:
        minimize: (Ψ + μ L_w) w = λ (L_b + ε I) w
        maximize: L_b w = λ (Ψ + μ L_w + ε I) w
    - Recover anchor-target u = -A^{-1} B w and build Z^(k) = [u; w^(k)].
      Projector for institution k uses training points T_k = [S~^(k); X^(k)]:
        B_k = (K(T_k,T_k) + λ I)^{-1} Z^(k)
        proj_k(X) = κ(X, T_k) B_k.
    """
    if graph_L_within is None or graph_L_between is None:
        raise ValueError("graph_nonlinear_X requires graph Laplacians over data (graph_L_within/graph_L_between).")

    if not anchors_inter or not Xs_train_inter:
        return [], np.zeros((0, 0)), np.array([]), []

    if len(anchors_inter) != len(Xs_train_inter):
        raise ValueError("anchors_inter and Xs_train_inter must have the same length for graph_nonlinear_X.")

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    # Determine per-institution gamma
    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    # Build Schur components A, B, C
    A_accum: Optional[np.ndarray] = None
    B_blocks: List[np.ndarray] = []
    C_blocks: List[np.ndarray] = []
    n_total = 0

    inv_SS_list: List[np.ndarray] = []
    n_list: List[int] = []

    for k, (S_k, X_k) in enumerate(zip(anchors_inter, Xs_train_inter)):
        gamma = gammas[k]
        n_k = X_k.shape[0]
        n_list.append(n_k)
        n_total += n_k

        # Kernel blocks
        K_SS = rbf_kernel(S_k, S_k, gamma=gamma)
        K_SX = rbf_kernel(S_k, X_k, gamma=gamma)  # r × n_k
        K_XS = K_SX.T                                  # n_k × r
        K_XX = rbf_kernel(X_k, X_k, gamma=gamma)

        inv_SS = np.linalg.inv(K_SS + nl_lambda * I_r)
        inv_SS_list.append(inv_SS)
        P_SS = K_SS @ inv_SS
        P_SX = P_SS @ K_SX
        P_XS = K_XS @ inv_SS
        P_XX = K_XS @ inv_SS @ K_SX

        I_nk = np.eye(n_k)
        A_k = (I_r - P_SS) @ (I_r - P_SS) + P_SX @ P_XS
        B_k = - (I_r - P_SS) @ P_SX - P_SX @ (I_nk - P_XX)
        C_k = P_XS @ P_SX + (I_nk - P_XX) @ (I_nk - P_XX)

        A_accum = A_k if A_accum is None else (A_accum + A_k)
        B_blocks.append(B_k)
        C_blocks.append(C_k)

    if A_accum is None:
        return [], np.zeros((0, 0)), np.array([]), gammas

    A = (A_accum + A_accum.T) * 0.5
    A_ridge = A + 1e-9 * np.eye(A.shape[0])
    B = np.hstack(B_blocks) if B_blocks else np.zeros((r, 0))
    C = block_diag(*C_blocks) if C_blocks else np.zeros((0, 0))

    # Ψ = C - B^T A^{-1} B via solve
    A_inv_B = np.linalg.solve(A_ridge, B) if B.size else np.zeros_like(B)
    Psi = C - B.T @ A_inv_B
    Psi = (Psi + Psi.T) * 0.5

    # Validate graph Laplacian sizes
    Lw = np.asarray(graph_L_within)
    Lb = np.asarray(graph_L_between)
    if Lw.shape != (n_total, n_total) or Lb.shape != (n_total, n_total):
        raise ValueError("graph Laplacian sizes must match total number of samples across institutions.")

    # Solve generalized eigenproblem
    A_min = Psi + graph_mu_align * Lw
    B_min = Lb + constraint_eps * np.eye(Lb.shape[0])
    A_min = (A_min + A_min.T) * 0.5
    B_min = (B_min + B_min.T) * 0.5

    A_max = (Lb + 0.0)
    B_max = Psi + graph_mu_align * Lw + constraint_eps * np.eye(Lb.shape[0])
    A_max = (A_max + A_max.T) * 0.5
    B_max = (B_max + B_max.T) * 0.5

    mode = (g_type or "").lower()
    use_max = ("maximize" in mode)
    A_use, B_use = (A_max, B_max) if use_max else (A_min, B_min)
    # SPD projection for B to stabilize generalized eigen solve
    B_use = _nearest_spd(B_use, min_eig=max(constraint_eps, 1e-9))

    eigvals_raw, eigvecs = _solve_gep_regularized(A_use, B_use, orth_ver=False)
    order = np.argsort(eigvals_raw) if not use_max else np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    W = eigvecs[:, select]

    for j in range(W.shape[1]):
        denom = float(W[:, j].T @ (B_use @ W[:, j]))
        if denom > 0:
            W[:, j] /= np.sqrt(denom)

    # Recover u = -A^{-1} B W
    U = -np.linalg.solve(A_ridge, B @ W) if B.size else np.zeros((r, take))

    # Build per-institution projectors
    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    row_offset = 0
    for k, (S_k, X_k) in enumerate(zip(anchors_inter, Xs_train_inter)):
        n_k = n_list[k]
        w_k = W[row_offset:row_offset + n_k, :]
        row_offset += n_k
        Z_k = np.vstack([U, w_k])  # (r + n_k) × m

        T_k = np.vstack([S_k, X_k])
        gamma = gammas[k]
        K_tk = rbf_kernel(T_k, T_k, gamma=gamma)
        ridge = nl_lambda * np.eye(K_tk.shape[0])
        B_k = np.linalg.solve(K_tk + ridge, Z_k)
        proj = make_kernel_integrator(T_k, B_k, gamma=gamma)
        projs.append(proj)

    Z_integ_like = U
    return projs, Z_integ_like, eigvals_selected, gammas


def build_graph_nonlinear_X_projectors_maximize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    graph_L_within: Optional[np.ndarray],
    graph_L_between: Optional[np.ndarray],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    return build_graph_nonlinear_X_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        nl_lambda=nl_lambda,
        graph_mu_align=graph_mu_align,
        constraint_eps=constraint_eps,
        graph_L_within=graph_L_within,
        graph_L_between=graph_L_between,
        g_type="graph_nonlinear_X_maximize",
    )


def build_graph_nonlinear_X_projectors_minimize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    graph_L_within: Optional[np.ndarray],
    graph_L_between: Optional[np.ndarray],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    return build_graph_nonlinear_X_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        nl_lambda=nl_lambda,
        graph_mu_align=graph_mu_align,
        constraint_eps=constraint_eps,
        graph_L_within=graph_L_within,
        graph_L_between=graph_L_between,
        g_type="graph_nonlinear_X_minimize",
    )
def build_graph_nonlinear_projectors_maximize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    L_within: Optional[np.ndarray],
    L_between: Optional[np.ndarray],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    return build_graph_nonlinear_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        nl_lambda=nl_lambda,
        graph_mu_align=graph_mu_align,
        constraint_eps=constraint_eps,
        L_within=L_within,
        L_between=L_between,
        g_type="graph_nonlinear_maximize",
    )


def build_graph_nonlinear_projectors_minimize(
    anchors_inter: List[np.ndarray],
    Xs_train_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    gamma_type: str = "auto",
    gamma_ratio_krr: float = 1.0,
    nl_lambda: float = 1e-2,
    graph_mu_align: float = 1.0,
    constraint_eps: float = 1e-6,
    L_within: Optional[np.ndarray],
    L_between: Optional[np.ndarray],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    return build_graph_nonlinear_projectors(
        anchors_inter,
        Xs_train_inter,
        dim_integrate,
        gamma_type=gamma_type,
        gamma_ratio_krr=gamma_ratio_krr,
        nl_lambda=nl_lambda,
        graph_mu_align=graph_mu_align,
        constraint_eps=constraint_eps,
        L_within=L_within,
        L_between=L_between,
        g_type="graph_nonlinear_minimize",
    )
