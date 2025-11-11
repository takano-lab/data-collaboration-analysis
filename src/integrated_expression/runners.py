from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import block_diag, eigh
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

from src.dimensionality_reduction import self_tuning_gamma

# --- Basic projector factories ---

def make_linear_integrator(G_k: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return projector X -> X @ G_k (linear right-multiplication)."""
    def projector(X: np.ndarray) -> np.ndarray:
        return X @ G_k
    return projector


def make_kernel_integrator(
    S_train: np.ndarray,
    B_k: np.ndarray,
    *,
    gamma: float,
    normalize: bool = False,
    mu_max: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return kernel projector X -> K(X, S_train) @ B_k with optional spectral normalization."""
    def projector(X: np.ndarray) -> np.ndarray:
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

    eigvals, eigvecs = np.linalg.eigh(C_tildeS)
    eigvals[eigvals < 0] = 0.0
    Z_integ = eigvecs[:, :dim_integrate]

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for anchor_inter_k in anchors_inter:
        proj, _Gk = compute_linear_integrator_from_Z_anchor(Z_integ, anchor_inter_k)
        projs.append(proj)
    return projs, Z_integ


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
    B = A_w + mu_align * A_align + lambda_rkhs * C_H
    B = (B + B.T) * 0.5 + (stability_eps + mu_align * 1e-9) * np.eye(B.shape[0])

    eigvals, eigvecs = _solve_gep_regularized(A_b, B, orth_ver=False)
    take = min(dim_integrate, eigvecs.shape[1])
    order = np.argsort(eigvals)[::-1]
    select = order[:take]
    eigvals_selected = eigvals[select]
    Alpha_stack = eigvecs[:, select]

    for j in range(Alpha_stack.shape[1]):
        vec = Alpha_stack[:, j]
        denom = float(vec.T @ (B @ vec))
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


def build_nonlinear_projectors(
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
    Kernel (nonlinear) based projector builders.
    Returns (projs_per_institution, Z_integ (r�~m_inter), eigvals (ascending)).
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
        Q = (M + M.T) * 0.5 + lw_alpha *(L_within - L_between)
    eigvals_raw, eigvecs = np.linalg.eigh(Q)
    eigvals_raw[eigvals_raw < 0] = 0.0
    order = np.argsort(eigvals_raw)
    eigvals_selected = eigvals_raw[order[:dim_integrate]]
    Z_integ = eigvecs[:, order[:dim_integrate]]
    for j in range(Z_integ.shape[1]):
        nz = np.linalg.norm(Z_integ[:, j])
        if nz > 0:
            Z_integ[:, j] /= nz

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(
            anchors_inter[i], B_k, gamma=gammas[i], normalize=K_normalization, mu_max=mu_max_list[i]
        )
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas


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
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray, List[float]]:
    if L_within is None or L_between is None:
        raise ValueError("graph_nonlinear requires both L_within and L_between.")

    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    gammas = _determine_kernel_gammas(anchors_inter, Xs_train_inter, gamma_type, gamma_ratio_krr)

    Ks, Ps, mu_max_list = [], [], []
    for i, anchor_inter_k in enumerate(anchors_inter):
        K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gammas[i])
        mu_max = None
        Ks.append(K)
        Ps.append(K @ np.linalg.inv(K + nl_lambda * I_r))
        mu_max_list.append(mu_max)

    M = sum((P - I_r).T @ (P - I_r) for P in Ps)
    M = (M + M.T) * 0.5

    A = M + graph_mu_align * L_within
    B = L_between + constraint_eps * np.eye(L_between.shape[0])
    B = (B + B.T) * 0.5

    eigvals_raw, eigvecs = eigh(A, B)
    order = np.argsort(eigvals_raw)[::-1]
    take = min(dim_integrate, eigvecs.shape[1])
    select = order[:take]
    eigvals_selected = eigvals_raw[select]
    Z_integ = eigvecs[:, select]

    for j in range(Z_integ.shape[1]):
        denom = float(Z_integ[:, j].T @ (B @ Z_integ[:, j]))
        if denom > 0:
            Z_integ[:, j] /= np.sqrt(denom)

    projs: List[Callable[[np.ndarray], np.ndarray]] = []
    for i, K in enumerate(Ks):
        B_k = np.linalg.inv(K + nl_lambda * I_r) @ Z_integ
        proj = make_kernel_integrator(anchors_inter[i], B_k, gamma=gammas[i])
        projs.append(proj)

    return projs, Z_integ, eigvals_selected, gammas

