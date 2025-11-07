from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import block_diag, eigh
from sklearn.metrics.pairwise import rbf_kernel

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
    Returns (projs_per_institution, Z_integ (r×m_inter), g_abs_sum).
    """
    centralized_anchor = np.hstack(anchors_inter)  # r × sum d_k
    U, _, _ = np.linalg.svd(centralized_anchor)
    U = U[:, :dim_integrate]

    Z_integ = U  # r × m_inter (retain for Z_integ)
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
    Returns (projs_per_institution, Z_integ (r×m_inter)).
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


def build_gep_projectors(
    anchors_inter: List[np.ndarray],
    dim_integrate: int,
    *,
    lambda_gen: float = 0.0,
    orth_ver: bool = False,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """
    Generalized eigen problem based projector builders.
    Returns (projs_per_institution, metrics_dict) where metrics contains:
      - V_sel, lambdas, g_abs_sum, sum_objective_function, g_norm_val_gep,
        g_mean_var, g_condition_number
    """
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    # Build W and B
    W_s_tilde = np.hstack(anchors_inter)
    blocks = [anchor_inter_k.T @ anchor_inter_k for anchor_inter_k in anchors_inter]
    epsilon = 1e-6
    B_s_tilde = blocks[0]
    for b in blocks[1:]:
        B_s_tilde = block_diag(B_s_tilde, b)
    B_s_tilde = B_s_tilde + epsilon * np.eye(B_s_tilde.shape[0])

    A_s_tilde = 2 * c * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde) + lambda_gen * np.eye(W_s_tilde.shape[1])

    if orth_ver:
        eigvals, eigvecs = eigh(A_s_tilde)
    else:
        eigvals, eigvecs = eigh(A_s_tilde, B_s_tilde)
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
    Returns (projs_per_institution, Z_integ (r×m_inter), eigvals (ascending)).
    """
    c = len(anchors_inter)
    r = anchors_inter[0].shape[0]
    I_r = np.eye(r)

    gammas: List[float] = []
    if gamma_type == "auto":
        for anchor_inter_k in anchors_inter:
            gammas.append(1.0 / anchor_inter_k.shape[1])
    elif gamma_type == "X_tuning":
        for X_tr in Xs_train_inter:
            gamma = self_tuning_gamma(X_tr, standardize=False, k=3, summary='median')
            gamma *= gamma_ratio_krr
            gammas.append(float(gamma))
    elif gamma_type == "fixed":
        for X_tr in Xs_train_inter:
            gamma =  gamma_ratio_krr
            gammas.append(float(gamma))
    else:
        # fallback
        for anchor_inter_k in anchors_inter:
            gammas.append(1.0 / anchor_inter_k.shape[1])

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
