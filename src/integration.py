from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import block_diag, eigh
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class IntegrationModel:
    kind: str  # 'linear', 'rbf', 'rbf+linear'
    # linear
    G: Optional[np.ndarray] = None  # (d_k, p)

    # rbf
    B: Optional[np.ndarray] = None  # (r, p)
    S_train: Optional[np.ndarray] = None  # (r, d_k)
    gamma: Optional[float] = None
    mu_max: Optional[float] = None
    lam: Optional[float] = None

    # rbf+linear
    alpha: Optional[np.ndarray] = None  # (r, p)
    beta: Optional[np.ndarray] = None   # (d_k, p)
    mu_max_lin: Optional[float] = None
    coeff_mode: Optional[str] = None


class Integrator:
    """Build integration functions for each method and return (Z, models, meta)."""

    @staticmethod
    def imakura(anchors_inter: List[np.ndarray], p_hat: int) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        # Stack anchors horizontally and take SVD to get Z
        centralized_anchor = np.hstack(anchors_inter)  # r x sum(d_k)
        U, _, _ = np.linalg.svd(centralized_anchor)
        U = U[:, :p_hat]
        Z = U.T  # p x r

        models: List[IntegrationModel] = []
        for S in anchors_inter:
            # S: r x d
            Gp = Z @ pinv(S.T)      # p x d
            G = Gp.T                # d x p
            models.append(IntegrationModel(kind='linear', G=G))
        return Z, models, {}

    @staticmethod
    def targetvec(anchors_inter: List[np.ndarray], p_hat: int) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        # Build C_tildeS = m I_r - sum S S^+
        m = len(anchors_inter)
        r = anchors_inter[0].shape[0]
        I_r = np.eye(r)
        C = m * I_r
        for S in anchors_inter:
            C -= S @ pinv(S)
        # eigen (symmetric)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals[eigvals < 0] = 0.0
        Z = eigvecs[:, :p_hat]  # r x p

        models: List[IntegrationModel] = []
        for S in anchors_inter:
            G = pinv(S) @ Z  # (d, p)
            models.append(IntegrationModel(kind='linear', G=G))
        return Z, models, {'eigvals': eigvals[:p_hat]}

    @staticmethod
    def gep(anchors_inter: List[np.ndarray], p_hat: int, lambda_gen: float = 0.0, orth_ver: bool = False) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        # Build W_tilde and B_tilde
        W_tilde = np.hstack(anchors_inter)  # r x sum(d_k)
        blocks = [S.T @ S for S in anchors_inter]
        eps = 1e-6
        B_tilde = block_diag(*blocks) + eps * np.eye(sum(S.shape[1] for S in anchors_inter))

        A_tilde = 2 * len(anchors_inter) * B_tilde - 2 * (W_tilde.T @ W_tilde) + lambda_gen * np.eye(W_tilde.shape[1])
        if orth_ver:
            eigvals, eigvecs = eigh(A_tilde)
        else:
            eigvals, eigvecs = eigh(A_tilde, B_tilde)
        order = np.argsort(eigvals)
        V_sel = eigvecs[:, order[:p_hat]]  # sum(d_k) x p

        # Slice per institution to get Gk
        models: List[IntegrationModel] = []
        cum = np.cumsum([0] + [S.shape[1] for S in anchors_inter])
        for k in range(len(anchors_inter)):
            Gk = V_sel[cum[k]:cum[k+1], :]  # d_k x p
            models.append(IntegrationModel(kind='linear', G=Gk))
        # Anchors-side target for visualization/consistency
        Z = W_tilde @ V_sel  # r x p
        return Z, models, {'eigvals': eigvals[order[:p_hat]]}

    @staticmethod
    def odc(anchors_inter: List[np.ndarray]) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        # Orthogonal Procrustes to align each to anchor_1 space
        anchor_1 = anchors_inter[0]
        models: List[IntegrationModel] = []
        for S in anchors_inter:
            M = S.T @ anchor_1
            U, _, Vh = np.linalg.svd(M, full_matrices=False)
            Gk = U @ Vh  # d_k x d_1
            models.append(IntegrationModel(kind='linear', G=Gk))
        Z = anchor_1  # for reference
        return Z, models, {}

    @staticmethod
    def nonlinear(anchors_inter: List[np.ndarray], anchors_test_inter: List[np.ndarray], Xs_train_inter: List[np.ndarray], p_hat: int, config, L_within: Optional[np.ndarray], L_between: Optional[np.ndarray]) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        r = anchors_inter[0].shape[0]
        I_r = np.eye(r)

        # gamma per institution
        gammas = []
        if getattr(config, 'gamma_type', 'auto') == 'auto':
            for S in anchors_inter:
                gammas.append(1.0 / S.shape[1])
        elif getattr(config, 'gamma_type', 'auto') == 'X_tuning':
            from src.utils import self_tuning_gamma
            for X in Xs_train_inter:
                g = self_tuning_gamma(X, standardize=False, k=3, summary='median')
                g *= getattr(config, 'gamma_ratio_krr', 1.0)
                gammas.append(g)
        else:
            for S in anchors_inter:
                gammas.append(1.0 / S.shape[1])

        lam = getattr(config, 'nl_lambda', 1e-2)
        K_norm = bool(getattr(config, 'K_normalization', False))

        Ks = []
        mu_list = []
        for S, g in zip(anchors_inter, gammas):
            K = rbf_kernel(S, S, gamma=g)
            if K_norm:
                mu = max(np.linalg.eigvalsh(K).max(), 1e-12)
                K = K / mu
                mu_list.append(mu)
            else:
                mu_list.append(1.0)
            Ks.append(K)

        Ps = [K @ np.linalg.inv(K + lam * I_r) for K in Ks]
        M = sum((P - I_r).T @ (P - I_r) for P in Ps)
        tr = np.trace(M)
        if tr > 1e-9:
            M = M / tr

        lw_alpha = float(getattr(config, 'lw_alpha', 0) or 0)
        lb_beta = float(getattr(config, 'lb_beta', 0) or 0)
        Q = M.copy()
        if L_within is not None:
            Q += lw_alpha * L_within
        if L_between is not None:
            Q -= lb_beta * L_between
        Q = (Q + Q.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(Q)
        eigvals[eigvals < 0] = 0.0
        Z = eigvecs[:, eigvals.argsort()[:p_hat]]  # r x p
        # normalize columns
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz

        models: List[IntegrationModel] = []
        for K, S, g, mu in zip(Ks, anchors_inter, gammas, mu_list):
            Bk = np.linalg.inv(K + lam * I_r) @ Z
            models.append(IntegrationModel(kind='rbf', B=Bk, S_train=S, gamma=g, mu_max=mu, lam=lam))

        meta = {'eigvals': eigvals[:p_hat]}
        return Z, models, meta

    @staticmethod
    def nonlinear_plus_linear(anchors_inter: List[np.ndarray], Xs_train_inter: List[np.ndarray], p_hat: int, config) -> Tuple[np.ndarray, List[IntegrationModel], dict]:
        r = anchors_inter[0].shape[0]
        I_r = np.eye(r)

        gammas = []
        if getattr(config, 'gamma_type', 'auto') == 'auto':
            for S in anchors_inter:
                gammas.append(1.0 / S.shape[1])
        elif getattr(config, 'gamma_type', 'auto') == 'X_tuning':
            from src.utils import self_tuning_gamma
            for X in Xs_train_inter:
                g = self_tuning_gamma(X, standardize=False, k=3, summary='median')
                gammas.append(g)
        else:
            for S in anchors_inter:
                gammas.append(1.0 / S.shape[1])

        lam = getattr(config, 'nl_lambda', 1e-2)

        Ks = []
        Ps_lambda = []
        mu_list = []

        USE_FIRST_ORDER = bool(lam >= 10.0)

        # Build P_lambda per institution
        for S, g in zip(anchors_inter, gammas):
            K_raw = rbf_kernel(S, S, gamma=g)
            mu = max(np.linalg.eigvalsh(K_raw).max(), 1e-12)
            K = K_raw / mu
            mu_list.append(mu)

            P_lin = S  # r x d
            if USE_FIRST_ORDER:
                G = P_lin.T @ P_lin
                G_inv = np.linalg.pinv(G)
                P_proj = P_lin @ G_inv @ P_lin.T
                P1 = K - K @ P_proj - P_proj @ K + P_proj @ K @ P_proj
                P_lambda = P_proj + (1.0 / lam) * P1
                coeff = ('first_order', (G_inv, P_proj))
            else:
                A_inv = np.linalg.inv(K + lam * I_r)
                try:
                    M = np.linalg.inv(P_lin.T @ A_inv @ P_lin)
                except np.linalg.LinAlgError:
                    M = np.linalg.pinv(P_lin.T @ A_inv @ P_lin)
                P_lambda = (K @ A_inv + (P_lin - K @ A_inv @ P_lin) @ M @ (P_lin.T @ A_inv))
                coeff = ('exact', (A_inv, M))

            Ks.append((K, P_lin, coeff, mu))
            Ps_lambda.append(P_lambda)

        M_tot = sum((P - I_r).T @ (P - I_r) for P in Ps_lambda)
        M_sym = 0.5 * (M_tot + M_tot.T)
        eigvals, eigvecs = np.linalg.eigh(M_sym)
        Z = eigvecs[:, eigvals.argsort()[:p_hat]]
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz

        models: List[IntegrationModel] = []
        for (K, P_lin, coeff, mu), S, g in zip(Ks, anchors_inter, gammas):
            mode, pack = coeff
            if mode == 'exact':
                A_inv, M = pack
                beta = M @ (P_lin.T @ A_inv @ Z)    # d x p
                alpha = A_inv @ (Z - P_lin @ beta)  # r x p
                models.append(IntegrationModel(kind='rbf+linear', alpha=alpha, beta=beta, mu_max=mu, coeff_mode=mode, S_train=S, gamma=g))
            else:
                G_inv, P_proj = pack
                beta0 = G_inv @ (P_lin.T @ Z)
                r0 = Z - P_lin @ beta0
                beta1 = G_inv @ (P_lin.T @ (K @ r0))
                beta = beta0 + (1.0 / lam) * beta1
                alpha = (1.0 / lam) * r0
                models.append(IntegrationModel(kind='rbf+linear', alpha=alpha, beta=beta, mu_max=mu, coeff_mode=mode, S_train=S, gamma=g))

        return Z, models, {'eigvals': eigvals[:p_hat]}
