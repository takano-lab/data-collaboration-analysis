from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Tuple, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import Config
from src.common import IntegratedArtifacts, IntermediateArtifacts
from .integrate_metrics import evaluate_nonlinearity_indices, integrate_metrics
from .runners import (
    build_gep_projectors,
    build_gep2_projectors,
    build_imakura_projectors,
    build_kernel_gep_projectors,
    build_kernel_graph_gep_projectors,
    build_graph_nonlinear_projectors,
    build_nonlinear_projectors,
    build_odc_projectors,
    build_targetvec_projectors,
)

logger = TypeVar("logger")
IntegrationRunner = Callable[["IntegratedExpressionBuilder"], Tuple[List, Dict[str, object]]]


class IntegratedExpressionBuilder:
    """
    Consumes intermediate artifacts and produces integrated (G-stage) representations.
    """

    def __init__(self, *, config: Config, logger: logger) -> None:
        self.config = config
        self.logger = logger

        # Dataset-level attributes (populated from intermediate artifacts).
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.Xs_train: list[np.ndarray] = []
        self.Xs_test: list[np.ndarray] = []
        self.ys_train: list[np.ndarray] = []
        self.ys_test: list[np.ndarray] = []

        # Intermediate attributes.
        self.anchor: np.ndarray = np.array([])
        self.anchor_test: np.ndarray = np.array([])
        self.anchor_y: np.ndarray = np.array([])
        self.anchor_y_test: np.ndarray = np.array([])
        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []
        self.anchors_inter: list[np.ndarray] = []
        self.anchors_test_inter: list[np.ndarray] = []
        self.L_within: np.ndarray | None = None
        self.L_between: np.ndarray | None = None
        self.graph_adjacency: np.ndarray | None = None
        self.graph_L_within: np.ndarray | None = None
        self.graph_L_between: np.ndarray | None = None

        # Integrated attributes.
        self.Xs_train_integ: list[np.ndarray] = []
        self.Xs_test_integ: list[np.ndarray] = []
        self.anchors_integ: list[np.ndarray] = []
        self.anchors_test_integ: list[np.ndarray] = []
        self.ys_train_integ: list[np.ndarray] = []
        self.ys_test_integ: list[np.ndarray] = []
        self.Z_integ: np.ndarray | None = None

        self.intermediate_artifacts: IntermediateArtifacts | None = None
        self.artifacts: IntegratedArtifacts | None = None

    # ------------------------------------------------------------------ #
    def run(self, intermediate_artifacts: IntermediateArtifacts) -> IntegratedArtifacts:
        self.intermediate_artifacts = intermediate_artifacts
        self._sync_from_intermediate(intermediate_artifacts)

        artifacts = self._build_integrated_artifacts()
        metrics = None
        if getattr(self.config, "evaluate_integrate_metrics", False):
            metrics = integrate_metrics(self)
        should_eval_lni = str(getattr(self.config, "G_type", "")).lower() == "nonlinear"
        if should_eval_lni:
            try:
                evaluate_nonlinearity_indices(self)
            except Exception as exc:  # pragma: no cover - diagnostics only
                if self.logger:
                    self.logger.warning("evaluate_nonlinearity_indices failed: %s", exc)

        if metrics is not None:
            artifacts = replace(artifacts, metrics=metrics)
        self.artifacts = artifacts
        self._sync_from_integrated(artifacts)
        return artifacts

    # ------------------------------------------------------------------ #
    def _build_integrated_artifacts(self) -> IntegratedArtifacts:
        if self.logger:
            self.logger.info("******************** Building G (integrated) ********************")
        projs, extras = self._build_projectors()
        if not projs:
            raise RuntimeError("No integration projectors were built.")

        self.Xs_train_integ = []
        self.Xs_test_integ = []
        self.anchors_integ = []
        self.anchors_test_integ = []

        count = min(len(projs), len(self.Xs_train_inter), len(self.Xs_test_inter), len(self.anchors_inter), len(self.anchors_test_inter))
        index_iter = tqdm(range(count), desc="Integrating institutions", unit="inst") if count > 1 else range(count)

        for idx in index_iter:
            proj = projs[idx]
            X_tr = self.Xs_train_inter[idx]
            X_te = self.Xs_test_inter[idx]
            anc_tr = self.anchors_inter[idx]
            anc_te = self.anchors_test_inter[idx]

            self.Xs_train_integ.append(proj(X_tr))
            self.Xs_test_integ.append(proj(X_te))
            self.anchors_integ.append(proj(anc_tr))
            self.anchors_test_integ.append(proj(anc_te))

        self.ys_train_integ = [np.asarray(y) for y in self.ys_train]
        self.ys_test_integ = [np.asarray(y) for y in self.ys_test]

        extras = extras or {}
        self.Z_integ = extras.get("Z_integ")

        return IntegratedArtifacts(
            intermediate=self.intermediate_artifacts,
            Xs_train_integ=list(self.Xs_train_integ),
            Xs_test_integ=list(self.Xs_test_integ),
            anchors_integ=list(self.anchors_integ),
            anchors_test_integ=list(self.anchors_test_integ),
            ys_train_integ=list(self.ys_train_integ),
            ys_test_integ=list(self.ys_test_integ),
            Z_integ=self.Z_integ,
        )

    def _build_projectors(self) -> Tuple[List, Dict[str, object]]:
        g_type_raw = getattr(self.config, "G_type", "Imakura")
        g_type_key = str(g_type_raw).lower()
        runner = _INTEGRATION_RUNNERS.get(g_type_key)
        if runner is None:
            raise ValueError(f"Unknown G_type: {g_type_raw}")
        return runner(self)

    def _sync_from_intermediate(self, artifacts: IntermediateArtifacts) -> None:
        dataset = artifacts.dataset
        self.train_df = dataset.train_df.copy()
        self.test_df = dataset.test_df.copy()
        self.Xs_train = list(dataset.Xs_train)
        self.Xs_test = list(dataset.Xs_test)
        self.ys_train = list(dataset.ys_train)
        self.ys_test = list(dataset.ys_test)

        self.anchor = artifacts.anchor
        self.anchor_test = artifacts.anchor_test
        self.anchor_y = artifacts.anchor_y
        self.anchor_y_test = artifacts.anchor_y_test
        self.Xs_train_inter = list(artifacts.Xs_train_inter)
        self.Xs_test_inter = list(artifacts.Xs_test_inter)
        self.anchors_inter = list(artifacts.anchors_inter)
        self.anchors_test_inter = list(artifacts.anchors_test_inter)
        self.L_within = artifacts.L_within
        self.L_between = artifacts.L_between
        self.graph_adjacency = artifacts.graph_adjacency
        self.graph_L_within = artifacts.graph_L_within
        self.graph_L_between = artifacts.graph_L_between

    def _sync_from_integrated(self, artifacts: IntegratedArtifacts) -> None:
        self.Xs_train_integ = list(artifacts.Xs_train_integ)
        self.Xs_test_integ = list(artifacts.Xs_test_integ)
        self.anchors_integ = list(artifacts.anchors_integ)
        self.anchors_test_integ = list(artifacts.anchors_test_integ)
        self.ys_train_integ = list(artifacts.ys_train_integ)
        self.ys_test_integ = list(artifacts.ys_test_integ)
        self.Z_integ = artifacts.Z_integ

# ---------------------------------------------------------------------- #
# Integration runners
# ---------------------------------------------------------------------- #
def _run_imakura_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ, g_abs_sum = build_imakura_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    analysis.config.g_abs_sum = g_abs_sum
    extras = {"Z_integ": Z_integ}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_targetvec_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ = build_targetvec_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    extras = {"Z_integ": Z_integ}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_gep_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    lambda_raw = getattr(analysis.config, "lambda_gen_eigen", 0.0)
    try:
        lambda_gen = float(lambda_raw)
    except (TypeError, ValueError):
        lambda_gen = 0.0
    projs, metrics = build_gep_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        lambda_gen=lambda_gen,
        orth_ver=bool(getattr(analysis.config, "orth_ver", False)),
    )
    for key, value in metrics.items():
        setattr(analysis.config, key, value)
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_gep2_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    lambda_raw = getattr(analysis.config, "lambda_gen_eigen", 0.0)
    try:
        lambda_gen = float(lambda_raw)
    except (TypeError, ValueError):
        lambda_gen = 0.0
    projs, metrics = build_gep2_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        lambda_gen=lambda_gen,
        orth_ver=bool(getattr(analysis.config, "orth_ver", False)),
    )
    for key, value in metrics.items():
        setattr(analysis.config, key, value)
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_odc_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ = build_odc_projectors(anchors_inter=analysis.anchors_inter)
    extras = {"Z_integ": Z_integ}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    lw_alpha = float(getattr(analysis.config, "lw_alpha", 0.0) or 0.0)
    projs, Z_integ, eigvals, gammas = build_nonlinear_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        K_normalization=bool(getattr(analysis.config, "K_normalization", False)),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        lw_alpha=lw_alpha,
        L_within=analysis.L_within,
        L_between=analysis.L_between,
    )
    gamma_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gamma_mean
    print(gammas)
    print("gammas")
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_graph_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    if analysis.L_within is None or analysis.L_between is None:
        raise ValueError("graph_nonlinear requires anchor Laplacians. Enable anchor_laplacian_k or lw_alpha > 0.")
    projs, Z_integ, eigvals, gammas = build_graph_nonlinear_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        graph_mu_align=float(getattr(analysis.config, "graph_mu_align", 1.0) or 1.0),
        constraint_eps=float(getattr(analysis.config, "graph_stability_eps", 1e-6) or 1e-6),
        L_within=analysis.L_within,
        L_between=analysis.L_between,
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_kernel_gep_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, metrics = build_kernel_gep_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
    )
    gammas = metrics.get("gammas", [])
    analysis.config.gamma_krr_means = float(np.mean(gammas)) if gammas else None
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_kernel_graph_gep_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    if analysis.graph_L_within is None or analysis.graph_L_between is None:
        raise ValueError("kernel_graph_gep requires graph Laplacians. Ensure graph_knn_k is configured.")
    projs, metrics = build_kernel_graph_gep_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        L_within_data=analysis.graph_L_within,
        L_between_data=analysis.graph_L_between,
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        mu_align=float(getattr(analysis.config, "graph_mu_align", 1.0) or 1.0),
        lambda_rkhs=float(getattr(analysis.config, "graph_lambda_rkhs", 1e-2) or 1e-2),
        stability_eps=float(getattr(analysis.config, "graph_stability_eps", 1e-6) or 1e-6),
    )
    gammas = metrics.get("gammas", [])
    analysis.config.gamma_krr_means = float(np.mean(gammas)) if gammas else None
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _dim_integrate(config: Config) -> int:
    dim = getattr(config, "dim_integrate", None)
    if dim is None:
        dim = getattr(config, "dim_intermediate", None)
    if dim is None:
        raise ValueError("dim_integrate is not set in the config.")
    return int(dim)


_INTEGRATION_RUNNERS: Dict[str, IntegrationRunner] = {
    "imakura": _run_imakura_integration,
    "targetvec": _run_targetvec_integration,
    "gep": _run_gep_integration,
    "gep_2": _run_gep2_integration,
    "gep2": _run_gep2_integration,
    "odc": _run_odc_integration,
    "nonlinear": _run_nonlinear_integration,
    "graph_nonlinear": _run_graph_nonlinear_integration,
    "kernel_gep": _run_kernel_gep_integration,
    "kernel_graph_gep": _run_kernel_graph_gep_integration,
}


IntegratedRepresentationBuilder = IntegratedExpressionBuilder  # backwards compatibility

__all__ = ["IntegratedExpressionBuilder", "IntegratedRepresentationBuilder"]
