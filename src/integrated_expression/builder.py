from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm

from config.config import Config
from src.common import ArtifactStore, IntegratedArtifacts, IntermediateArtifacts

from .integrate_metrics import evaluate_nonlinearity_indices, integrate_metrics
from .kernel_target_optimization import build_nonlinear_projectors_faster
from .runners import (
    _effective_rank,
    _zerosum_helmert_basis,
    build_gep2_projectors,
    build_faster_gep_projectors,
    build_gep_projectors,
    build_graph_nonlinear_projectors,
    build_graph_nonlinear_X_projectors,
    build_graph_nonlinear_X_projectors_maximize,
    build_graph_nonlinear_X_projectors_minimize,
    build_imakura_projectors,
    build_imakura_new_projectors,
    build_linear_nonridge_projectors,
    build_kernel_gep_projectors,
    build_kernel_graph_gep_projectors,
    build_nonlinear_projectors,
    build_nonlinear_projectors_maximize,
    build_nonlinear_imakura_Z_projectors,
    build_gep_singular_projectors,
    build_gep_singular_2_projectors,
    build_laplacian_nonlinear_projectors,
    build_laplacian_nonlinear_new_projectors,
    build_multi_cca_projectors,
    build_nonlinear_mlp_projectors,
    build_odc_projectors,
    build_laplacian_targetvec_projectors,
    build_targetvec_projectors,
    build_targetvec_singular_projectors,
    build_targetvec_new_projectors,
    build_gep_new_projectors,
    build_nonlinear_new_projectors,
)
from src.intermediate_expression.anchor_utils import build_laplacians_from_anchor_labels

logger = TypeVar("logger")
IntegrationRunner = Callable[["IntegratedExpressionBuilder"], Tuple[List, Dict[str, object]]]


class IntegratedExpressionBuilder:
    """
    Consumes intermediate artifacts and produces integrated (G-stage) representations.
    """

    def __init__(self, *, config: Config, logger: logger, store: Optional[ArtifactStore] = None) -> None:
        self.config = config
        self.logger = logger
        self.store = store or ArtifactStore(logger=logger)

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

        if getattr(self.config, "evaluate_sub_metrics", False):
            gtype_key = str(getattr(self.config, "G_type", "")).lower()
            lni_enabled_types = {
                "nonlinear",
                "nonlinear_new",
                "nonlinear_maximize",
                "graph_nonlinear",
                "graph_nonlinear_minimize",
                "graph_nonlinear_maximize",
                "graph_nonlinear_x",
                "graph_nonlinear_x_minimize",
                "graph_nonlinear_x_maximize",
                "kernel_gep",
                "kernel_graph_gep",
                "kernel_graph_gep_minimize",
                "kernel_graph_gep_maximize",
                "laplacian_nonlinear",
                "laplacian_nonlinear_new",
            }
            should_eval_lni = gtype_key in lni_enabled_types
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

        # Persist integrated artifacts for downstream analysis/visualization
        # only when explicitly requested.
        if bool(getattr(self.config, "preserve_integrated_data", False)):
            try:
                self.store.save("integrate", getattr(self.config, "integrated_name", None), artifacts)
            except Exception as exc:  # pragma: no cover - defensive logging only
                if self.logger:
                    self.logger.warning("Failed to save integrated artifacts: %s", exc)

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
        # ここで Z_integ のランクを見る
        if self.Z_integ is not None:
            rank = np.linalg.matrix_rank(self.Z_integ)
            print(f"[Z_integ] shape={self.Z_integ.shape}, rank={rank}")
            # 必要なら config にも入れておく
            setattr(self.config, "Z_integ_rank", float(rank))
        # Build aggregated / convenience representations for analysis.
        X_integ = np.vstack(self.Xs_train_integ) if self.Xs_train_integ else None
        # Take a representative integrated anchor set (first non-empty) and its labels.
        anchor_integ = None
        for arr in self.anchors_integ:
            if arr is not None and arr.size > 0:
                anchor_integ = arr
                break
        anchor_integ_y = None
        if anchor_integ is not None and getattr(self, "anchor_y", None) is not None:
            if self.anchor_y.size > 0:
                anchor_integ_y = np.asarray(self.anchor_y)

        return IntegratedArtifacts(
            intermediate=self.intermediate_artifacts,
            Xs_train_integ=list(self.Xs_train_integ),
            Xs_test_integ=list(self.Xs_test_integ),
            anchors_integ=list(self.anchors_integ),
            anchors_test_integ=list(self.anchors_test_integ),
            ys_train_integ=list(self.ys_train_integ),
            ys_test_integ=list(self.ys_test_integ),
            Z_integ=self.Z_integ,
            X_integ=X_integ,
            anchor_integ=anchor_integ,
            anchor_integ_y=anchor_integ_y,
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

    def _build_anchor_laplacians_for_integrated(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Build label-aware anchor Laplacians (L_within, L_between) at integration stage.
        Uses the same construction as intermediate_expression.builder.
        """
        anchor = self.anchor
        anchor_y = self.anchor_y
        if anchor.size == 0 or anchor_y.size == 0:
            return None, None

        assign_k = int(getattr(self.config, "anchor_assign_k", 10) or 10)
        # Use graph_knn_k as the neighborhood size for anchor label-aware Laplacians
        # (paper-style within/penalty graphs).
        graph_k_cfg = getattr(self.config, "graph_knn_k", None)
        try:
            anchor_lap_k = int(graph_k_cfg) if graph_k_cfg is not None else int(assign_k)
        except (TypeError, ValueError):
            anchor_lap_k = int(assign_k)
        gamma = getattr(self.config, "laplacian_gamma", None)

        L_within, L_between = build_laplacians_from_anchor_labels(
            anchor=anchor,
            anchor_y=anchor_y,
            gamma=gamma,
            k_neighbors=anchor_lap_k,
            metric="euclidean",
            logger=self.logger,
        )
        return L_within, L_between

# ---------------------------------------------------------------------- #
# Integration runners


def _apply_zerosum_helmert_to_anchors(
    anchors_inter: list[np.ndarray],
    zerosum: bool,
) -> list[np.ndarray]:
    """
    Optionally project anchor intermediate representations onto the zero-sum
    subspace using a Helmert basis.

    When `zerosum` is True and there are r >= 2 anchor samples shared across
    institutions, each anchor matrix A_k (shape: r x d_k) is transformed to
    B^T A_k where B is the r x (r-1) Helmert basis. This yields (r-1) x d_k
    anchors living in the zero-sum subspace along the anchor-sample axis.
    """
    if not zerosum:
        return anchors_inter
    if not anchors_inter:
        return []

    r = anchors_inter[0].shape[0]
    if r <= 1:
        raise ValueError("zerosum Helmert transform requires at least 2 anchor samples.")

    B = _zerosum_helmert_basis(r)
    BT = B.T

    transformed: list[np.ndarray] = []
    for A in anchors_inter:
        if A.shape[0] != r:
            raise ValueError(
                "zerosum Helmert transform requires all anchor matrices to share the same "
                "number of rows (anchor samples)."
            )
        transformed.append(BT @ A)
    return transformed
# ---------------------------------------------------------------------- #
def _run_imakura_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, Z_integ, g_abs_sum = build_imakura_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    analysis.config.g_abs_sum = g_abs_sum
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="imakura"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_imakura_new_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    truncated = bool(getattr(analysis.config, "truncated", False))
    projs, Z_integ, g_abs_sum = build_imakura_new_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        truncated=truncated,
    )
    analysis.config.g_abs_sum = g_abs_sum
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="imakura_new"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_targetvec_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    projs, Z_integ = build_targetvec_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        zerosum=zerosum,
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="targetvec"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_laplacian_targetvec_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 0.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 0.0

    graph_k_cfg = getattr(analysis.config, "graph_knn_k", None)
    try:
        graph_k = int(graph_k_cfg) if graph_k_cfg is not None else 0
    except (TypeError, ValueError):
        graph_k = 0
    if graph_k <= 0:
        graph_k = 10

    projs, Z_integ, eigvals = build_laplacian_targetvec_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        graph_mu_align=graph_mu_align,
        laplacian_k=graph_k,
        zerosum=zerosum,
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ, "eigvals": eigvals}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="laplacian_targetvec"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_targetvec_new_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    truncated = bool(getattr(analysis.config, "truncated", False))
    projs, Z_integ = build_targetvec_new_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        truncated=truncated,
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="targetvec_new"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_targetvec_singular_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, Z_integ, eigvals = build_targetvec_singular_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        zerosum=zerosum,
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ, "eigvals": eigvals}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="targetvec_singular"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_gep_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    lambda_raw = getattr(analysis.config, "lambda_gen_eigen", 0.0)
    try:
        lambda_gen = float(lambda_raw)
    except (TypeError, ValueError):
        lambda_gen = 0.0
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, metrics = build_gep_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        lambda_gen=lambda_gen,
        orth_ver=bool(getattr(analysis.config, "orth_ver", False)),
    )

    for key, value in metrics.items():
        setattr(analysis.config, key, value)
    extras: Dict[str, object] = dict(metrics)
    extras.setdefault("Z_integ", None)
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="gep"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_gep_new_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    """
    Integration runner for pairwise (QR+SVD-based) GEP_new.
    """
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    truncated = bool(getattr(analysis.config, "truncated", False))
    projs, Z_integ = build_gep_new_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        truncated=truncated,
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="gep_new"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_gep_singular_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    """
    Integration runner for rank-deficient tolerant QR+SVD (gep_singular).
    """
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, Z_integ, eigvals = build_gep_singular_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ, "eigvals": eigvals}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="gep_singular"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_gep_singular_2_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    """
    Integration runner for rank-deficient tolerant QR+SVD (gep_singular_2).
    """
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, Z_integ, eigvals = build_gep_singular_2_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ, "eigvals": eigvals}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="gep_singular_2"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_multi_cca_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, metrics = build_multi_cca_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        stability_eps=float(getattr(analysis.config, "multi_cca_stability_eps", 1e-6) or 1e-6),
    )
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
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, metrics = build_gep2_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        lambda_gen=lambda_gen,
        orth_ver=bool(getattr(analysis.config, "orth_ver", False)),
    )
    for key, value in metrics.items():
        setattr(analysis.config, key, value)
    extras: Dict[str, object] = dict(metrics)
    extras.setdefault("Z_integ", None)
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="gep2"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_faster_gep_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    """
    Integration runner for QR+SVD based faster_gep.
    """
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, metrics = build_faster_gep_projectors(
        anchors_inter=anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    extras: Dict[str, object] = dict(metrics)
    extras.setdefault("Z_integ", None)
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="faster_gep"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_odc_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    zerosum = bool(getattr(analysis.config, "zerosum", False))
    anchors_inter = _apply_zerosum_helmert_to_anchors(analysis.anchors_inter, zerosum)
    projs, Z_integ = build_odc_projectors(anchors_inter=anchors_inter)
    extras: Dict[str, object] = {"Z_integ": Z_integ}
    projs, extras = _apply_random_linear_post_transform(
        analysis.config, projs, extras, method_tag="odc"
    )
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 0.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 0.0
    if graph_mu_align == 0.0:
        L_within, L_between = None, None
    else:
        L_within, L_between = analysis._build_anchor_laplacians_for_integrated()
    projs, Z_integ, eigvals, gammas = build_nonlinear_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        K_normalization=bool(getattr(analysis.config, "K_normalization", False)),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        graph_mu_align=graph_mu_align,
        L_within=L_within,
        L_between=L_between,
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
    )
    gamma_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gamma_mean
    print(gammas)
    print("gammas")
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_new_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    # zerosum constraint matrix:
    # - "identity": Z^T Z = I  (common in older code paths / L_b = I)
    # - "l_between": Z^T L_between Z = I  (paper-style generalized constraint)
    zerosum_constraint = str(getattr(analysis.config, "zerosum_constraint", "identity") or "identity").lower()
    L_within, L_between = analysis._build_anchor_laplacians_for_integrated()
    constraint = L_between if zerosum_constraint in {"l_between", "between", "lb"} else None
    eps = float(getattr(analysis.config, "graph_stability_eps", 1e-9) or 1e-9)
    projs, Z_integ, eigvals, gammas = build_nonlinear_new_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        K_normalization=bool(getattr(analysis.config, "K_normalization", False)),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
        constraint_matrix=constraint,
        constraint_eps=eps,
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_imakura_Z_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ, eigvals, gammas = build_nonlinear_imakura_Z_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        K_normalization=bool(getattr(analysis.config, "K_normalization", False)),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_mlp_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    hidden_cfg = getattr(analysis.config, "nonlinear_mlp_hidden_dims", None)
    if hidden_cfg is None:
        hidden_dims = [500, 200]
    elif isinstance(hidden_cfg, (list, tuple)):
        hidden_dims = [int(v) for v in hidden_cfg]
    else:
        hidden_dims = [int(v.strip()) for v in str(hidden_cfg).split(",") if v.strip()]

    projs, Z_integ, eigvals, train_losses = build_nonlinear_mlp_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        hidden_dims=hidden_dims,
        mlp_lambda=float(
            getattr(
                analysis.config,
                "mlp_lambda",
                getattr(analysis.config, "nl_lambda", 1e-3),
            )
            or 0.0
        ),
        epochs=int(getattr(analysis.config, "nonlinear_mlp_epochs", 500) or 500),
        lr=float(getattr(analysis.config, "nonlinear_mlp_lr", 1e-3) or 1e-3),
        batch_size=getattr(analysis.config, "nonlinear_mlp_batch_size", None),
        seed=int(getattr(analysis.config, "seed", 0) or 0),
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
    )
    analysis.config.nonlinear_mlp_hidden_dims = hidden_dims
    analysis.config.nonlinear_mlp_final_losses = train_losses
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "train_losses": train_losses}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_graph_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 1.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 1.0
    L_within, L_between = analysis._build_anchor_laplacians_for_integrated()
    if L_within is None or L_between is None:
        raise ValueError("graph_nonlinear requires anchor Laplacians built from anchor labels.")
    projs, Z_integ, eigvals, gammas = build_graph_nonlinear_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        graph_mu_align=graph_mu_align,
        constraint_eps=float(getattr(analysis.config, "graph_stability_eps", 1e-6) or 1e-6),
        L_within=L_within,
        L_between=L_between,
        g_type=getattr(analysis.config, "G_type", None),
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
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
    # Require graph Laplacians (from intermediate data) for kernel_graph_gep
    if analysis.graph_L_within is None or analysis.graph_L_between is None:
        raise ValueError(
            "kernel_graph_gep requires graph Laplacians built from intermediate data. "
            "Set config.graph_knn_k > 0 to enable build_laplacians_from_intermediate_data and retry."
        )
    L_within_use = analysis.graph_L_within
    L_between_use = analysis.graph_L_between
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 1.0)
    mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 1.0
    projs, metrics = build_kernel_graph_gep_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        L_within_data=L_within_use,
        L_between_data=L_between_use,
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        mu_align=mu_align,
        lambda_rkhs=float(getattr(analysis.config, "graph_lambda_rkhs", 1e-2) or 1e-2),
        stability_eps=float(getattr(analysis.config, "graph_stability_eps", 1e-6) or 1e-6),
        g_type=getattr(analysis.config, "G_type", None),
    )
    gammas = metrics.get("gammas", [])
    analysis.config.gamma_krr_means = float(np.mean(gammas)) if gammas else None
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_nonlinear_max_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    lw_alpha = float(getattr(analysis.config, "lw_alpha", 0.0) or 0.0)
    projs, Z_integ, eigvals, gammas = build_nonlinear_projectors_maximize(
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
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_graph_nonlinear_X_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    # Require graph Laplacians (from intermediate data) for graph_nonlinear_X
    if analysis.graph_L_within is None or analysis.graph_L_between is None:
        raise ValueError(
            "graph_nonlinear_X requires graph Laplacians built from intermediate data. "
            "Set config.graph_knn_k > 0 to enable build_laplacians_from_intermediate_data and retry."
        )
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 1.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 1.0
    projs, Z_integ, eigvals, gammas = build_graph_nonlinear_X_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        graph_mu_align=graph_mu_align,
        constraint_eps=float(getattr(analysis.config, "graph_stability_eps", 1e-6) or 1e-6),
        graph_L_within=analysis.graph_L_within,
        graph_L_between=analysis.graph_L_between,
        g_type=getattr(analysis.config, "G_type", None),
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_laplacian_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    regularization = getattr(analysis.config, "regularization", "graph")
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 1.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 1.0
    # Use graph_knn_k for unlabeled Laplacian k
    graph_k_cfg = getattr(analysis.config, "graph_knn_k", None)
    try:
        graph_k = int(graph_k_cfg) if graph_k_cfg is not None else 0
    except (TypeError, ValueError):
        graph_k = 0
    if graph_k <= 0:
        graph_k = 10
    projs, Z_integ, eigvals, gammas = build_laplacian_nonlinear_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        anchor=analysis.anchor,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        graph_mu_align=graph_mu_align,
        laplacian_k=graph_k,
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
        regularization=regularization,
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean

    # Effective rank of the true kernel matrices (per institution).
    kernel_type_key = str(getattr(analysis.config, "kernel_type", "rbf") or "rbf").lower()
    eff_ranks: list[float] = []
    for k, anchor_inter_k in enumerate(analysis.anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            gamma_k = gammas[k] if k < len(gammas) else 1.0
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gamma_k)
        eff_ranks.append(_effective_rank(K))
    eff_rank_mean = float(np.mean(eff_ranks)) if eff_ranks else 0.0
    analysis.config.kernel_effective_rank_mean = eff_rank_mean

    extras = {
        "Z_integ": Z_integ,
        "eigvals": eigvals,
        "gammas": gammas,
        "kernel_effective_ranks": eff_ranks,
    }
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_laplacian_nonlinear_new_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    regularization = getattr(analysis.config, "regularization", "graph")
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 1.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 1.0
    graph_k_cfg = getattr(analysis.config, "graph_knn_k", None)
    try:
        graph_k = int(graph_k_cfg) if graph_k_cfg is not None else 0
    except (TypeError, ValueError):
        graph_k = 0
    if graph_k <= 0:
        graph_k = 10

    reg_key = str(regularization or "graph").lower()
    # "graph": label-agnostic Laplacian (no target labels)
    # "target-graph": label-aware within/penalty graphs (uses anchor_y)
    # "penal-target-graph": label-aware additive/subtractive graph, A += mu * (Lw - Lb)
    if reg_key in {"target-graph", "target_graph", "penal-target-graph", "penal_target_graph"}:
        L_within, L_between = analysis._build_anchor_laplacians_for_integrated()
    else:
        L_within, L_between = None, None

    # For stability in generalized eigenproblems (target-graph only).
    eps = float(getattr(analysis.config, "graph_stability_eps", 1e-9) or 1e-9)

    # Constraint selection for target-graph:
    # - "between" (default): use L_between as mass matrix (paper-style)
    # - "identity": use I as mass matrix (ignore between-class constraint)
    # Optional:
    # - target_graph_between_weight in [0, 1]:
    #     B = (1-w) I + w * scaled(L_between)
    #   to softly control the amount of between-class separation.
    constraint_matrix = None
    if reg_key in {"target-graph", "target_graph"}:
        constraint_mode = str(getattr(analysis.config, "target_graph_constraint", "between") or "between").lower()
        r = int(analysis.anchor.shape[0]) if analysis.anchor is not None else 0
        if constraint_mode in {"identity", "i", "eye"}:
            if r > 0:
                constraint_matrix = np.eye(r, dtype=float)
        elif constraint_mode in {"between", "b"}:
            w_raw = getattr(analysis.config, "target_graph_between_weight", 1.0)
            try:
                w = float(w_raw)
            except (TypeError, ValueError):
                w = 1.0
            w = max(0.0, min(1.0, w))
            if w < 1.0 and r > 0:
                if L_between is None:
                    raise ValueError("target_graph_between_weight requires L_between.")
                Lb = np.asarray(L_between, dtype=float)
                if Lb.shape != (r, r):
                    raise ValueError(f"L_between must be shape {(r, r)} but got {Lb.shape}")
                Lb = (Lb + Lb.T) * 0.5
                tr_Lb = float(np.trace(Lb))
                scale_b = float(r) / max(tr_Lb, 1e-12) if tr_Lb > 0 else 1.0
                constraint_matrix = (1.0 - w) * np.eye(r, dtype=float) + w * (scale_b * Lb)

    projs, Z_integ, eigvals, gammas = build_laplacian_nonlinear_new_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        anchor=analysis.anchor,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        graph_mu_align=graph_mu_align,
        laplacian_k=graph_k,
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
        regularization=regularization,
        K_normalization=bool(getattr(analysis.config, "K_normalization", False)),
        constraint_matrix=constraint_matrix,
        constraint_eps=eps,
        L_within=L_within,
        L_between=L_between,
    )

    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonridge_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ, eigvals = build_linear_nonridge_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
    )
    extras = {"Z_integ": Z_integ, "eigvals": eigvals}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_nonridge_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    # Use the same unlabeled graph as Laplacian-based nonlinear integration,
    # but default graph_mu_align to 0 so that the graph term is disabled
    # unless explicitly requested.
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 0.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 0.0

    graph_k_cfg = getattr(analysis.config, "graph_knn_k", None)
    try:
        graph_k = int(graph_k_cfg) if graph_k_cfg is not None else 0
    except (TypeError, ValueError):
        graph_k = 0
    if graph_k <= 0:
        graph_k = 10

    from .runners import build_laplacian_nonlinear_nonridge_projectors

    projs, Z_integ, eigvals, gammas = build_laplacian_nonlinear_nonridge_projectors(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        anchor=analysis.anchor,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        graph_mu_align=graph_mu_align,
        laplacian_k=graph_k,
        zerosum=False,
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean

    # Effective rank of the true kernel matrices (per institution), for logging.
    kernel_type_key = str(getattr(analysis.config, "kernel_type", "rbf") or "rbf").lower()
    eff_ranks: list[float] = []
    for k, anchor_inter_k in enumerate(analysis.anchors_inter):
        if kernel_type_key == "linear":
            K = anchor_inter_k @ anchor_inter_k.T
        else:
            gamma_k = gammas[k] if k < len(gammas) else 1.0
            K = rbf_kernel(anchor_inter_k, anchor_inter_k, gamma=gamma_k)
        eff_ranks.append(_effective_rank(K))
    eff_rank_mean = float(np.mean(eff_ranks)) if eff_ranks else 0.0
    analysis.config.kernel_effective_rank_mean = eff_rank_mean

    extras = {
        "Z_integ": Z_integ,
        "eigvals": eigvals,
        "gammas": gammas,
        "kernel_effective_ranks": eff_ranks,
    }
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_faster_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    graph_mu_align_cfg = getattr(analysis.config, "graph_mu_align", 0.0)
    graph_mu_align = float(graph_mu_align_cfg) if graph_mu_align_cfg is not None else 0.0

    graph_k_cfg = getattr(analysis.config, "graph_knn_k", None)
    try:
        graph_k = int(graph_k_cfg) if graph_k_cfg is not None else 0
    except (TypeError, ValueError):
        graph_k = 0
    if graph_k <= 0:
        graph_k = 10

    rank_nystrom_cfg = getattr(analysis.config, "rank_nystrom", None)
    print("rank_nystrom:", rank_nystrom_cfg)
    try:
        rank_nystrom = int(rank_nystrom_cfg) if rank_nystrom_cfg is not None else 200
    except (TypeError, ValueError):
        rank_nystrom = 200

    lobpcg_tol_cfg = getattr(analysis.config, "lobpcg_tol", 1e-5)
    try:
        lobpcg_tol = float(lobpcg_tol_cfg)
    except (TypeError, ValueError):
        lobpcg_tol = 1e-5

    lobpcg_maxiter_cfg = getattr(analysis.config, "lobpcg_maxiter", 200)
    try:
        lobpcg_maxiter = int(lobpcg_maxiter_cfg)
    except (TypeError, ValueError):
        lobpcg_maxiter = 200

    use_faiss_graph = bool(getattr(analysis.config, "use_faiss_graph", False))
    fast_use_nystrom = bool(getattr(analysis.config, "fast_use_nystrom", True))
    fast_use_lobpcg = bool(getattr(analysis.config, "fast_use_lobpcg", True))
    public_anchor_count = int(getattr(analysis.config, "public_anchor_count", 0) or 0)

    projs, Z_integ, eigvals, gammas = build_nonlinear_projectors_faster(
        anchors_inter=analysis.anchors_inter,
        Xs_train_inter=analysis.Xs_train_inter,
        anchor=analysis.anchor,
        dim_integrate=_dim_integrate(analysis.config),
        gamma_type=getattr(analysis.config, "gamma_type", "auto"),
        gamma_ratio_krr=getattr(analysis.config, "gamma_ratio_krr", 1.0),
        nl_lambda=getattr(analysis.config, "nl_lambda", 1e-2),
        kernel_type=str(getattr(analysis.config, "kernel_type", "rbf") or "rbf"),
        graph_mu_align=graph_mu_align,
        laplacian_k=graph_k,
        zerosum=bool(getattr(analysis.config, "zerosum", False)),
        rank_nystrom=rank_nystrom,
        lobpcg_tol=lobpcg_tol,
        lobpcg_maxiter=lobpcg_maxiter,
        use_faiss_graph=use_faiss_graph,
        use_nystrom=fast_use_nystrom,
        use_lobpcg=fast_use_lobpcg,
        public_anchor_count=public_anchor_count,
        random_state=getattr(analysis.config, "seed", None),
    )
    gammas_mean = float(np.mean(gammas)) if gammas else None
    analysis.config.gamma_krr_means = gammas_mean
    extras: Dict[str, object] = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _dim_integrate(config: Config) -> int:
    """
    Determine and cache the integration dimension.

    Preference order:
      1) config.dim_integrate (if set)
      2) config.dim_intermediate
    The resolved value is written back to config.dim_integrate so that
    downstream helpers can reliably infer the output dimension.
    """
    dim = getattr(config, "dim_integrate", None)
    if dim is None:
        dim = getattr(config, "dim_intermediate", None)
    if dim is None:
        raise ValueError("dim_integrate is not set in the config.")
    dim_int = int(dim)
    if getattr(config, "dim_integrate", None) is None:
        config.dim_integrate = dim_int
    return dim_int


def _apply_random_linear_post_transform(
    config: Config,
    projs: list,
    extras: Dict[str, object],
    *,
    method_tag: str,
) -> tuple[list, Dict[str, object]]:
    """
    Apply a shared random linear transform on the right to all integrated
    representations produced by linear projectors (X -> X @ G_i).

    - Generate an orthogonal matrix Q via QR with seed (seed + Q_seed).
    - Generate an invertible matrix R with seed (seed + R_seed).
    - Combined transform T = Q @ R is applied as:
        proj'_i(X) = proj_i(X) @ T
      and, if Z_integ is present, Z'_integ = Z_integ @ T.

    If seeds are not provided, or the output dimension cannot be inferred,
    this function is a no-op.
    """
    # Infer output dimension: prefer Z_integ, fallback to config.dim_integrate.
    Z = extras.get("Z_integ")
    if isinstance(Z, np.ndarray) and Z.ndim == 2 and Z.shape[1] > 0:
        dim_out = int(Z.shape[1])
    else:
        dim_cfg = getattr(config, "dim_integrate", None)
        dim_out = int(dim_cfg) if dim_cfg is not None else None

    if dim_out is None or dim_out <= 0:
        return projs, extras

    base_seed = getattr(config, "seed", None)
    base = int(base_seed) if base_seed is not None else 0

    q_offset = getattr(config, "Q_seed", None)
    r_offset = getattr(config, "R_seed", None)
    if q_offset is None and r_offset is None:
        # No dedicated seeds specified -> keep behavior unchanged.
        return projs, extras

    def _as_int_offset(val: object) -> int:
        """Safely convert possibly list/tuple/scalar to int offset."""
        if val is None:
            return 0
        if isinstance(val, (list, tuple)):
            # e.g., grid search may pass [1,2,...]; take the first
            return _as_int_offset(val[0] if val else 0)
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    q_seed = base + _as_int_offset(q_offset)
    r_seed = base + _as_int_offset(r_offset)

    rng_Q = np.random.default_rng(q_seed)
    rng_R = np.random.default_rng(r_seed)

    # Random orthogonal via QR
    A_Q = rng_Q.normal(size=(dim_out, dim_out))
    Q, R_qr = np.linalg.qr(A_Q)
    diag = np.sign(np.diag(R_qr))
    diag[diag == 0] = 1.0
    Q = Q * diag

    # Random well-conditioned invertible matrix
    max_trials = 50
    T_R = None
    for _ in range(max_trials):
        M = rng_R.normal(size=(dim_out, dim_out))
        try:
            cond = np.linalg.cond(M)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(cond) and cond < 1e12:
            T_R = M
            break
    if T_R is None:
        T_R = np.eye(dim_out)

    T = Q @ T_R

    def _wrap_proj(proj):
        def wrapped(X):
            return proj(X) @ T

        return wrapped

    new_projs = [_wrap_proj(p) for p in projs]

    if isinstance(Z, np.ndarray) and Z.ndim == 2 and Z.shape[1] == dim_out:
        extras = dict(extras)
        extras["Z_integ"] = Z @ T

    # Expose the transform for downstream inspection / debugging.
    setattr(config, f"random_transform_{method_tag}", T)

    return new_projs, extras


_INTEGRATION_RUNNERS: Dict[str, IntegrationRunner] = {
    "imakura": _run_imakura_integration,
    "imakura_new": _run_imakura_new_integration,
    "targetvec": _run_targetvec_integration,
    "laplacian_targetvec": _run_laplacian_targetvec_integration,
    "targetvec_singular": _run_targetvec_singular_integration,
    "targetvec_new": _run_targetvec_new_integration,
    "gep": _run_gep_integration,
    "gep_new": _run_gep_new_integration,
    "gep_2": _run_gep2_integration,
    "gep2": _run_gep2_integration,
    "faster_gep": _run_faster_gep_integration,
    "gep_singular": _run_gep_singular_integration,
    "gep_singular_2": _run_gep_singular_2_integration,
    "gep_singular2": _run_gep_singular_2_integration,
    "odc": _run_odc_integration,
    "nonridge": _run_nonridge_integration,
    "nonlinear": _run_nonlinear_integration,
    "nonlinear_new": _run_nonlinear_new_integration,
    "nonlinear_imakura_z": _run_nonlinear_imakura_Z_integration,
    "nonlinear_mlp": _run_nonlinear_mlp_integration,
    "nonlinear_nonridge": _run_nonlinear_nonridge_integration,
    "nonlinear_maximize": _run_nonlinear_max_integration,
    "nonlinear_faster": _run_nonlinear_faster_integration,
    "laplacian_nonlinear": _run_laplacian_nonlinear_integration,
    "laplacian_nonlinear_new": _run_laplacian_nonlinear_new_integration,
    "graph_nonlinear": _run_graph_nonlinear_integration,
    "graph_nonlinear_minimize": _run_graph_nonlinear_integration,
    "graph_nonlinear_maximize": _run_graph_nonlinear_integration,
    "kernel_gep": _run_kernel_gep_integration,
    "kernel_graph_gep": _run_kernel_graph_gep_integration,
    "kernel_graph_gep_minimize": _run_kernel_graph_gep_integration,
    "kernel_graph_gep_maximize": _run_kernel_graph_gep_integration,
    "graph_nonlinear_x": _run_graph_nonlinear_X_integration,
    "graph_nonlinear_x_minimize": _run_graph_nonlinear_X_integration,
    "graph_nonlinear_x_maximize": _run_graph_nonlinear_X_integration,
    "multi_cca": _run_multi_cca_integration,
}


IntegratedRepresentationBuilder = IntegratedExpressionBuilder  # backwards compatibility

__all__ = ["IntegratedExpressionBuilder", "IntegratedRepresentationBuilder"]
