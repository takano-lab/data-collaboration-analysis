from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from config.config import Config
from src.integration import (
    build_gep_projectors,
    build_imakura_projectors,
    build_nonlinear_projectors,
    build_odc_projectors,
    build_targetvec_projectors,
)
from src.institution_data_and_intermediate_representation.anchor_data import (
    assign_anchor_labels as _assign_anchor_labels,
    build_laplacians_from_anchor_labels as _build_laplacians_from_anchor_labels,
)
from src.integrated_representation.integrate_metrics import (
    evaluate_nonlinearity_indices as _evaluate_nonlinearity_indices,
    integrate_metrics as _integrate_metrics,
)

logger = TypeVar("logger")
IntegrationRunner = Callable[["IntegratedExpressionBuilder"], Tuple[list, Dict[str, object]]]


class IntegratedExpressionBuilder:
    """Consumes intermediate representations and builds integrated (G-stage) outputs."""

    def __init__(
        self,
        *,
        config: Config,
        logger: logger,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        Xs_train: Sequence[np.ndarray] | None = None,
        Xs_test: Sequence[np.ndarray] | None = None,
        ys_train: Sequence[np.ndarray] | None = None,
        ys_test: Sequence[np.ndarray] | None = None,
        Xs_train_inter: Sequence[np.ndarray] | None = None,
        Xs_test_inter: Sequence[np.ndarray] | None = None,
        anchors_inter: Sequence[np.ndarray] | None = None,
        anchors_test_inter: Sequence[np.ndarray] | None = None,
        anchor: np.ndarray | None = None,
        anchor_test: np.ndarray | None = None,
        anchor_y: np.ndarray | None = None,
        anchor_y_test: np.ndarray | None = None,
        L_within: np.ndarray | None = None,
        L_between: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.train_df = train_df.copy(deep=True) if isinstance(train_df, pd.DataFrame) else pd.DataFrame(train_df or [])
        self.test_df = test_df.copy(deep=True) if isinstance(test_df, pd.DataFrame) else pd.DataFrame(test_df or [])
        self.Xs_train: list[np.ndarray] = list(Xs_train or [])
        self.Xs_test: list[np.ndarray] = list(Xs_test or [])
        self.ys_train: list[np.ndarray] = [np.asarray(y) for y in (ys_train or [])]
        self.ys_test: list[np.ndarray] = [np.asarray(y) for y in (ys_test or [])]

        self.Xs_train_inter: list[np.ndarray] = list(Xs_train_inter or [])
        self.Xs_test_inter: list[np.ndarray] = list(Xs_test_inter or [])
        self.anchors_inter: list[np.ndarray] = list(anchors_inter or [])
        self.anchors_test_inter: list[np.ndarray] = list(anchors_test_inter or [])

        self.Xs_train_integ: list[np.ndarray] = []
        self.Xs_test_integ: list[np.ndarray] = []
        self.anchors_integ: list[np.ndarray] = []
        self.anchors_test_integ: list[np.ndarray] = []
        self.ys_train_integ: list[np.ndarray] = []
        self.ys_test_integ: list[np.ndarray] = []

        self.anchor = np.asarray(anchor) if anchor is not None else np.array([])
        self.anchor_test = np.asarray(anchor_test) if anchor_test is not None else np.array([])
        self.anchor_y = np.asarray(anchor_y) if anchor_y is not None else np.array([])
        self.anchor_y_test = np.asarray(anchor_y_test) if anchor_y_test is not None else np.array([])

        self.L_within = L_within
        self.L_between = L_between
        self.Z_integ: np.ndarray | None = None

    @classmethod
    def from_institution_builder(cls, builder) -> "IntegratedExpressionBuilder":
        return cls(
            config=builder.config,
            logger=builder.logger,
            train_df=builder.train_df,
            test_df=builder.test_df,
            Xs_train=builder.Xs_train,
            Xs_test=builder.Xs_test,
            ys_train=builder.ys_train,
            ys_test=builder.ys_test,
            Xs_train_inter=builder.Xs_train_inter,
            Xs_test_inter=builder.Xs_test_inter,
            anchors_inter=builder.anchors_inter,
            anchors_test_inter=builder.anchors_test_inter,
            anchor=builder.anchor,
            anchor_test=builder.anchor_test,
            anchor_y=getattr(builder, "anchor_y", None),
            anchor_y_test=getattr(builder, "anchor_y_test", None),
            L_within=getattr(builder, "L_within", None),
            L_between=getattr(builder, "L_between", None),
        )

    def run(self) -> None:
        self.make_integrate_expression()
        if getattr(self.config, "evaluate_integrate_metrics", False):
            self.integrate_metrics()
        try:
            self.evaluate_nonlinearity_indices()
        except Exception as exc:
            self._log(f"[WARN] evaluate_nonlinearity_indices failed: {exc}", level="warning")

    # ------------------------------------------------------------------ #
    # Integration + metrics
    # ------------------------------------------------------------------ #
    def make_integrate_expression(self) -> None:
        self._log("******************** Building G (integrated) ********************")
        g_type_raw = getattr(self.config, "G_type", "Imakura")
        g_type_key = str(g_type_raw).lower()
        runner = _INTEGRATION_RUNNERS.get(g_type_key)

        if runner is None:
            self._log(f"Unknown G_type: {g_type_raw}", level="warning")
            return

        try:
            projs, extras = runner(self)
        except Exception as exc:
            self._log(f"Failed to build integration projectors for {g_type_raw}: {exc}", level="error")
            raise

        if not projs:
            self._log(f"No projectors were returned for G_type={g_type_raw}", level="warning")
            return

        self.Xs_train_integ = []
        self.Xs_test_integ = []
        self.anchors_integ = []
        self.anchors_test_integ = []

        for proj, X_tr, X_te, anc_tr, anc_te in zip(
            projs, self.Xs_train_inter, self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter
        ):
            self.Xs_train_integ.append(proj(X_tr))
            self.Xs_test_integ.append(proj(X_te))
            self.anchors_integ.append(proj(anc_tr))
            self.anchors_test_integ.append(proj(anc_te))

        self.ys_train_integ = [np.asarray(y) for y in self.ys_train]
        self.ys_test_integ = [np.asarray(y) for y in self.ys_test]

        extras = extras or {}
        if "Z_integ" in extras:
            self.Z_integ = extras["Z_integ"]

    def integrate_metrics(self) -> dict:
        return _integrate_metrics(self)

    def evaluate_nonlinearity_indices(self) -> dict:
        return _evaluate_nonlinearity_indices(self)

    def assign_anchor_labels(self, k: int = 5) -> None:
        self.anchor_y, self.anchor_y_test = _assign_anchor_labels(
            anchor=self.anchor,
            anchor_test=self.anchor_test,
            Xs_train=self.Xs_train,
            ys_train=self.ys_train,
            k=k,
        )

    def build_laplacians_from_anchor_labels(self, gamma: float | None = None) -> None:
        L_within, L_between = _build_laplacians_from_anchor_labels(
            anchor=self.anchor,
            anchor_y=self.anchor_y,
            gamma=gamma,
            logger=self.logger,
        )
        self.L_within = L_within
        self.L_between = L_between

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger is None:
            return
        try:
            log_fn = getattr(self.logger, level, None)
            if callable(log_fn):
                log_fn(msg)
            else:
                self.logger.info(msg)
        except Exception:
            pass


def _dim_integrate(config: Config) -> int:
    dim = getattr(config, "dim_integrate", None)
    if dim is None:
        dim = getattr(config, "dim_intermediate", None)
    if dim is None:
        raise ValueError("dim_integrate が設定されていません。")
    return int(dim)


def _run_imakura_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ, g_abs_sum = build_imakura_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
    )
    extras: Dict[str, object] = {"Z_integ": Z_integ, "g_abs_sum": g_abs_sum}
    analysis.Z_integ = Z_integ
    analysis.config.g_abs_sum = g_abs_sum
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
    projs, metrics = build_gep_projectors(
        anchors_inter=analysis.anchors_inter,
        dim_integrate=_dim_integrate(analysis.config),
        lambda_gen=getattr(analysis.config, "lambda_gen_eigen", 0.0),
        orth_ver=bool(getattr(analysis.config, "orth_ver", False)),
    )
    for key in ["jreg_gep", "g_norm_val_gep", "sum_objective_function", "g_mean_var", "g_condition_number", "g_abs_sum"]:
        if key in metrics:
            setattr(analysis.config, key, metrics[key])
    extras = dict(metrics)
    extras.setdefault("Z_integ", None)
    analysis.Z_integ = extras.get("Z_integ")
    return projs, extras


def _run_odc_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    projs, Z_integ = build_odc_projectors(
        anchors_inter=analysis.anchors_inter,
    )
    extras = {"Z_integ": Z_integ}
    analysis.Z_integ = Z_integ
    return projs, extras


def _run_nonlinear_integration(analysis: "IntegratedExpressionBuilder") -> tuple[list, Dict[str, object]]:
    analysis._log("[integration] runner=nonlinear")
    lw_alpha = float(getattr(analysis.config, "lw_alpha", 0.0) or 0.0)
    if lw_alpha != 0.0:
        assign_k = int(getattr(analysis.config, "anchor_assign_k", 5) or 5)
        analysis.assign_anchor_labels(k=assign_k)
        analysis.build_laplacians_from_anchor_labels()

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
    extras = {"Z_integ": Z_integ, "eigvals": eigvals, "gammas": gammas}
    analysis.Z_integ = Z_integ
    return projs, extras


_INTEGRATION_RUNNERS: Dict[str, IntegrationRunner] = {
    "imakura": _run_imakura_integration,
    "targetvec": _run_targetvec_integration,
    "gep": _run_gep_integration,
    "odc": _run_odc_integration,
    "nonlinear": _run_nonlinear_integration,
}


IntegratedRepresentationBuilder = IntegratedExpressionBuilder  # backwards compatibility

__all__ = ["IntegratedExpressionBuilder", "IntegratedRepresentationBuilder"]
