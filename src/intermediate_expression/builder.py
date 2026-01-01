from __future__ import annotations

from dataclasses import replace
import copy
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config.config import Config
from src.common import ArtifactStore, DatasetArtifacts, IntermediateArtifacts
from src.dimensionality_reduction import (
    build_dimensionality_projector,
    build_shared_subspace_projectors,
)

from .anchor_utils import (
    assign_anchor_labels,
    build_laplacians_from_anchor_labels,
    build_laplacians_from_intermediate_data,
    produce_anchor,
    _valid_label_mask,
)

MAX_INTERMEDIATE_ARTIFACTS = 100

class IntermediateExpressionBuilder:
    """
    Builds anchors and intermediate representations from dataset artifacts.
    """

    def __init__(self, *, config: Config, logger, store: Optional[ArtifactStore] = None) -> None:
        self.config = config
        self.logger = logger
        self.store = store or ArtifactStore(logger=logger)

        self.dataset_artifacts: DatasetArtifacts | None = None
        self.artifacts: IntermediateArtifacts | None = None

        self.anchor: np.ndarray = np.array([])
        self.anchor_test: np.ndarray = np.array([])
        self.anchor_y: np.ndarray = np.array([])
        self.anchor_y_test: np.ndarray = np.array([])
        self.Xs_train_inter: List[np.ndarray] = []
        self.Xs_test_inter: List[np.ndarray] = []
        self.anchors_inter: List[np.ndarray] = []
        self.anchors_test_inter: List[np.ndarray] = []
        self.L_within: np.ndarray | None = None
        self.L_between: np.ndarray | None = None
        self.graph_adjacency: np.ndarray | None = None
        self.graph_L_within: np.ndarray | None = None
        self.graph_L_between: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def run(self, dataset_artifacts: DatasetArtifacts) -> IntermediateArtifacts:
        self.dataset_artifacts = dataset_artifacts
        if getattr(self.config, "load_intermediate_data", False):
            loaded = self._load_from_store()
            if loaded is not None:
                artifacts = self._maybe_normalize_artifacts(loaded)
                artifacts = self._maybe_attach_anchor_laplacians(artifacts)
                artifacts = self._maybe_attach_graph_laplacians(artifacts)
                self._sync_from_artifacts(artifacts)
                self.artifacts = artifacts
                if self.logger:
                    self.logger.info("Loaded intermediate artifacts from cache.")
                return artifacts

        raw_artifacts = self._build_intermediate(dataset_artifacts)

        if getattr(self.config, "load_intermediate_data", False):
            self.store.save("intermediate", getattr(self.config, "intermediate_name", None), raw_artifacts)
            self.store.prune("intermediate", keep=MAX_INTERMEDIATE_ARTIFACTS)

        artifacts = self._maybe_normalize_artifacts(raw_artifacts)
        artifacts = self._maybe_attach_anchor_laplacians(artifacts)
        artifacts = self._maybe_attach_graph_laplacians(artifacts)
        self._sync_from_artifacts(artifacts)
        self.artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------ #
    def _build_intermediate(self, dataset: DatasetArtifacts) -> IntermediateArtifacts:
        if self.logger:
            self.logger.info("Building intermediate artifacts (anchors + projectors).")
        if not dataset.Xs_train:
            raise RuntimeError("Dataset artifacts do not contain institutional splits.")

        num_features = dataset.Xs_train[0].shape[1]
        public_X = getattr(dataset, "public_anchor", None)
        public_y = getattr(dataset, "public_anchor_y", None)
        # Config.__getattr__ always returns None for missing keys, so treat None as "use it".
        raw_use_public_anchor = getattr(self.config, "use_public_anchor", None)
        use_public_anchor = True if raw_use_public_anchor is None else bool(raw_use_public_anchor)

        has_public_data = (
            public_X is not None
            and public_y is not None
            and getattr(public_X, "size", 0) > 0
            and getattr(public_y, "size", 0) > 0
        )
        include_public_in_anchor = use_public_anchor and has_public_data

        # Expose the number of public anchors (clipped by num_anchor_data) via config
        # so that downstream Nyström landmarks can prioritize them.
        public_count = 0
        if include_public_in_anchor:
            try:
                public_count = int(min(public_X.shape[0], int(self.config.num_anchor_data)))
            except Exception:
                public_count = int(public_X.shape[0])
        setattr(self.config, "public_anchor_count", max(0, int(public_count if include_public_in_anchor else 0)))

        # If SMOTE is requested but public anchors are absent, fall back to Gaussian anchors.
        anchor_cfg = self.config
        anchor_method = getattr(self.config, "anchor_method", "gaussian")
        anchor_method_lower = str(anchor_method).lower()
        if anchor_method_lower == "smote" and not has_public_data:
            if self.logger:
                self.logger.info("SMOTE requested but public anchors unavailable; using gaussian anchors instead.")
            anchor_cfg = copy.copy(self.config)
            anchor_cfg.anchor_method = "gaussian"

        if anchor_method_lower == "smote" and has_public_data:
            self.anchor, self.anchor_y = produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_features,
                seed=self.config.seed,
                config=anchor_cfg,
                train_df=dataset.train_df,
                Xs_train=dataset.Xs_train,
                Xs_test=dataset.Xs_test,
                ys_train=dataset.ys_train,
                ys_test=dataset.ys_test,
                smote_X=public_X,
                smote_y=public_y,
                include_public_anchor=include_public_in_anchor,
                return_labels=True,
            )
            self.anchor_test, self.anchor_y_test = produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_features,
                seed=self.config.seed + 1,
                config=anchor_cfg,
                train_df=dataset.train_df,
                Xs_train=dataset.Xs_train,
                Xs_test=dataset.Xs_test,
                ys_train=dataset.ys_train,
                ys_test=dataset.ys_test,
                smote_X=public_X,
                smote_y=public_y,
                include_public_anchor=include_public_in_anchor,
                return_labels=True,
            )
        else:
            self.anchor = produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_features,
                seed=self.config.seed,
                config=anchor_cfg,
                train_df=dataset.train_df,
                Xs_train=dataset.Xs_train,
                Xs_test=dataset.Xs_test,
                ys_train=dataset.ys_train,
                ys_test=dataset.ys_test,
            )
            self.anchor_test = produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_features,
                seed=self.config.seed + 1,
                config=anchor_cfg,
                train_df=dataset.train_df,
                Xs_train=dataset.Xs_train,
                Xs_test=dataset.Xs_test,
                ys_train=dataset.ys_train,
                ys_test=dataset.ys_test,
            )

        projectors = self._build_projectors(dataset)
        self.Xs_train_inter = []
        self.Xs_test_inter = []
        self.anchors_inter = []
        self.anchors_test_inter = []

        for projector, X_train, X_test in zip(projectors, dataset.Xs_train, dataset.Xs_test):
            X_train_reduced = projector(X_train)
            X_test_reduced = projector(X_test)
            anchor_reduced = projector(self.anchor)
            anchor_test_reduced = projector(self.anchor_test)
            self.Xs_train_inter.append(X_train_reduced)
            self.Xs_test_inter.append(X_test_reduced)
            self.anchors_inter.append(anchor_reduced)
            self.anchors_test_inter.append(anchor_test_reduced)

        # Assign anchor labels if not already determined (e.g., non-SMOTE methods)
        if self.anchor_y.size == 0 or self.anchor_y_test.size == 0:
            assign_k = int(getattr(self.config, "anchor_assign_k", 10) or 10)
            max_anchor_dist = float(getattr(self.config, "anchor_label_max_dist", 0.0) or 0.0)
            self.anchor_y, self.anchor_y_test = assign_anchor_labels(
                anchors_inter=self.anchors_inter,
                anchors_test_inter=self.anchors_test_inter,
                Xs_train_inter=self.Xs_train_inter,
                ys_train=dataset.ys_train,
                k=assign_k,
                max_neighbor_dist=max_anchor_dist,
            )

        # 全機関からラベルが付かなかったアンカー（無ラベル）は、そもそもアンカーとして除外する
        if self.anchor_y.size:
            valid_mask = _valid_label_mask(self.anchor_y)
            if not np.all(valid_mask):
                num_before = self.anchor.shape[0]
                num_after = int(np.count_nonzero(valid_mask))
                print(
                    f"[IntermediateExpressionBuilder] dropping {num_before - num_after} unlabeled anchors "
                    f"(kept {num_after})"
                )
                self.anchor = self.anchor[valid_mask]
                if self.anchor_test.size:
                    self.anchor_test = self.anchor_test[valid_mask]
                self.anchor_y = self.anchor_y[valid_mask]
                if self.anchor_y_test.size:
                    self.anchor_y_test = self.anchor_y_test[valid_mask]
                self.anchors_inter = [anc[valid_mask] for anc in self.anchors_inter]
                self.anchors_test_inter = [anc[valid_mask] for anc in self.anchors_test_inter]

        lw_alpha = float(getattr(self.config, "lw_alpha", 0.0) or 0.0)
        need_anchor_laplacian = (lw_alpha > 0.0) or self._needs_anchor_laplacian()
        if need_anchor_laplacian:
            gamma = getattr(self.config, "laplacian_gamma", None)
            anchor_lap_k = int(getattr(self.config, "anchor_laplacian_k", assign_k) or assign_k)
            self.L_within, self.L_between = build_laplacians_from_anchor_labels(
                anchor=self.anchor,
                anchor_y=self.anchor_y,
                gamma=gamma,
                k_neighbors=anchor_lap_k,
                logger=self.logger,
            )
        else:
            self.L_within = None
            self.L_between = None

        graph_k = getattr(self.config, "graph_knn_k", None)
        graph_k = int(graph_k) if graph_k is not None else None
        if self._needs_graph_laplacian() and graph_k is not None and graph_k > 0:
            metric = getattr(self.config, "graph_knn_metric", "euclidean") or "euclidean"
            adjacency, graph_Lw, graph_Lb = build_laplacians_from_intermediate_data(
                Xs_inter=self.Xs_train_inter,
                anchors_inter=self.anchors_inter,
                ys=dataset.ys_train,
                k_neighbors=graph_k,
                metric=metric,
                normalize=bool(getattr(self.config, "graph_laplacian_normalize", True)),
                logger=self.logger,
            )
            self.graph_adjacency = adjacency
            self.graph_L_within = graph_Lw
            self.graph_L_between = graph_Lb
        else:
            self.graph_adjacency = None
            self.graph_L_within = None
            self.graph_L_between = None

        return IntermediateArtifacts(
            dataset=dataset,
            anchor=self.anchor,
            anchor_test=self.anchor_test,
            anchor_y=self.anchor_y,
            anchor_y_test=self.anchor_y_test,
            Xs_train_inter=list(self.Xs_train_inter),
            Xs_test_inter=list(self.Xs_test_inter),
            anchors_inter=list(self.anchors_inter),
            anchors_test_inter=list(self.anchors_test_inter),
            L_within=self.L_within,
            L_between=self.L_between,
            graph_adjacency=self.graph_adjacency,
            graph_L_within=self.graph_L_within,
            graph_L_between=self.graph_L_between,
        )

    def _build_projectors(self, dataset: DatasetArtifacts):
        tf = getattr(self.config, "True_F_type", None)
        if isinstance(tf, (list, tuple)) and tf:
            ftypes = list(tf)
        elif isinstance(tf, str) and tf:
            ftypes = _FTYPE_MIXTURES.get(tf, [tf])
        else:
            ftypes = [getattr(self.config, "F_type", "svd")]

        # samespan 系は共通基底を共有できるよう、最初の機関データで一度だけ作成
        shared_basis = None
        use_shared_span = any(str(ft).lower().startswith("samespan") for ft in ftypes)
        if use_shared_span and dataset.Xs_train:
            base_X = dataset.Xs_train[0]
            try:
                from src.dimensionality_reduction import SVDScratch

                svd = SVDScratch(n_components=self.config.dim_intermediate, center=False)
                svd.fit(base_X)
                shared_basis = svd.components_.T
                setattr(self.config, "_shared_F_basis", shared_basis)
            except Exception:
                shared_basis = None
        
        # 特別扱い: 共有部分空間 F_type=shared_subspace
        if all(str(ft).lower() == "shared_subspace" for ft in ftypes):
            count = len(dataset.Xs_train)
            if count == 0:
                return []
            num_features = dataset.Xs_train[0].shape[1]
            projectors = build_shared_subspace_projectors(
                num_features=num_features,
                num_institution=count,
                config=self.config,
            )
            return projectors

        base_seed = int(getattr(self.config, "f_seed", 0) or 0)
        projectors = []
        count = len(dataset.Xs_train)

        # Build per-institution F_type list, with optional svd_ratio override
        ftypes_per_inst = [ftypes[i % len(ftypes)] for i in range(count)] if count > 0 else []
        try:
            svd_ratio = float(getattr(self.config, "svd_ratio", None))
        except (TypeError, ValueError):
            svd_ratio = None
        if svd_ratio is not None:
            svd_ratio = min(max(svd_ratio, 0.0), 1.0)
            n_svd = int(round(count * svd_ratio))
            for i in range(min(n_svd, count)):
                ftypes_per_inst[i] = "svd"

        index_iter = tqdm(range(count), desc="Building intermediate projectors", unit="inst") if count > 1 else range(count)
        for idx in index_iter:
            X_train = dataset.Xs_train[idx]
            current_ftype = ftypes_per_inst[idx]
            y_train = dataset.ys_train[idx] if idx < len(dataset.ys_train) else None
            projector = build_dimensionality_projector(
                X=X_train,
                n_components=self.config.dim_intermediate,
                F_type=current_ftype,
                seed=base_seed + idx,
                y=y_train,
                config=self.config,
            )
            projectors.append(projector)
        return projectors

    def _load_from_store(self) -> IntermediateArtifacts | None:
        cached = self.store.load("intermediate", getattr(self.config, "intermediate_name", None))
        if isinstance(cached, IntermediateArtifacts):
            return cached
        return None

    def _needs_anchor_laplacian(self) -> bool:
        if float(getattr(self.config, "lw_alpha", 0.0) or 0.0) > 0.0:
            return True
        g_type_raw = getattr(self.config, "G_type", "")
        if isinstance(g_type_raw, str):
            return g_type_raw.lower() in {"graph_nonlinear", "graph_nonlinear_minimize", "graph_nonlinear_maximize"}
        try:
            return any(str(val).lower() in {"graph_nonlinear", "graph_nonlinear_minimize", "graph_nonlinear_maximize"} for val in g_type_raw)
        except TypeError:
            return False

    def _needs_graph_laplacian(self) -> bool:
        g_type_raw = getattr(self.config, "G_type", "")
        if isinstance(g_type_raw, str):
            return g_type_raw.lower() in {
                "kernel_graph_gep",
                "kernel_graph_gep_minimize",
                "kernel_graph_gep_maximize",
                "graph_nonlinear_x",
                "graph_nonlinear_x_minimize",
                "graph_nonlinear_x_maximize",
            }
        try:
            return any(
                str(val).lower() in {
                    "kernel_graph_gep",
                    "kernel_graph_gep_minimize",
                    "kernel_graph_gep_maximize",
                    "graph_nonlinear_x",
                    "graph_nonlinear_x_minimize",
                    "graph_nonlinear_x_maximize",
                }
                for val in g_type_raw
            )
        except TypeError:
            return False

    def _maybe_attach_anchor_laplacians(self, artifacts: IntermediateArtifacts) -> IntermediateArtifacts:
        if not self._needs_anchor_laplacian():
            return artifacts
        if (artifacts.L_within is not None) and (artifacts.L_between is not None):
            return artifacts
        if artifacts.anchor.size == 0 or artifacts.anchor_y.size == 0:
            return artifacts
        assign_k = int(getattr(self.config, "anchor_assign_k", 10) or 10)
        anchor_lap_k = int(getattr(self.config, "anchor_laplacian_k", assign_k) or assign_k)
        gamma = getattr(self.config, "laplacian_gamma", None)
        L_within, L_between = build_laplacians_from_anchor_labels(
            anchor=artifacts.anchor,
            anchor_y=artifacts.anchor_y,
            gamma=gamma,
            k_neighbors=anchor_lap_k,
            logger=self.logger,
        )
        return replace(artifacts, L_within=L_within, L_between=L_between)

    def _maybe_attach_graph_laplacians(self, artifacts: IntermediateArtifacts) -> IntermediateArtifacts:
        if not self._needs_graph_laplacian():
            return artifacts
        if artifacts.graph_L_within is not None and artifacts.graph_L_between is not None:
            return artifacts
        graph_k = getattr(self.config, "graph_knn_k", None)
        if graph_k is None:
            return artifacts
        graph_k = int(graph_k)
        if graph_k <= 0:
            return artifacts
        metric = getattr(self.config, "graph_knn_metric", "euclidean") or "euclidean"
        adjacency, graph_Lw, graph_Lb = build_laplacians_from_intermediate_data(
            Xs_inter=artifacts.Xs_train_inter,
            anchors_inter=artifacts.anchors_inter,
            ys=artifacts.dataset.ys_train,
            k_neighbors=graph_k,
            metric=metric,
            normalize=bool(getattr(self.config, "graph_laplacian_normalize", True)),
            logger=self.logger,
        )
        return replace(
            artifacts,
            graph_adjacency=adjacency,
            graph_L_within=graph_Lw,
            graph_L_between=graph_Lb,
        )

    def _sync_from_artifacts(self, artifacts: IntermediateArtifacts) -> None:
        self.artifacts = artifacts
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

    def _maybe_normalize_artifacts(self, artifacts: IntermediateArtifacts) -> IntermediateArtifacts:
        if not getattr(self.config, "inter_normalization", False):
            return artifacts

        anchors_inter = []
        anchors_test_inter = []
        Xs_train_inter = []
        Xs_test_inter = []

        count = len(artifacts.anchors_inter)
        for idx in range(count):
            anchor = artifacts.anchors_inter[idx]
            anchor_test = artifacts.anchors_test_inter[idx] if idx < len(artifacts.anchors_test_inter) else None
            X_train = artifacts.Xs_train_inter[idx] if idx < len(artifacts.Xs_train_inter) else None
            X_test = artifacts.Xs_test_inter[idx] if idx < len(artifacts.Xs_test_inter) else None

            if anchor is None or anchor.size == 0:
                anchors_inter.append(anchor)
                anchors_test_inter.append(anchor_test)
                Xs_train_inter.append(X_train)
                Xs_test_inter.append(X_test)
                continue

            scaler = StandardScaler()
            anchor_scaled = scaler.fit_transform(anchor)
            anchors_inter.append(anchor_scaled)
            anchors_test_inter.append(scaler.transform(anchor_test) if (anchor_test is not None and anchor_test.size > 0) else anchor_test)
            Xs_train_inter.append(scaler.transform(X_train) if (X_train is not None and X_train.size > 0) else X_train)
            Xs_test_inter.append(scaler.transform(X_test) if (X_test is not None and X_test.size > 0) else X_test)

        return replace(
            artifacts,
            anchors_inter=anchors_inter,
            anchors_test_inter=anchors_test_inter,
            Xs_train_inter=Xs_train_inter,
            Xs_test_inter=Xs_test_inter,
        )

_FTYPE_MIXTURES = {
    "kernel_pca_svd_mixed": ["kernel_pca_self_tuning", "svd"],
    "ae_dm_mixed": ["ae", "dm"],
    "ae_svd_mixed": ["ae", "svd"],
    "ae_dm_svd_mixed": ["ae", "dm", "svd"],
    "ae_dm_kpca_svd_mixed": ["ae", "dm", "kernel_pca_gamma_fixed", "svd"],
    "umap_svd_mixed": ["umap", "svd"],
}


__all__ = ["IntermediateExpressionBuilder"]
