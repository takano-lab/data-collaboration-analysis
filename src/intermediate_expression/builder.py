from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config.config import Config
from src.common import ArtifactStore, DatasetArtifacts, IntermediateArtifacts
from src.dimensionality_reduction import build_dimensionality_projector
from .anchor_utils import (
    assign_anchor_labels,
    build_laplacians_from_anchor_labels,
    produce_anchor,
)


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

    # ------------------------------------------------------------------ #
    def run(self, dataset_artifacts: DatasetArtifacts) -> IntermediateArtifacts:
        self.dataset_artifacts = dataset_artifacts
        if getattr(self.config, "load_intermediate_data", False):
            loaded = self._load_from_store()
            if loaded is not None:
                artifacts = self._maybe_normalize_artifacts(loaded)
                self._sync_from_artifacts(artifacts)
                self.artifacts = artifacts
                if self.logger:
                    self.logger.info("Loaded intermediate artifacts from cache.")
                return artifacts

        raw_artifacts = self._build_intermediate(dataset_artifacts)

        if getattr(self.config, "load_intermediate_data", False):
            self.store.save("intermediate", getattr(self.config, "intermediate_name", None), raw_artifacts)

        artifacts = self._maybe_normalize_artifacts(raw_artifacts)
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
        self.anchor = produce_anchor(
            num_row=self.config.num_anchor_data,
            num_col=num_features,
            seed=self.config.seed,
            config=self.config,
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
            config=self.config,
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

        assign_k = int(getattr(self.config, "anchor_assign_k", 5) or 5)
        self.anchor_y, self.anchor_y_test = assign_anchor_labels(
            anchor=self.anchor,
            anchor_test=self.anchor_test,
            Xs_train=dataset.Xs_train,
            ys_train=dataset.ys_train,
            k=assign_k,
        )

        lw_alpha = float(getattr(self.config, "lw_alpha", 0.0) or 0.0)
        if lw_alpha > 0.0:
            gamma = getattr(self.config, "laplacian_gamma", None)
            self.L_within, self.L_between = build_laplacians_from_anchor_labels(
                anchor=self.anchor,
                anchor_y=self.anchor_y,
                gamma=gamma,
                logger=self.logger,
            )
        else:
            self.L_within = None
            self.L_between = None

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
        )

    def _build_projectors(self, dataset: DatasetArtifacts):
        tf = getattr(self.config, "True_F_type", None)
        if isinstance(tf, (list, tuple)) and tf:
            ftypes = list(tf)
        elif isinstance(tf, str) and tf:
            ftypes = _FTYPE_MIXTURES.get(tf, [tf])
        else:
            ftypes = [getattr(self.config, "F_type", "svd")]

        base_seed = int(getattr(self.config, "f_seed", 0) or 0)
        projectors = []
        count = len(dataset.Xs_train)
        index_iter = tqdm(range(count), desc="Building intermediate projectors", unit="inst") if count > 1 else range(count)
        for idx in index_iter:
            X_train = dataset.Xs_train[idx]
            current_ftype = ftypes[idx % len(ftypes)]
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
