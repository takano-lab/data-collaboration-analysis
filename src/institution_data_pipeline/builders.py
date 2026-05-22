from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.config import Config
from src.common import ArtifactStore, DatasetArtifacts
from .institution_data import prepare_institutional_dataset
from .load_data import load_data

MAX_DATASET_ARTIFACTS = 100

@dataclass
class DatasetCache:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class InstitutionDatasetBuilder:
    """
    Responsible only for raw data loading and institution-level train/test splits.
    """

    def __init__(self, *, config: Config, logger, store: Optional[ArtifactStore] = None) -> None:
        self.config = config
        self.logger = logger
        self.store = store or ArtifactStore(logger=logger)

        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.Xs_train: list = []
        self.Xs_test: list = []
        self.ys_train: list = []
        self.ys_test: list = []
        self.public_anchor = None
        self.public_anchor_y = None
        self.artifacts: DatasetArtifacts | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> DatasetArtifacts:
        if self.artifacts is not None:
            return self.artifacts

        if getattr(self.config, "load_df_data", False):
            loaded = self._load_from_store()
            if loaded:
                artifacts = self._apply_dimension_overrides(loaded)
                self._sync_from_artifacts(artifacts)
                self.artifacts = artifacts
                if self.logger:
                    self.logger.info("Loaded dataset artifacts from cache.")
                return artifacts

        artifacts = self._build_dataset()
        artifacts = self._apply_dimension_overrides(artifacts)
        if getattr(self.config, "load_df_data", False):
            self.store.save("dataset", getattr(self.config, "df_name", None), artifacts)
            self.store.prune("dataset", keep=MAX_DATASET_ARTIFACTS)
        self._sync_from_artifacts(artifacts)
        self.artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------ #
    def _build_dataset(self) -> DatasetArtifacts:
        if self.logger:
            self.logger.info("Building dataset artifacts (load_data -> prepare_institutional_dataset).")
        raw_df = load_data(config=self.config)
        (
            Xs_train,
            Xs_test,
            ys_train,
            ys_test,
            train_df,
            test_df,
            public_anchor,
            public_anchor_y,
        ) = prepare_institutional_dataset(raw_df, self.config)

        return DatasetArtifacts(
            train_df=train_df,
            test_df=test_df,
            Xs_train=list(Xs_train),
            Xs_test=list(Xs_test),
            ys_train=list(ys_train),
            ys_test=list(ys_test),
            public_anchor=public_anchor,
            public_anchor_y=public_anchor_y,
        )

    def _load_from_store(self) -> DatasetArtifacts | None:
        cached = self.store.load("dataset", getattr(self.config, "df_name", None))
        if isinstance(cached, DatasetArtifacts):
            if self._cached_artifact_is_stale(cached):
                if self.logger:
                    self.logger.info("Ignoring stale dataset artifact; rebuilding dataset artifacts.")
                return None
            self._sync_from_artifacts(cached)
            self.artifacts = cached
            return cached
        return None

    def _cached_artifact_is_stale(self, artifacts: DatasetArtifacts) -> bool:
        dataset = str(getattr(self.config, "dataset", "") or "")
        anchor_method = getattr(self.config, "anchor_method", None)
        use_public_anchor = bool(getattr(self.config, "use_public_anchor", True))
        if dataset == "har_subject" and anchor_method == "smote" and use_public_anchor:
            public_anchor = getattr(artifacts, "public_anchor", None)
            if public_anchor is None or getattr(public_anchor, "size", 0) == 0:
                return True
            prefer_raw = getattr(self.config, "har_subject_test_from_train_remaining", None)
            prefer_train_remaining_test = True if prefer_raw is None else bool(prefer_raw)
            test_df = getattr(artifacts, "test_df", None)
            if prefer_train_remaining_test and test_df is not None and "split" in test_df.columns:
                return not test_df["split"].astype(str).eq("train").all()
            if not prefer_train_remaining_test:
                return test_df is not None and len(test_df) < 2947
        return False

    def _sync_from_artifacts(self, artifacts: DatasetArtifacts) -> None:
        self.train_df = artifacts.train_df.copy()
        self.test_df = artifacts.test_df.copy()
        self.Xs_train = list(artifacts.Xs_train)
        self.Xs_test = list(artifacts.Xs_test)
        self.ys_train = list(artifacts.ys_train)
        self.ys_test = list(artifacts.ys_test)
        self.public_anchor = getattr(artifacts, "public_anchor", None)
        self.public_anchor_y = getattr(artifacts, "public_anchor_y", None)

    def _apply_dimension_overrides(self, artifacts: DatasetArtifacts) -> DatasetArtifacts:
        xs_train = artifacts.Xs_train
        actual_institutions = len(xs_train)
        try:
            self.config.num_institution = int(actual_institutions)
        except (TypeError, ValueError):
            self.config.num_institution = actual_institutions

        base_dim = xs_train[0].shape[1] if xs_train else None

        def _resolve_dim(value):
            if base_dim is None or value is None:
                return value
            if isinstance(value, str):
                v = value.strip()
                try:
                    if v.startswith(("+", "-")):
                        delta = float(v)
                        return max(1, int(round(base_dim + delta)))
                    if v.startswith("*"):
                        factor = float(v[1:])
                        return max(1, int(round(base_dim * factor)))
                    if v.startswith("/"):
                        divisor = float(v[1:])
                        if divisor != 0:
                            return max(1, int(round(base_dim / divisor)))
                except ValueError:
                    return value
            return value

        max_dim_raw = getattr(self.config, "max_dim", 1000)
        if isinstance(max_dim_raw, (int, float)) and max_dim_raw is not None:
            max_dim = max(1, int(round(max_dim_raw)))
        else:
            max_dim = 1000

        resolved_dim_inter = _resolve_dim(getattr(self.config, "dim_intermediate", None))
        if resolved_dim_inter is not None:
            if isinstance(resolved_dim_inter, (int, float)):
                self.config.dim_intermediate = max(1, min(int(round(resolved_dim_inter)), max_dim))
            else:
                self.config.dim_intermediate = resolved_dim_inter

        resolved_dim_integ = _resolve_dim(getattr(self.config, "dim_integrate", None))
        if resolved_dim_integ is not None:
            if isinstance(resolved_dim_integ, (int, float)):
                self.config.dim_integrate = max(1, min(int(round(resolved_dim_integ)), max_dim))
            else:
                self.config.dim_integrate = resolved_dim_integ

        return artifacts

__all__ = ["InstitutionDatasetBuilder"]
