from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.config import Config
from src.common import ArtifactStore, DatasetArtifacts
from .institution_data import prepare_institutional_dataset
from .load_data import load_data


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
        self.artifacts: DatasetArtifacts | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> DatasetArtifacts:
        if self.artifacts is not None:
            return self.artifacts

        if getattr(self.config, "load_df_data", False):
            loaded = self._load_from_store()
            if loaded:
                self._log("Loaded dataset artifacts from cache.")
                return loaded

        artifacts = self._build_dataset()
        if getattr(self.config, "load_df_data", False):
            self.store.save("dataset", getattr(self.config, "df_name", None), artifacts)
        self._sync_from_artifacts(artifacts)
        self.artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------ #
    def _build_dataset(self) -> DatasetArtifacts:
        self._log("Building dataset artifacts (load_data -> prepare_institutional_dataset).")
        raw_df = load_data(config=self.config)
        (
            Xs_train,
            Xs_test,
            ys_train,
            ys_test,
            train_df,
            test_df,
        ) = prepare_institutional_dataset(raw_df, self.config)
        return DatasetArtifacts(
            train_df=train_df,
            test_df=test_df,
            Xs_train=list(Xs_train),
            Xs_test=list(Xs_test),
            ys_train=list(ys_train),
            ys_test=list(ys_test),
        )

    def _load_from_store(self) -> DatasetArtifacts | None:
        cached = self.store.load("dataset", getattr(self.config, "df_name", None))
        if isinstance(cached, DatasetArtifacts):
            self._sync_from_artifacts(cached)
            self.artifacts = cached
            return cached
        return None

    def _sync_from_artifacts(self, artifacts: DatasetArtifacts) -> None:
        self.train_df = artifacts.train_df.copy()
        self.test_df = artifacts.test_df.copy()
        self.Xs_train = list(artifacts.Xs_train)
        self.Xs_test = list(artifacts.Xs_test)
        self.ys_train = list(artifacts.ys_train)
        self.ys_test = list(artifacts.ys_test)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)


__all__ = ["InstitutionDatasetBuilder"]
