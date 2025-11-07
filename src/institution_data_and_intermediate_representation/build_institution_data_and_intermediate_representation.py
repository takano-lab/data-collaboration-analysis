from __future__ import annotations

from typing import Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config.config import Config
from src.dimensionality_reduction import build_dimensionality_projector

from .anchor_data import (
    assign_anchor_labels as _assign_anchor_labels,
    build_laplacians_from_anchor_labels as _build_laplacians_from_anchor_labels,
    produce_anchor as _produce_anchor,
)
from .data_presevation import DataPreservationManager
from .institution_data import prepare_institutional_dataset as _prepare_institutional_dataset
from .load_data import load_data as _load_data

logger = TypeVar("logger")


class InstitutionDatasetBuilder:
    """
    Handles dataset loading, institution-level splitting, anchor generation,
    intermediate representation building, and preservation/loading of those artifacts.
    """

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
    ) -> None:
        self.config = config
        self.logger = logger
        self.train_df = train_df.copy(deep=True) if isinstance(train_df, pd.DataFrame) else pd.DataFrame(train_df or [])
        self.test_df = test_df.copy(deep=True) if isinstance(test_df, pd.DataFrame) else pd.DataFrame(test_df or [])
        self.Xs_train: list[np.ndarray] = list(Xs_train or [])
        self.Xs_test: list[np.ndarray] = list(Xs_test or [])
        self.ys_train: list[np.ndarray] = list(ys_train or [])
        self.ys_test: list[np.ndarray] = list(ys_test or [])

        self.anchor: np.ndarray = np.array([])
        self.anchor_y: np.ndarray = np.array([])
        self.anchor_test: np.ndarray = np.array([])
        self.anchor_y_test: np.ndarray = np.array([])

        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []
        self.anchors_inter: list[np.ndarray] = []
        self.anchors_test_inter: list[np.ndarray] = []

        self.Z_integ: Optional[np.ndarray] = None
        self.L_within: Optional[np.ndarray] = None
        self.L_between: Optional[np.ndarray] = None

        self._preserver = DataPreservationManager(config, logger)

    # ------------------------------------------------------------------ #
    # Convenience getters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stack_features(parts: Sequence[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([])
        return np.vstack(parts)

    @staticmethod
    def _stack_labels(parts: Sequence[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([])
        return np.hstack(parts)

    # ------------------------------------------------------------------ #
    # Preservation helpers
    # ------------------------------------------------------------------ #
    def _build_df_bundle(self) -> dict[str, Sequence[object]]:
        bundle: dict[str, Sequence[object]] = {}
        if self.Xs_train:
            bundle["Xs_train"] = list(self.Xs_train)
        if self.Xs_test:
            bundle["Xs_test"] = list(self.Xs_test)
        if self.ys_train:
            bundle["ys_train"] = list(self.ys_train)
        if self.ys_test:
            bundle["ys_test"] = list(self.ys_test)
        if isinstance(self.train_df, pd.DataFrame) and not self.train_df.empty:
            bundle["train_df"] = [self.train_df.copy(deep=True)]
        if isinstance(self.test_df, pd.DataFrame) and not self.test_df.empty:
            bundle["test_df"] = [self.test_df.copy(deep=True)]
        return bundle

    def _save_current_df_bundle(self) -> None:
        if not bool(getattr(self.config, "load_df_data", False)):
            return
        bundle = self._build_df_bundle()
        if bundle:
            self._preserver.save_bundle("df", getattr(self.config, "df_name", None), bundle)

    @staticmethod
    def _restore_dataframe(saved: Sequence[object], current: pd.DataFrame) -> pd.DataFrame:
        if not saved:
            return current
        for candidate in saved:
            if isinstance(candidate, pd.DataFrame):
                return candidate.copy(deep=True)
        first = next((candidate for candidate in saved if candidate is not None), None)
        if first is None:
            return current
        arr = np.asarray(first)
        if arr.ndim != 2:
            return current
        columns = list(current.columns) if isinstance(current, pd.DataFrame) else None
        try:
            if columns and len(columns) == arr.shape[1]:
                return pd.DataFrame(arr, columns=columns)
            return pd.DataFrame(arr)
        except Exception:
            return current

    def load_existing_df_data(self) -> None:
        """
        Load preserved institutional dataset if available.
        """
        preserved = self._preserver.load_bundle("df", getattr(self.config, "df_name", None))
        if preserved:
            self.Xs_train = list(preserved.get("Xs_train", self.Xs_train) or [])
            self.Xs_test = list(preserved.get("Xs_test", self.Xs_test) or [])
            self.ys_train = list(preserved.get("ys_train", self.ys_train) or [])
            self.ys_test = list(preserved.get("ys_test", self.ys_test) or [])
            self.train_df = self._restore_dataframe(list(preserved.get("train_df", [])), self.train_df)
            self.test_df = self._restore_dataframe(list(preserved.get("test_df", [])), self.test_df)
            loaded = bool(self.Xs_train and self.Xs_test)
            self._log(f"[preserved] df data loaded: {loaded}")
        else:
            self._log("[preserved] df data not found; rebuilding.", level="warning")

        should_preserve = bool(getattr(self.config, "load_df_data", False))
        should_preserve = bool(getattr(self.config, "load_df_data", False))
        if not self.Xs_train or not self.Xs_test:
            if not self.train_df.empty and not self.test_df.empty:
                self.Xs_train, self.Xs_test, self.ys_train, self.ys_test = self.train_test_split(
                    train_df=self.train_df,
                    test_df=self.test_df,
                    num_institution=self.config.num_institution,
                    num_institution_user=self.config.num_institution_user,
                    y_name=self.config.y_name,
                )
                if should_preserve:
                    self._save_current_df_bundle()
                    self._log("Saved rebuilt df bundle to preserved storage.")

    def _load_intermediate_data(self) -> bool:
        should_preserve = bool(getattr(self.config, "load_intermediate_data", False))
        if not should_preserve:
            return False
        preserved = self._preserver.load_bundle("intermediate", getattr(self.config, "intermediate_name", None))
        if preserved:
            self.Xs_train_inter = list(preserved.get("Xs_train_inter", self.Xs_train_inter) or [])
            self.Xs_test_inter = list(preserved.get("Xs_test_inter", self.Xs_test_inter) or [])
            self.anchors_inter = list(preserved.get("anchors_inter", self.anchors_inter) or [])
            self.anchors_test_inter = list(preserved.get("anchors_test_inter", self.anchors_test_inter) or [])
            loaded = bool(self.Xs_train_inter and self.Xs_test_inter and self.anchors_inter and self.anchors_test_inter)
            self._log(f"[preserved] intermediate data loaded: {loaded}")
            return loaded
        self._log("[preserved] intermediate data not found or invalid; rebuilding.", level="warning")
        return False

    def _save_intermediate_bundle(self) -> None:
        if not bool(getattr(self.config, "load_intermediate_data", False)):
            return
        bundle = {
            "Xs_train_inter": list(self.Xs_train_inter or []),
            "Xs_test_inter": list(self.Xs_test_inter or []),
            "anchors_inter": list(self.anchors_inter or []),
            "anchors_test_inter": list(self.anchors_test_inter or []),
        }
        if any(bundle.values()):
            self._preserver.save_bundle("intermediate", getattr(self.config, "intermediate_name", None), bundle)
            self._log("Saved rebuilt intermediate bundle to preserved storage.")

    # ------------------------------------------------------------------ #
    # Anchor helpers
    # ------------------------------------------------------------------ #
    def produce_anchor(self, num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
        return _produce_anchor(
            num_row=num_row,
            num_col=num_col,
            seed=seed,
            config=self.config,
            train_df=self.train_df,
            Xs_train=self.Xs_train,
            Xs_test=self.Xs_test,
            ys_train=self.ys_train,
            ys_test=self.ys_test,
        )

    def assign_anchor_labels(self, k: int = 5) -> None:
        self.anchor_y, self.anchor_y_test = _assign_anchor_labels(
            anchor=self.anchor,
            anchor_test=self.anchor_test,
            Xs_train=self.Xs_train,
            ys_train=self.ys_train,
            k=k,
        )

    def build_laplacians_from_anchor_labels(self, gamma: Optional[float] = None) -> None:
        L_within, L_between = _build_laplacians_from_anchor_labels(
            anchor=self.anchor,
            anchor_y=self.anchor_y,
            gamma=gamma,
            logger=self.logger,
        )
        self.L_within = L_within
        self.L_between = L_between

    # ------------------------------------------------------------------ #
    # Public orchestration
    # ------------------------------------------------------------------ #
    def make_institution_dataset(self) -> None:
        """
        Build or load train/test DataFrames and institution-wise splits.
        """
        load_preserve = bool(getattr(self.config, "load_df_data", False))
        if load_preserve:
            self.load_existing_df_data()
        if self.Xs_train and self.Xs_test and not self.train_df.empty and not self.test_df.empty:
            return

        self._log("loading dataset for institution splits")
        df = _load_data(config=self.config)
        (
            Xs_train,
            Xs_test,
            ys_train,
            ys_test,
            train_df,
            test_df,
        ) = _prepare_institutional_dataset(df, self.config)
        self.Xs_train = list(Xs_train)
        self.Xs_test = list(Xs_test)
        self.ys_train = list(ys_train)
        self.ys_test = list(ys_test)
        self.train_df = train_df
        self.test_df = test_df
        if load_preserve:
            self._save_current_df_bundle()

    def make_intermediate_expression(self) -> None:
        """
        Build or load anchors and intermediate representations.
        """
        if not self.Xs_train or not self.Xs_test:
            self.make_institution_dataset()

        num_col = self.Xs_train[0].shape[1]
        if self.anchor.size == 0:
            self.anchor = self.produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_col,
                seed=self.config.seed,
            )
        if self.anchor_test.size == 0:
            self.anchor_test = self.produce_anchor(
                num_row=self.config.num_anchor_data,
                num_col=num_col,
                seed=self.config.seed + 1,
            )

        if self._load_intermediate_data():
            return
        self._build_intermediate_expression()
        self._save_intermediate_bundle()

    # ------------------------------------------------------------------ #
    # Intermediate representation
    # ------------------------------------------------------------------ #
    def _build_intermediate_expression(self) -> None:
        self._log("******************** Building F (intermediate) ********************")
        self.config.f_seed = 0

        tf = getattr(self.config, "True_F_type", None)
        if isinstance(tf, (list, tuple)) and len(tf) > 0:
            ftype_sequence = list(tf)
        elif isinstance(tf, str) and len(tf) > 0:
            mix_map = {
                "kernel_pca_svd_mixed": ["kernel_pca_self_tuning", "svd"],
                "ae_dm_mixed": ["ae", "dm"],
                "ae_svd_mixed": ["ae", "svd"],
                "ae_dm_svd_mixed": ["ae", "dm", "svd"],
                "ae_dm_kpca_svd_mixed": ["ae", "dm", "kernel_pca_gamma_fixed", "svd"],
            }
            ftype_sequence = mix_map.get(tf, [tf])
        else:
            ftype_sequence = [self.config.F_type]

        projectors: list = []
        for idx, (X_train, _) in enumerate(zip(self.Xs_train, self.Xs_test)):
            self.config.F_type = ftype_sequence[idx % len(ftype_sequence)]
            current_seed = self.config.f_seed
            y_train = self.ys_train[idx] if idx < len(self.ys_train) else None
            projector = build_dimensionality_projector(
                X=X_train,
                n_components=self.config.dim_intermediate,
                F_type=self.config.F_type,
                seed=current_seed,
                y=y_train,
                config=self.config,
            )
            projectors.append(projector)
            self.config.f_seed += 1

        self.Xs_train_inter = []
        self.Xs_test_inter = []
        self.anchors_inter = []
        self.anchors_test_inter = []

        inter_norm = getattr(self.config, "inter_normalization", False)
        for projector, X_train, X_test in zip(projectors, self.Xs_train, self.Xs_test):
            X_train_reduced = projector(X_train)
            X_test_reduced = projector(X_test)
            anchor_reduced = projector(self.anchor)
            anchor_test_reduced = projector(self.anchor_test)

            if not inter_norm:
                self.Xs_train_inter.append(X_train_reduced)
                self.Xs_test_inter.append(X_test_reduced)
                self.anchors_inter.append(anchor_reduced)
                self.anchors_test_inter.append(anchor_test_reduced)
            else:
                scaler = StandardScaler()
                anchor_scaled = scaler.fit_transform(anchor_reduced)
                self.anchors_inter.append(anchor_scaled)
                self.Xs_train_inter.append(scaler.transform(X_train_reduced))
                self.Xs_test_inter.append(scaler.transform(X_test_reduced))
                if anchor_test_reduced is not None:
                    self.anchors_test_inter.append(scaler.transform(anchor_test_reduced))
                else:
                    self.anchors_test_inter.append(None)

        if self.Xs_train_inter:
            self._log(f"Intermediate representation shape: {self.Xs_train_inter[0].shape}")

    def save_artifacts(
        self,
        *,
        save_dir: Optional[str] = None,
        items: Optional[Sequence[str]] = None,
        filename_suffix: Optional[str] = None,
    ) -> dict:
        return self._preserver.save_artifacts(
            self,
            save_dir=save_dir,
            items=items,
            filename_suffix=filename_suffix,
        )

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    @staticmethod
    def train_test_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_institution: int,
        num_institution_user: int,
        y_name: str = "target",
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        train_df = train_df.copy()
        test_df = test_df.copy()
        y_train_ser = train_df[y_name]
        X_train_df = train_df.drop(y_name, axis=1)
        y_test_ser = test_df[y_name]
        X_test_df = test_df.drop(y_name, axis=1)

        Xs_train, Xs_test = [], []
        ys_train, ys_test = [], []

        for institute_start in tqdm(range(0, num_institution * num_institution_user, num_institution_user)):
            Xs_train.append(X_train_df[institute_start:institute_start + num_institution_user].values)
            Xs_test.append(X_test_df[institute_start:institute_start + num_institution_user].values)
            ys_train.append(y_train_ser[institute_start:institute_start + num_institution_user].values)
            ys_test.append(y_test_ser[institute_start:institute_start + num_institution_user].values)

        return Xs_train, Xs_test, ys_train, ys_test

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


__all__ = ["InstitutionDatasetBuilder"]
