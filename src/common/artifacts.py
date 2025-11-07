from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def _ensure_array_list(items: List[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(arr) for arr in items]


@dataclass(frozen=True)
class DatasetArtifacts:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    Xs_train: List[np.ndarray]
    Xs_test: List[np.ndarray]
    ys_train: List[np.ndarray]
    ys_test: List[np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_df", self.train_df.copy())
        object.__setattr__(self, "test_df", self.test_df.copy())
        object.__setattr__(self, "Xs_train", _ensure_array_list(self.Xs_train))
        object.__setattr__(self, "Xs_test", _ensure_array_list(self.Xs_test))
        object.__setattr__(self, "ys_train", _ensure_array_list(self.ys_train))
        object.__setattr__(self, "ys_test", _ensure_array_list(self.ys_test))


@dataclass(frozen=True)
class IntermediateArtifacts:
    dataset: DatasetArtifacts
    anchor: np.ndarray
    anchor_test: np.ndarray
    anchor_y: np.ndarray
    anchor_y_test: np.ndarray
    Xs_train_inter: List[np.ndarray]
    Xs_test_inter: List[np.ndarray]
    anchors_inter: List[np.ndarray]
    anchors_test_inter: List[np.ndarray]
    L_within: Optional[np.ndarray] = None
    L_between: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", np.asarray(self.anchor))
        object.__setattr__(self, "anchor_test", np.asarray(self.anchor_test))
        object.__setattr__(self, "anchor_y", np.asarray(self.anchor_y))
        object.__setattr__(self, "anchor_y_test", np.asarray(self.anchor_y_test))
        object.__setattr__(self, "Xs_train_inter", _ensure_array_list(self.Xs_train_inter))
        object.__setattr__(self, "Xs_test_inter", _ensure_array_list(self.Xs_test_inter))
        object.__setattr__(self, "anchors_inter", _ensure_array_list(self.anchors_inter))
        object.__setattr__(self, "anchors_test_inter", _ensure_array_list(self.anchors_test_inter))


@dataclass(frozen=True)
class IntegratedArtifacts:
    intermediate: IntermediateArtifacts
    Xs_train_integ: List[np.ndarray]
    Xs_test_integ: List[np.ndarray]
    anchors_integ: List[np.ndarray]
    anchors_test_integ: List[np.ndarray]
    ys_train_integ: List[np.ndarray]
    ys_test_integ: List[np.ndarray]
    Z_integ: Optional[np.ndarray] = None
    metrics: Optional[dict] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "Xs_train_integ", _ensure_array_list(self.Xs_train_integ))
        object.__setattr__(self, "Xs_test_integ", _ensure_array_list(self.Xs_test_integ))
        object.__setattr__(self, "anchors_integ", _ensure_array_list(self.anchors_integ))
        object.__setattr__(self, "anchors_test_integ", _ensure_array_list(self.anchors_test_integ))
        object.__setattr__(self, "ys_train_integ", _ensure_array_list(self.ys_train_integ))
        object.__setattr__(self, "ys_test_integ", _ensure_array_list(self.ys_test_integ))
        if self.Z_integ is not None:
            object.__setattr__(self, "Z_integ", np.asarray(self.Z_integ))
