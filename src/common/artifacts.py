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
    smote_anchor: Optional[np.ndarray] = None
    smote_anchor_y: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_df", self.train_df.copy())
        object.__setattr__(self, "test_df", self.test_df.copy())
        object.__setattr__(self, "Xs_train", _ensure_array_list(self.Xs_train))
        object.__setattr__(self, "Xs_test", _ensure_array_list(self.Xs_test))
        object.__setattr__(self, "ys_train", _ensure_array_list(self.ys_train))
        object.__setattr__(self, "ys_test", _ensure_array_list(self.ys_test))
        if self.smote_anchor is not None:
            object.__setattr__(self, "smote_anchor", np.asarray(self.smote_anchor))
        if self.smote_anchor_y is not None:
            object.__setattr__(self, "smote_anchor_y", np.asarray(self.smote_anchor_y))


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
    graph_adjacency: Optional[np.ndarray] = None
    graph_L_within: Optional[np.ndarray] = None
    graph_L_between: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", np.asarray(self.anchor))
        object.__setattr__(self, "anchor_test", np.asarray(self.anchor_test))
        object.__setattr__(self, "anchor_y", np.asarray(self.anchor_y))
        object.__setattr__(self, "anchor_y_test", np.asarray(self.anchor_y_test))
        object.__setattr__(self, "Xs_train_inter", _ensure_array_list(self.Xs_train_inter))
        object.__setattr__(self, "Xs_test_inter", _ensure_array_list(self.Xs_test_inter))
        object.__setattr__(self, "anchors_inter", _ensure_array_list(self.anchors_inter))
        object.__setattr__(self, "anchors_test_inter", _ensure_array_list(self.anchors_test_inter))
        if self.L_within is not None:
            object.__setattr__(self, "L_within", np.asarray(self.L_within))
        if self.L_between is not None:
            object.__setattr__(self, "L_between", np.asarray(self.L_between))
        if self.graph_adjacency is not None:
            object.__setattr__(self, "graph_adjacency", np.asarray(self.graph_adjacency))
        if self.graph_L_within is not None:
            object.__setattr__(self, "graph_L_within", np.asarray(self.graph_L_within))
        if self.graph_L_between is not None:
            object.__setattr__(self, "graph_L_between", np.asarray(self.graph_L_between))


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
    # Aggregated / convenience views for analysis & visualization
    # - X_integ: stacked integrated training representations across institutions
    # - anchor_integ: one representative integrated anchor set (e.g. first institution)
    # - anchor_integ_y: labels for anchor_integ
    X_integ: Optional[np.ndarray] = None
    anchor_integ: Optional[np.ndarray] = None
    anchor_integ_y: Optional[np.ndarray] = None
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
        if self.X_integ is not None:
            object.__setattr__(self, "X_integ", np.asarray(self.X_integ))
        if self.anchor_integ is not None:
            object.__setattr__(self, "anchor_integ", np.asarray(self.anchor_integ))
        if self.anchor_integ_y is not None:
            object.__setattr__(self, "anchor_integ_y", np.asarray(self.anchor_integ_y))
