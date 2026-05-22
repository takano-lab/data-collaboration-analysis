from __future__ import annotations

from typing import Any

import pandas as pd


REGRESSION_DATASETS = {
    "housing",
    "diabetes",
    "slice_localization",
    "ct_slice_localization",
    "ujiindoorloc",
    "ujiindoorloc_longitude",
    "ujiindoorloc_latitude",
    "blog",
    "blogfeedback",
}

REGRESSION_METRICS = {"rmse", "r2", "mae", "mse"}
CLASSIFICATION_METRICS = {"accuracy", "auc", "f1", "precision", "recall"}
REGRESSION_MODELS = {"linear_regression", "ridge", "lasso", "random_forest_regressor"}


def _task_value(raw: Any) -> str | None:
    if raw is None or raw is False:
        return None
    if isinstance(raw, (list, tuple, set)):
        if not raw:
            return None
        raw = next(iter(raw))
    value = str(raw).strip().lower()
    if value in {"", "none", "undefined"}:
        return None
    aliases = {
        "reg": "regression",
        "regressor": "regression",
        "continuous": "regression",
        "class": "classification",
        "clf": "classification",
        "classifier": "classification",
    }
    return aliases.get(value, value)


def resolve_task(config: Any, df: pd.DataFrame | None = None, *, update_config: bool = True) -> str:
    """
    Resolve task type for the current run.

    Priority:
      1. config.task / config.problem_type if explicitly set
      2. known dataset defaults
      3. metric/model hints
      4. numeric target heuristic
      5. classification fallback
    """
    explicit = _task_value(getattr(config, "task", None))
    if explicit is None:
        explicit = _task_value(getattr(config, "problem_type", None))
    if explicit in {"regression", "classification"}:
        task = explicit
    else:
        dataset = str(getattr(config, "dataset", "") or "").lower()
        metric = str(getattr(config, "metrics", "") or "").lower()
        h_model = str(getattr(config, "h_model", "") or "").lower()

        if dataset in REGRESSION_DATASETS:
            task = "regression"
        elif metric in REGRESSION_METRICS or h_model in REGRESSION_MODELS:
            task = "regression"
        elif metric in CLASSIFICATION_METRICS:
            task = "classification"
        elif df is not None and "target" in df.columns:
            target = df["target"]
            if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > max(20, len(target) // 20):
                task = "regression"
            else:
                task = "classification"
        else:
            task = "classification"

    if update_config:
        config.task = task
    return task


def is_regression_task(config: Any, df: pd.DataFrame | None = None, *, update_config: bool = True) -> bool:
    return resolve_task(config, df, update_config=update_config) == "regression"
