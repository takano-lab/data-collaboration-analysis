from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.paths import OUTPUT_DIR

logger = Any


class DataPreservationManager:
    """Handles persistence/loading of institution and intermediate datasets."""

    def __init__(self, config, logger: logger | None = None) -> None:
        self.config = config
        self.logger = logger

    # ------------------------------------------------------------------ #
    # generic helpers
    # ------------------------------------------------------------------ #
    def _preserved_root(self) -> Path:
        return OUTPUT_DIR / "preserved_df"

    @staticmethod
    def _safe_name(name: Optional[str]) -> str:
        if not name:
            return "unnamed"
        cleaned = []
        for ch in str(name):
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        slug = "".join(cleaned)
        slug = "_".join(filter(None, slug.split("_")))
        return slug or "unnamed"

    def _preserved_path(self, category: str, name: Optional[str]) -> Path:
        safe = self._safe_name(name)
        base = self._preserved_root() / category
        return base / f"{safe}.pkl"

    @staticmethod
    def _bundle_to_dataframe(bundle: Dict[str, Sequence[object]]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for key, arrays in bundle.items():
            stored: list[object] = []
            for arr in arrays:
                if isinstance(arr, (pd.DataFrame, pd.Series)):
                    stored.append(arr.copy(deep=True))
                else:
                    stored.append(np.asarray(arr))
            rows.append({"part": key, "values": stored})
        return pd.DataFrame(rows, columns=["part", "values"])

    @staticmethod
    def _dataframe_to_bundle(df: pd.DataFrame) -> Dict[str, list[object]]:
        bundle: Dict[str, list[object]] = {}
        for _, row in df.iterrows():
            key = str(row.get("part"))
            raw_vals = row.get("values", [])
            arrays: list[object] = []
            try:
                for arr in raw_vals:
                    if isinstance(arr, (pd.DataFrame, pd.Series)):
                        arrays.append(arr.copy(deep=True))
                    else:
                        arrays.append(np.asarray(arr))
            except Exception:
                arrays = []
            bundle[key] = arrays
        return bundle

    # ------------------------------------------------------------------ #
    # Persistence primitives
    # ------------------------------------------------------------------ #
    def load_bundle(self, category: str, name: Optional[str]) -> Optional[Dict[str, list[object]]]:
        path = self._preserved_path(category, name)
        if not path.exists():
            return None
        try:
            df = pd.read_pickle(path)
        except Exception as exc:
            self._log(f"[preserved] failed to load {path}: {exc}", level="warning")
            return None
        if not isinstance(df, pd.DataFrame):
            return None
        return self._dataframe_to_bundle(df)

    def save_bundle(self, category: str, name: Optional[str], bundle: Dict[str, Sequence[object]]) -> None:
        if not bundle:
            return
        path = self._preserved_path(category, name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            df = self._bundle_to_dataframe(bundle)
            df.to_pickle(path)
        except Exception as exc:
            self._log(f"[preserved] failed to save {path}: {exc}", level="warning")

    # ------------------------------------------------------------------ #
    # Convenience utilities
    # ------------------------------------------------------------------ #
    def save_artifacts(
        self,
        obj: Any,
        *,
        save_dir: Optional[str] = None,
        items: Optional[Sequence[str]] = None,
        filename_suffix: Optional[str] = None,
    ) -> dict:
        """
        Persist selected numpy arrays/dataframes from the builder object as CSV files.
        """
        out: dict = {}
        default_dir = Path(getattr(self.config, "output_path", ".")) / "dataframe"
        base = Path(save_dir) if save_dir is not None else default_dir
        base.mkdir(parents=True, exist_ok=True)

        if filename_suffix is None:
            df_name = getattr(self.config, "df_name", None)
            if df_name:
                filename_suffix = self._safe_name(df_name)
        if filename_suffix:
            filename_suffix = f"_{filename_suffix}"
        else:
            filename_suffix = ""

        available_items = items or [
            "train_df",
            "test_df",
            "anchor",
            "anchor_test",
            "anchors_inter",
            "anchors_test_inter",
            "Xs_train_inter",
            "Xs_test_inter",
            "X_train_integ",
            "X_test_integ",
        ]

        def _arr_to_df(arr: np.ndarray, *, add_cols: dict | None = None, col_prefix: str = "dim") -> pd.DataFrame:
            df = pd.DataFrame(arr, columns=[f"{col_prefix}{i}" for i in range(arr.shape[1])])
            if add_cols:
                for k, v in add_cols.items():
                    df[k] = v
            return df

        def _lists_to_df(lst: list[np.ndarray], *, add_cols_each: list[dict] | None = None, col_prefix: str = "dim") -> pd.DataFrame:
            dfs = []
            add_cols_each = add_cols_each or [{} for _ in lst]
            for arr, extra in zip(lst, add_cols_each):
                dfs.append(_arr_to_df(arr, add_cols=extra, col_prefix=col_prefix))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        for item in available_items:
            if not hasattr(obj, item):
                continue
            value = getattr(obj, item)
            if value is None:
                continue
            path = base / f"{item}{filename_suffix}.csv"
            try:
                if isinstance(value, pd.DataFrame):
                    value.to_csv(path, index=False)
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    add_cols = [{"institution": idx} for idx in range(len(value))]
                    _lists_to_df(value, add_cols_each=add_cols).to_csv(path, index=False)
                elif isinstance(value, np.ndarray):
                    _arr_to_df(value).to_csv(path, index=False)
                else:
                    continue
                out[item] = str(path)
            except Exception as exc:
                self._log(f"[WARN] failed to save artifact {item}: {exc}", level="warning")
        return out

    # ------------------------------------------------------------------ #
    def _log(self, msg: str, *, level: str = "info") -> None:
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
