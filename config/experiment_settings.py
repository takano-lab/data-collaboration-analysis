from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import yaml

_YAML_PATH = Path(__file__).with_suffix(".yaml")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        # デフォルトYAMLが無い場合でも動けるように空dictで返す
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_from_data(data: dict) -> dict:
    param_grid = OrderedDict(data.get("param_grid", {}))
    defaults = dict(data.get("defaults", {}))
    or_groups = list(data.get("or_groups", []))
    or_group_key_set = {key for group in or_groups for key in group}
    return {
        "PARAM_GRID": param_grid,
        "ERROR_SKIP": bool(data.get("error_skip", False)),
        "PARAM_COLUMNS": list(data.get("param_columns", [])),
        "DF_COLUMNS": list(data.get("df_columns", [])),
        "INTERMEDIATE_COLUMNS": list(data.get("intermediate_columns", [])),
        "PLOT_COLUMNS": list(data.get("plot_columns", [])),
        "MEAN_PARAM": data.get("mean_param", "seed_values"),
        "DEFAULTS": defaults,
        "OR_GROUPS": or_groups,
        "OR_GROUP_KEY_SET": or_group_key_set,
        "_DATASET_DEFAULTS": dict(data.get("dataset_defaults", {})),
        "RULES": list(data.get("rules", [])),
    }


def load_settings(yaml_path: Path | str | None = None) -> dict:
    """
    YAML を読み込み、main.py 側が使う定数セットを返す。
    """
    path = Path(yaml_path) if yaml_path else _YAML_PATH
    data = _load_yaml(path)
    return _build_from_data(data)


_DATA = _load_yaml(_YAML_PATH)
_SETTINGS = _build_from_data(_DATA)

# 1) 全探索したいパラメータ（config.◯◯に代入）
PARAM_GRID: Dict[str, List[Any]] = _SETTINGS["PARAM_GRID"]

# 2-1) 実行失敗時の挙動（True でスキップ、False で例外をそのまま投げる）
ERROR_SKIP: bool = _SETTINGS["ERROR_SKIP"]

# 3) DataFrameに保持したい「パラメータ列」（順序もこの通り）
PARAM_COLUMNS: List[str] = _SETTINGS["PARAM_COLUMNS"]

# 3-2) train/test_df のDataFrameに保持する名前
DF_COLUMNS: List[str] = _SETTINGS["DF_COLUMNS"]

# 3-3) 中間表現のDataFrameに保持する名前
INTERMEDIATE_COLUMNS: List[str] = _SETTINGS["INTERMEDIATE_COLUMNS"]

# plot ファイル名を組み立てる際に使用するカラム
PLOT_COLUMNS: List[str] = _SETTINGS["PLOT_COLUMNS"]

# 平均を取る対象のパラメータ
MEAN_PARAM: str = _SETTINGS["MEAN_PARAM"]

# 4) 条件ルール
#    - LOCK: 条件一致時に指定パラメータを固定（そのキーは“ループしない”）
#    - SKIP: 条件一致の組合せを丸ごとスキップ
DEFAULTS: Dict[str, Any] = _SETTINGS["DEFAULTS"]

OR_GROUPS: List[List[str]] = _SETTINGS["OR_GROUPS"]

OR_GROUP_KEY_SET = _SETTINGS["OR_GROUP_KEY_SET"]

_DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = _SETTINGS["_DATASET_DEFAULTS"]

RULES: List[Dict[str, Any]] = _SETTINGS["RULES"]

__all__ = [
    "PARAM_GRID",
    "ERROR_SKIP",
    "PARAM_COLUMNS",
    "DF_COLUMNS",
    "INTERMEDIATE_COLUMNS",
    "PLOT_COLUMNS",
    "MEAN_PARAM",
    "DEFAULTS",
    "OR_GROUPS",
    "OR_GROUP_KEY_SET",
    "_DATASET_DEFAULTS",
    "RULES",
    "load_settings",
]
