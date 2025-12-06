from __future__ import annotations

import argparse
import statistics
from collections import OrderedDict

# runners/runner_grid.py
from itertools import product
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from typing import Any, Dict, List, Sequence

import pandas as pd

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from experiments.experiment import run_once  # ←元main改名
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

# ====== ユーザーが編集するのはここだけ ======

# 1) 全探索したいパラメータ（config.◯◯に代入）

PARAM_GRID: Dict[str, List[Any]] = OrderedDict({
    "dataset": [
    "fashion_mnist",
    #"mnist",
    #"mnist_1248",
    #"mice",
    #"statlog",
    #"qsar",
    #"digits",
    #"breast_cancer",
    #"adult",
    # "glass", 
    # "seeds", 
    # "letter_recognition",
    # "wine_quality",
    #"har",
    #"cifar10",
    # "diabetes130",
    # "bank_marketing",
    # "cifar10_800",
    #'3D_gaussian_clusters',
    #"concentric_three_circles",
    #"iris",
    # "ecoli",
    # "vowel"
    # "coil20"
],# + ["medmnist_{}".format(i) for i in [3, 5, 6, 7]],
    "svd_ratio":[0, 0.5, 1], #, 0.25, 0.5, 0.75, 1
    #"wine_quality", "glass", "seeds", "letter_recognition"],#"wine_quality", #"qsar","mice", "statlog", "breast_cancer", "adult", "digits",],     # 例: ["qsar","mice"]
    "h_model": ["mlp"],             # 例: ["mlp","random_forest"] svm_linear_classifier
    "F_type": ["svd"], # "svd", "kernel_pca_self_tuning", "kernel_pca_svd_mixed" "kernel_pca", "lpp" # "kernel_pca_self_tuning" "kernel_pca_svd_mixed",
    "G_type": ["laplacian_nonlinear"],#, laplacian_nonlinear"graph_nonlinear", "nonlinear" "imakura", "odc", "gep", "fl", "centralize", "individual"#,"graph_nonlinear", "nonlinear" "graph_nonlinear_maximize",  "graph_nonlinear_x", "graph_nonlinear_x_maximize", "kernel_gep", "kernel_graph_gep",  "kernel_graph_gep_maximize", 
    "gamma_type": ["fixed"], # "X_tuning", "y_tuning", "fixed"  # 例: ["X_tuning","y_tuning"] # "individual",
    "gamma_ratio_krr": [1], #, 0.1, 0.3, 3, 10 
    "graph_knn_k": [10],
    "graph_mu_align": [1], # 0.5, 1, 3, 10, 30
    "graph_lambda_rkhs": [1],
    "graph_stability_eps": [1],
    "num_anchor_data": [1000],
    "nl_lambda": [1], # LOCKで止められる, 0.00001
    "lw_alpha": [0],
    "metrics": ["accuracy"], #"accuracy"
    "kernel_type": ["rbf"],
    "visualize": [False],
    #"feature_num": [2],
    "dim_intermediate": [10],#[20, 10, 5, 2], 6
    #"dim_integrate": [10],#[20, 10, 5, 2], 6
    "num_institution_user": [100],
    "num_institution": [10],
    "K_normalization":[False],
    "anchor_method":["smote"], #gaussian smote 
    "smote_ratio":[1],#, 0, 0.1, 0.25, 0.5, 1],
    "data_distribution":["bias"],# "division" "even"
    "inter_normalization":[True],
    "evaluate_integrate_metrics":[True],
    "load_df_data":[False],
    "load_intermediate_data":[False],
    "preserve_integrated_data":[False],
    "umap_neighbors":[10],
    "bias_ratio":[0.1], # 0.1, 0.4, 0.6, 0.95, 0.8 
    "max_dim":[10000],
    "zerosum":[True],
    "anchor_label_max_dist": [100000],
    "seed_values":[[i for i in range(0, 1)]],  # 例: [[1,2,3,4,5]
    })

# "gamma_ratio"
# "smote_ratio"
# "bias_ratio":[0.9],

# 2-1) 実行失敗時の挙動（True でスキップ、False で例外をそのまま投げる）
ERROR_SKIP = True

# 3) DataFrameに保持したい「パラメータ列」（順序もこの通り）
PARAM_COLUMNS: List[str] = [
    "dataset", "h_model", "F_type", "G_type", "gamma_ratio", "gamma_type", "gamma_ratio_krr", "num_anchor_data", "nl_lambda", "lw_alpha", "metrics", "dim_intermediate", "dim_integrate", "num_institution_user", "num_institution", "anchor_method", "smote_ratio", "data_distribution", "inter_normalization", "evaluate_integrate_metrics", "umap_neighbors", "bias_ratio", "svd_ratio", "zerosum", "kernel_type"
]

# 3-2) train/test_df のDataFrameに保持する名前
DF_COLUMNS: List[str] = [
    "dataset", "data_distribution", "num_institution_user", "num_institution", "seed_values", "bias_ratio"
]

# 3-3) 中間表現のDataFrameに保持する名前
INTERMEDIATE_COLUMNS: List[str] = [
    "dataset", "F_type", "gamma_ratio", "data_distribution", "dim_intermediate", "num_institution_user", "num_institution", "seed_values", "umap_neighbors", "bias_ratio", "anchor_method", "smote_ratio", "num_anchor_data","max_dim", "svd_ratio", "anchor_label_max_dist"
]

# 4) 条件ルール
#    - LOCK: 条件一致時に指定パラメータを固定（そのキーは“ループしない”）
#    - SKIP: 条件一致の組合せを丸ごとスキップ
DEFAULTS = {
    "y_name": "target",
    "nl_lambda": 0.1,
    "gamma_ratio": 1,
    "gamma_ratio_krr": 1,
    #"num_institution_user": 50,
    "visualize": True,
    "feature_num": None,
    "dim_intermediate": None,
    #"dim_integrate": 3, ######
    "num_institution": None,
    "lambda_gen_eigen": 0,
    "orth_ver": False,
    "K_normalization":True,
    "inter_normalization":True,
    "lw_alpha":False,
    "smote_ratio":1.0,
    "umap_metric":"random", # euclidean correlation cosine
    "graph_knn_k": None,
    "graph_mu_align": 1.0,
    "graph_lambda_rkhs": 1e-2,
    "graph_stability_eps": 1e-6,
}

DEFAULT_SMOTE = 1.0
DEFAULT_BIAS = 0.8
NON_DEFAULT_SMOTE = [0, 0.1, 0.25, 0.5]          # 1.0 以外
NON_DEFAULT_BIAS = [0.8, 0.4, 0.6, 0.95]         # 0.8 以外
NON_DEFAULT_GAMMA_KRR = [0.1, 0.3, 3, 10]        # 1 以外
# --- 追加: dataset ごとのデフォルト適用（定数のみ。動的は未設定）---
_DATASET_DEFAULTS = {
    #"qsar":                 {"feature_num": 41},#, "dim_intermediate": 37, "dim_integrate": 37, "num_institution_user": 25, "num_institution": 20},
    #"adult":                {"feature_num": 51},#, "dim_intermediate": 50, "dim_integrate": 50, "num_institution_user": 150, "num_institution": 10},
    #"diabetes130":          {"feature_num": 200},#, "dim_intermediate": 100, "dim_integrate": 100, "num_institution_user": 500, "num_institution": 10},
    #"mice":                 {"feature_num": 77},#, "dim_intermediate": 46, "dim_integrate": 46, "num_institution_user": 50, "num_institution": 5},
    #"breast_cancer":        {"feature_num": 15},#, "num_institution_user": 60},
    #"digits":               {"dim_intermediate": 15, "dim_integrate": 15, "num_institution_user": 100, "num_institution": 10},
    # "mnist":                {"dim_intermediate": 10, "dim_integrate": 10, "num_institution_user": 50, "num_institution": 10},
    #"mnist":                {"dim_intermediate": 20, "dim_integrate": 20, "num_institution_user": 100, "num_institution": 10, "data_distribution": "division", "gamma_ratio":1, "gamma_ratio_krr":1, "umap_metric":"correlation"},
    #"mnist_1248":           {"dim_intermediate": 20, "dim_integrate": 20, "num_institution_user": 100, "num_institution": 10, "data_distribution": "division", "gamma_ratio":1, "gamma_ratio_krr":1, "umap_metric":"correlation"},
    #"fashion_mnist":        {"dim_intermediate": 20, "dim_integrate": 20, "num_institution_user": 100, "num_institution": 10, "umap_metric":"correlation"},
    #"concentric_circles":   {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 500, "num_institution": 2},
    #"concentric_three_circles": {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 500, "num_institution": 2},
    #"two_gaussian_distributions": {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 50, "num_institution": 5},
    #"3D_gaussian_clusters": {"feature_num": 3, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
    #"3D_8_gaussian_clusters": {"feature_num": 3, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
    #"digits_":             {"dim_intermediate": 4, "dim_integrate": 4, "num_institution": 10, "num_institution_user": 100},
    #"digits_v2":           {"dim_intermediate": 30, "dim_integrate": 30, "num_institution": 29, "num_institution_user": 30},
    #"housing":             {"num_institution": 10, "num_institution_user": 10},
    #"statlog":             {"num_institution_user": 200},
    #"wine_quality":       {"feature_num": 11},#, "dim_intermediate": 8},
    #"glass":              {"feature_num": 9},#,  "dim_intermediate": 6},
    #"seeds":              {"feature_num": 7},#,  "dim_intermediate": 5},
    #"letter_recognition": {"feature_num": 16, "num_institution_user": 20},#, "dim_intermediate": 12},
    #"iris":               {"dim_intermediate": 3},
    #"ecoli":              {"dim_intermediate": 5},
    #"vowel":              {"dim_intermediate": 4},
    #"cifar10":              {"dim_intermediate": 20, "dim_integrate": 20, "num_institution_user":100, "data_distribution": "division", "gamma_ratio":0.0001, "gamma_ratio_krr":1, "F_type": "kernel_pca_gamma_fixed"},
}
    
RULES: List[Dict[str, Any]] = [
    {"type": "LOCK", "when": {"G_type": ["centralize", "individual"]}, "lock": {"gamma_ratio": DEFAULTS["gamma_ratio"]}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "imakura", "gep", "gep2",  "odc",]}, "lock": {"nl_lambda": DEFAULTS["nl_lambda"]}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "fl", "imakura", "gep", "gep2",  "odc",]}, "lock": {"gamma_ratio_krr": DEFAULTS["gamma_ratio_krr"]}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "fl", "imakura", "gep", "gep2",  "odc"]}, "lock": {"graph_mu_align": 0, "graph_lambda_rkhs": 0, "graph_stability_eps": 0}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "fl", "imakura", "gep", "gep2",  "odc"]}, "lock": {"graph_knn_k": None}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "fl", "imakura", "gep", "gep2",  "odc"]}, "lock": {"zerosum": False}},
    {"type": "LOCK", "when": {"G_type": ["graph_nonlinear"]}, "lock": {"graph_knn_k":100000}},
    #{"type": "LOCK", "when": {"G_type": ["nonlinear"]}, "lock": {"inter_normalization": False}},
    {"type": "LOCK", "when": {"G_type": ["graph_nonlinear_x_maximize"]}, "lock": {"graph_mu_align": 0.5}},
    {"type": "LOCK", "when": {"G_type": ["graph_nonlinear",  "kernel_graph_gep", "kernel_gep", "gep", "imakura", "odc", "individual", "centralize",  "fl"]}, "lock": {"lw_alpha":  DEFAULTS["lw_alpha"]}},
    # F_type による固定: ae のとき比率指定と gamma_type=median
    #{"type": "LOCK", "when": {"F_type": ["ae"]}, "lock": {"dim_intermediate": "*0.8", "dim_integrate": "*0.8", "gamma_type": "median"}},
    # F_type による固定: umap のとき固定次元と gamma_type=fixed
    #{"type": "LOCK", "when": {"F_type": ["umap"]}, "lock": {"dim_intermediate": 6, "dim_integrate": 6, "gamma_type": "fixed"}},
    #{"type": "SKIP", "when": {"F_type": ["kernel_pca"], "G_type": ["GEP_weighted"]}},
    #{"type": "SKIP", "when": {"gamma_ratio_krr": NON_DEFAULT_GAMMA_KRR, "smote_ratio": NON_DEFAULT_SMOTE}},
    #{"type": "SKIP", "when": {"gamma_ratio_krr": NON_DEFAULT_GAMMA_KRR, "bias_ratio": NON_DEFAULT_BIAS}},
    #{"type": "SKIP", "when": {"smote_ratio": NON_DEFAULT_SMOTE, "bias_ratio": NON_DEFAULT_BIAS}},
]
# ============================================


def _generate_unique_combos(grid: Dict[str, List[Any]]):
    """
    通常グリッド生成。空リストは DEFAULTS にフォールバック、DEFAULTS も無ければ product 対象外。
    """
    pairs: list[tuple[str, List[Any]]] = []
    for k in grid.keys():
        vals = grid.get(k, [])
        if not vals:
            if (k in DEFAULTS) and (DEFAULTS[k] is not None):
                vals = [DEFAULTS[k]]
            else:
                continue
        pairs.append((k, vals))

    if not pairs:
        yield {}
        return

    seen = set()
    for tup in product(*(vals for _, vals in pairs)):
        base = {k: v for (k, _), v in zip(pairs, tup)}
        after = _apply_lock_rules(base)
        if _skip_by_rules(after):
            continue
        norm = tuple((k, _make_hashable(after.get(k))) for k, _ in pairs)
        if norm in seen:
            continue
        seen.add(norm)
        yield after
            
def _match(cond: Dict[str, List[Any]], combo: Dict[str, Any]) -> bool:
    return all(k in combo and combo[k] in vals for k, vals in cond.items())

def _apply_lock_rules(combo: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(combo)
    for r in RULES:
        if r.get("type") == "LOCK" and _match(r.get("when", {}), out):
            out.update(r.get("lock", {}))
    return out

def _skip_by_rules(combo: Dict[str, Any]) -> bool:
    for r in RULES:
        if r.get("type") == "SKIP" and _match(r.get("when", {}), combo):
            return True
    return False

def _apply_dataset_defaults(cfg: Config, dataset: str) -> None:
    d = _DATASET_DEFAULTS.get(dataset, {})
    for k, v in d.items():
        cur = getattr(cfg, k, None)
        if cur is None or (isinstance(cur, (int, float)) and cur <= 0):
            setattr(cfg, k, v)

def _is_empty(v) -> bool:
    return (
        v is None or
        (isinstance(v, (int, float)) and v < 0) or
        (isinstance(v, str) and v.strip().lower() in ("", "undefined", "none"))
    )

def _apply_defaults(cfg: Config, dataset: str, combo: dict | None = None) -> None:
    """
    優先順位:
      1) ユーザ指定（PARAM_GRIDで明示）→ 上書きしない
      2) _DATASET_DEFAULTS（優先して適用）
      3) DEFAULTS（残りを埋める）
    """ 
    # 2) dataset固有（ユーザ明示は尊重）
    ds = _DATASET_DEFAULTS.get(dataset, {})
    for k, v in ds.items():
        if combo and (k in combo):
            continue
        cur = getattr(cfg, k, None)
        if _is_empty(cur) and not _is_empty(v):
            setattr(cfg, k, v)

    # 3) グローバル既定（残りのみ、undefined/None は適用しない）
    for k, v in DEFAULTS.items():
        cur = getattr(cfg, k, None)
        if _is_empty(cur) and not _is_empty(v):
            setattr(cfg, k, v)

def _make_hashable(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((k, _make_hashable(val)) for k, val in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_make_hashable(x) for x in v)
    if isinstance(v, set):
        return tuple(sorted(_make_hashable(x) for x in v))
    return v

def _sanitize_for_identifier(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        value_str = "true" if value else "false"
    elif isinstance(value, float):
        if value.is_integer():
            value_str = str(int(value))
        else:
            value_str = f"{value}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    else:
        value_str = str(value)
    value_str = value_str.strip()
    if not value_str:
        return "none"
    value_str = value_str.replace(" ", "_")
    cleaned = []
    for ch in value_str:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    normalized = "".join(cleaned)
    normalized = "_".join(filter(None, normalized.split("_")))
    return normalized.lower() or "none"

def _build_identifier(columns: Sequence[str], cfg: Config) -> str:
    parts: list[str] = []
    for col in columns:
        val = getattr(cfg, col, None)
        parts.append(f"{col}_{_sanitize_for_identifier(val)}")
    return "_".join(parts)

def _generate_unique_combos(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    seen = set()
    for tup in product(*(grid[k] for k in keys)):
        base = {k: v for k, v in zip(keys, tup)}
        after = _apply_lock_rules(base)
        if _skip_by_rules(after):
            continue
        norm = tuple((k, _make_hashable(after.get(k))) for k in keys)
        if norm in seen:
            continue
        seen.add(norm)
        yield after

def _set_config_from_combo(cfg: Config, combo: Dict[str, Any]) -> None:
    """dataset/metrics/visualize以外は config に流し込む。 ???"""
    for k, v in combo.items():
        if k in ("dataset", "metrics", "seed_values", "seeds"):
            continue
        setattr(cfg, k, v)
    # True_F_type を常に同期
    if hasattr(cfg, "F_type"):
        cfg.True_F_type = cfg.F_type

def run_grid(
    config: Config,
    grid: Dict[str, List[Any]] | None = None,
    logger_=None,
) -> pd.DataFrame:
    rows = []
    all_columns = PARAM_COLUMNS + [
        "score_mean", "score_stdev",
        # 新しい集計カラム（seed ループの平均）
        "lni_inter_test", "lni_integ_test", "integ_metrics_train", "integ_metrics_test",
    ]
    # 追加: 注入値の優先適用
    grid = grid or PARAM_GRID
    log = logger_ if logger_ is not None else getLogger(__name__)

    base_paths = dict(output_path=config.output_path, input_path=INPUT_DIR)
    combos_iter = _generate_unique_combos(grid)

    for combo in combos_iter:
        dataset = combo["dataset"]
        metrics_name = combo["metrics"]
        cfg = Config(**base_paths)
        vals = []
        log.info(f"[pattern] { {k: combo[k] for k in PARAM_COLUMNS if k in combo} }")

        # ループ内で集計するメトリクスの一時リスト
        lni_inter_vals = []
        lni_integ_vals = []
        integ_train_vals = []
        integ_test_vals = []

        seeds_raw = None
        for key in ("seed_values", "seeds"):
            if key in combo:
                seeds_raw = combo[key]
                break

        seeds_list: list[int]
        if seeds_raw is not None:
            if isinstance(seeds_raw, (int, float, str)):
                seeds_list = [int(seeds_raw)]
            else:
                try:
                    seeds_list = [int(s) for s in seeds_raw]
                except TypeError:
                    seeds_list = [int(seeds_raw)]
        else:
            seeds_list = [0]

        if not seeds_list:
            seeds_list = [0]


        for i in seeds_list:
            seed_value = int(i)
            cfg.seed = seed_value
            cfg.dataset = dataset
            cfg.metrics = metrics_name
            cfg.plot_name = f"{dataset}_{combo.get('F_type','-')}_{combo.get('G_type','-')}_graph_knn_k_{combo.get('graph_knn_k','-')}_graph_mu_align_{combo.get('graph_mu_align','-')}_{combo.get('graph_lambda_rkhs','-')}_{combo.get('zerosum','-')}.png"
            _set_config_from_combo(cfg, combo)
            _apply_defaults(cfg, dataset, combo)
            cfg.seed_values = seed_value
            cfg.seeds = seed_value
            cfg.df_name = _build_identifier(DF_COLUMNS, cfg)
            cfg.intermediate_name = _build_identifier(INTERMEDIATE_COLUMNS, cfg)
            # identifier for integrated artifacts
            cfg.integrated_name = _build_identifier(PARAM_COLUMNS, cfg)
            def _run_and_collect() -> float:
                val = run_once(cfg, log)
                vals.append(float(val))
                record_config_to_cfg(cfg)
                record_value_to_cfg(cfg, "評価値", val)
                for key, value in cfg.__dict__.items():
                    print(f"{key} = {value}")

                # ループごとのメトリクスを収集（存在すれば）
                try:
                    v = getattr(cfg, "lni_inter_test", None)
                    if v is not None:
                        lni_inter_vals.append(float(v))
                except Exception:
                    pass
                try:
                    v = getattr(cfg, "lni_integ_test", None)
                    if v is not None:
                        lni_integ_vals.append(float(v))
                except Exception:
                    pass
                try:
                    v = getattr(cfg, "integ_metrics_train", None)
                    if v is not None:
                        integ_train_vals.append(float(v))
                except Exception:
                    pass
                try:
                    v = getattr(cfg, "integ_metrics_test", None)
                    if v is not None:
                        integ_test_vals.append(float(v))
                except Exception:
                    pass
                return float(val)

            if ERROR_SKIP:
                try:
                    _run_and_collect()
                except Exception as e:
                    msg = f"[skip] seed={i}, dataset={dataset}, G_type={combo.get('G_type')}, reason={e}"
                    log.info(msg)
                    try:
                        log.exception(msg)
                    except Exception:
                        pass
                    continue
            else:
                _run_and_collect()

        mean_val = sum(vals) / len(vals) if vals else 0.0
        stdev_val = statistics.stdev(vals) if len(vals) > 1 else 0.0

        row = {k: combo.get(k, None) for k in PARAM_COLUMNS}
        row.update({
            "score_mean": mean_val,
            "score_stdev": stdev_val,
        })
        # seed ループで収集したメトリクスの平均を記録（有限値のみ平均）
        def _mean_finite(xs: list[float]) -> float:
            import math
            vals_ = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
            return (sum(vals_) / len(vals_)) if vals_ else 0.0

        row.update({
            "lni_inter_test": _mean_finite(lni_inter_vals),
            "lni_integ_test": _mean_finite(lni_integ_vals),
            "integ_metrics_train": _mean_finite(integ_train_vals),
            "integ_metrics_test": _mean_finite(integ_test_vals),
        })

        out_path = cfg.output_path / f"result_grid_{dataset}.csv"
        one = pd.DataFrame([row], columns=all_columns)
        header_needed = not out_path.exists()
        one.to_csv(out_path, mode="a", header=header_needed, index=False, encoding="utf-8-sig")
        log.info(f"[saved] {out_path}")

        rows.append(row)

    df_all = pd.DataFrame(rows, columns=all_columns)
    return df_all

# run.py
from config.config import Config
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

if __name__ == "__main__":
    # 引数処理はここだけ（デフォルトは 0912）
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="11095_gep_umap")
    args = parser.parse_args()

    # 出力先を決定
    output_path = OUTPUT_DIR / args.run_name

    # Config/Logger をここでだけ作成
    config = Config(output_path=output_path, input_path=INPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    logger.handlers.clear()  # reset handlers

    file_handler = FileHandler(filename=config.output_path / "result.log", encoding="utf-8")
    file_handler.setLevel(INFO)
    logger.addHandler(file_handler)

    console_handler = StreamHandler()
    console_handler.setLevel(INFO)
    console_handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 実行
    df = run_grid(config, logger_=logger)
    df.to_csv(config.output_path / "result_grid_all.csv", index=False, encoding="utf-8-sig")
