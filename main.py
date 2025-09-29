from __future__ import annotations
from itertools import product
from itertools import chain
import argparse
from logging import INFO, FileHandler, getLogger
import statistics
import pandas as pd
from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

# runners/runner_grid.py
from itertools import product
from collections import OrderedDict
import statistics
import pandas as pd
from typing import Any, Dict, List
from experiments.experiment import run_once  # ←元main改名
from config.config import Config
# ====== ユーザーが編集するのはここだけ ======

# 1) 全探索したいパラメータ（config.◯◯に代入）
PARAM_GRID: Dict[str, List[Any]] = OrderedDict({
    "dataset": [
    #"mice",
    #"statlog",
    #"qsar",
    #"breast_cancer",
    #"adult",
    #"digits",
    #"glass", "seeds", "letter_recognition",
    #"wine_quality",
    #"har",
    #"diabetes130",
    #"bank_marketing",
    #"mnist",
    "mnist_1248",
    #"fashion_mnist",
    #'3D_gaussian_clusters',
    #"concentric_three_circles",
    #"iris",
    #"ecoli",
    #"vowel"
],#"wine_quality", "glass", "seeds", "letter_recognition"],#"wine_quality", #"qsar","mice", "statlog", "breast_cancer", "adult", "digits",],     # 例: ["qsar","mice"]
    "h_model": ["svm_linear_classifier"],             # 例: ["mlp","random_forest"] svm_linear_classifier
    "F_type": ["kernel_pca_self_tuning",], # "svd", "kernel_pca_self_tuning", "kernel_pca_svd_mixed" "kernel_pca", "lpp" # "kernel_pca_self_tuning" "kernel_pca_svd_mixed",
    "G_type": ["Imakura", "nonlinear"], # 'centralize', "individual", "Imakura", "GEP",  "ODC" # 'centralize', "individual",
    #"gamma_ratio": [1],#[0.1, 0.3, 1, 3, 10],             # 例: [0.1,1,5]
    "gamma_type": ["X_tuning"], # "X_tuning", "y_tuning", "fixed"  # 例: ["X_tuning","y_tuning"]
    "gamma_ratio_krr": [1],
    "num_anchor_data": [1000],
    "nl_lambda": [1],        # LOCKで止められる, 0.00001
    "lw_alpha": [0.3],
    "lambda_pred": [0],
    "lambda_offdiag": [0],
    "metrics": ["auc"],
    "visualize": [False],
    #"feature_num": [41],
    "dim_intermediate": [30],#[20, 10, 5, 2],
    "num_institution_user": [200],#[50, 100, 200, 400],
    "num_institution": [3],
    "K_normalization":[False],
    "anchor_method":["smote"], #
})

# 2) ループ回数（seed を 0..loop_num-1 で回します）
LOOP_NUM = 1

# 3) DataFrameに保持したい「パラメータ列」（順序もこの通り）
PARAM_COLUMNS: List[str] = [
    "dataset", "h_model", "F_type", "G_type", "gamma_type", "gamma_ratio", "gamma_ratio_krr",
    "num_anchor_data", "nl_lambda", "dim_intermediate", "num_institution_user", "K_normalization", "anchor_method"
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
    "feature_num": None,
    "dim_intermediate": None,
    "dim_integrate": None,
    "num_institution": None,
    "lambda_gen_eigen": 0,
    "orth_ver": False,
    "K_normalization":True,
}

# --- 追加: dataset ごとのデフォルト適用（定数のみ。動的は未設定）---
_DATASET_DEFAULTS = {
    "qsar":                 {"feature_num": 41},#, "dim_intermediate": 37, "dim_integrate": 37, "num_institution_user": 25, "num_institution": 20},
    "adult":                {"feature_num": 51},#, "dim_intermediate": 50, "dim_integrate": 50, "num_institution_user": 150, "num_institution": 10},
    "diabetes130":          {"feature_num": 200},#, "dim_intermediate": 100, "dim_integrate": 100, "num_institution_user": 500, "num_institution": 10},
    "mice":                 {"feature_num": 77},#, "dim_intermediate": 46, "dim_integrate": 46, "num_institution_user": 50, "num_institution": 5},
    "breast_cancer":        {"feature_num": 15},#, "num_institution_user": 60},
    #"digits":               {"dim_intermediate": 15, "dim_integrate": 15, "num_institution_user": 100, "num_institution": 10},
    # "mnist":                {"dim_intermediate": 10, "dim_integrate": 10, "num_institution_user": 50, "num_institution": 10},
    "mnist":                {"dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 200, "num_institution": 4},
    #"fashion_mnist":        {"dim_intermediate": 10, "dim_integrate": 10, "num_institution_user": 50, "num_institution": 10},
    "concentric_circles":   {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 500, "num_institution": 2},
    "concentric_three_circles": {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 500, "num_institution": 2},
    "two_gaussian_distributions": {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution_user": 50, "num_institution": 5},
    "3D_gaussian_clusters": {"feature_num": 3, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
    "3D_8_gaussian_clusters": {"feature_num": 3, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
    "digits_":             {"dim_intermediate": 4, "dim_integrate": 4, "num_institution": 10, "num_institution_user": 100},
    "digits_v2":           {"dim_intermediate": 30, "dim_integrate": 30, "num_institution": 29, "num_institution_user": 30},
    "housing":             {"num_institution": 10, "num_institution_user": 10},
    #"statlog":             {"num_institution_user": 200},
    "wine_quality":       {"feature_num": 11},#, "dim_intermediate": 8},
    "glass":              {"feature_num": 9},#,  "dim_intermediate": 6},
    "seeds":              {"feature_num": 7},#,  "dim_intermediate": 5},
    "letter_recognition": {"feature_num": 16},#, "dim_intermediate": 12},
    "iris":               {"dim_intermediate": 3},
    "ecoli":              {"dim_intermediate": 5},
    "vowel":              {"dim_intermediate": 4},
}
RULES: List[Dict[str, Any]] = [
    {"type": "LOCK", "when": {"G_type": ["centralize", "individual"]}, "lock": {"gamma_ratio": DEFAULTS["gamma_ratio"]}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "Imakura", "GEP",  "ODC",]}, "lock": {"nl_lambda": DEFAULTS["nl_lambda"]}},
    {"type": "LOCK", "when": {"G_type": ['centralize', "individual", "Imakura", "GEP",  "ODC",]}, "lock": {"gamma_ratio_krr": DEFAULTS["gamma_ratio_krr"]}},
    #{"type": "SKIP", "when": {"F_type": ["kernel_pca"], "G_type": ["GEP_weighted"]}},
]
# ============================================

# ↓ 追記: CSV 由来のコンボ設定（汎用）
CSV_COMBO_PATH = r"c:\Users\sueya\Downloads\imakura_odc_condition_results.csv"
CSV_OVERRIDE_MAP: Dict[str, List[str]] = {
    # G_type ごとに、CSVの値で上書き・固定するカラム名を列挙
    "Imakura": ["dataset", "h_model", "F_type", "gamma_ratio", "dim_intermediate", "num_institution_user"],
}
# CSV 抽出の cond フィルタはオフ（必要なら True に）
CSV_USE_COND = False

def _iter_csv_combos(grid: Dict[str, List[Any]], csv_path: str, override_map: Dict[str, List[str]]):
    """
    CSVの特定G_type行を抽出し、override_mapで指定したキーだけCSV値で固定。
    それ以外のキーは PARAM_GRID を総当り（空は DEFAULTS、G_type が空なら CSV の G_type）で回す。
    抽出に使った G_type は“固定しない”。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    if CSV_USE_COND and "cond" in df.columns:
        df = df[df["cond"] == True]

    targets = set(override_map.keys())
    df = df[df["G_type"].isin(targets)].copy()
    if df.empty:
        return

    for g in targets:
        keys = override_map[g]
        sub = df[df["G_type"] == g]
        if sub.empty:
            continue

        sub = sub.dropna(subset=keys).drop_duplicates(subset=keys)

        for _, row in sub.iterrows():
            # CSVで固定するのは指定6項目のみ（G_typeは固定しない）
            fixed = {k: row[k] for k in keys}

            # 残りのキーは PARAM_GRID（空は DEFAULTS、G_type が空なら CSV の G_type）で総当り
            pairs: list[tuple[str, list[Any]]] = []
            for k in grid.keys():
                if k in keys:
                    continue
                vals = list(grid.get(k, []))
                if not vals:
                    if k == "G_type":
                        csv_g = row.get("G_type", None)
                        if pd.notna(csv_g):
                            vals = [csv_g]  # 抽出条件の G_type をフォールバックに使用
                    elif (k in DEFAULTS) and (DEFAULTS[k] is not None):
                        vals = [DEFAULTS[k]]
                    else:
                        # product 対象外（後段の _apply_defaults が埋める）
                        continue
                pairs.append((k, vals))

            if pairs:
                for tup in product(*(vals for _, vals in pairs)):
                    base = {k: v for (k, _), v in zip(pairs, tup)}
                    base.update(fixed)
                    after = _apply_lock_rules(base)
                    if _skip_by_rules(after):
                        continue
                    yield after
            else:
                # すべてCSV固定で他に回すものが無い場合（G_type が無ければこの分岐には来ない想定）
                after = _apply_lock_rules(dict(fixed))
                if not _skip_by_rules(after):
                    yield after
                    
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
        norm = tuple(sorted(after.items()))
        if norm in seen:
            continue
        seen.add(norm)
        yield after
            
def _iter_default_combos_excluding(grid: Dict[str, List[Any]], excluded_gtypes: set[str]):
    """除外 G_type を除いた通常のPARAM_GRIDコンボ"""
    for combo in _generate_unique_combos(grid):
        if combo.get("G_type") in excluded_gtypes:
            continue
        yield combo

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

def _generate_unique_combos(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    seen = set()
    for tup in product(*(grid[k] for k in keys)):
        base = {k: v for k, v in zip(keys, tup)}
        after = _apply_lock_rules(base)
        if _skip_by_rules(after):
            continue
        norm = tuple((k, after.get(k)) for k in keys)
        if norm in seen:
            continue
        seen.add(norm)
        yield after

def _set_config_from_combo(cfg: Config, combo: Dict[str, Any]) -> None:
    """dataset/metrics/visualize以外は config に流し込む。 ???"""
    for k, v in combo.items():
        if k in ("dataset", "metrics"):
            continue
        setattr(cfg, k, v)
    # True_F_type を常に同期
    if hasattr(cfg, "F_type"):
        cfg.True_F_type = cfg.F_type

def run_grid(
    config: Config,
    use_csv: bool | None = None,
    grid: Dict[str, List[Any]] | None = None,
    loop_num: int | None = None,
    csv_override_map: Dict[str, List[str]] | None = None,
    csv_combo_path: str | None = None,
    logger_=None,
) -> pd.DataFrame:
    rows = []
    all_columns = PARAM_COLUMNS + [
        "loop_num", "score_mean", "score_stdev",
        "even_ind_mean", "odd_ind_mean", "ind_mean",
        "mean_mean", "even_mean", "odd_mean", "integ_metrics_mean"
    ]
    # 追加: 注入値の優先適用
    grid = grid or PARAM_GRID
    loop = LOOP_NUM if loop_num is None else int(loop_num)
    use_csv_flag = True if use_csv is None else bool(use_csv)
    override_map = CSV_OVERRIDE_MAP if csv_override_map is None else csv_override_map
    csv_path = CSV_COMBO_PATH if csv_combo_path is None else csv_combo_path
    log = logger_ if logger_ is not None else getLogger(__name__)

    base_paths = dict(output_path=config.output_path, input_path=INPUT_DIR)

    if use_csv_flag and override_map:
        csv_iter = _iter_csv_combos(grid, csv_path, override_map) or iter(())
        def_iter = _iter_default_combos_excluding(grid, set(override_map.keys()))
        combos_iter = chain(csv_iter, def_iter)
    else:
        combos_iter = _generate_unique_combos(grid)

    for combo in combos_iter:
        dataset = combo["dataset"]
        metrics_name = combo["metrics"]
        cfg = Config(**base_paths)
        vals = []
        print(f"[pattern] { {k: combo[k] for k in PARAM_COLUMNS if k in combo} }")

        for i in range(loop):
            cfg.seed = i
            cfg.dataset = dataset
            cfg.metrics = metrics_name
            cfg.plot_name = f"_0913_{dataset}_{combo.get('F_type','-')}_{combo.get('G_type','-')}_{combo.get('K_normalization','-')}.png"

            _set_config_from_combo(cfg, combo)
            _apply_defaults(cfg, dataset, combo)

            #try:
            val = run_once(cfg, log)
            vals.append(float(val))
            record_config_to_cfg(cfg)
            record_value_to_cfg(cfg, "評価値", val)
            # except Exception as e:
            #     msg = f"[skip] seed={i}, dataset={dataset}, G_type={combo.get('G_type')}, reason={e}"
            #     print(msg)
            #     try:
            #         log.exception(msg)
            #     except Exception:
            #         pass
            #     try:
            #         record_value_to_cfg(cfg, "error", str(e))
            #     except Exception:
            #         pass
            #     continue

        mean_val = sum(vals) / len(vals) if vals else 0.0
        stdev_val = statistics.stdev(vals) if len(vals) > 1 else 0.0

        row = {k: combo.get(k, None) for k in PARAM_COLUMNS}
        row.update({
            "loop_num": loop,
            "score_mean": mean_val,
            "score_stdev": stdev_val,
        })
        row.update({
            "even_ind_mean": getattr(cfg, "losses_even_ind", 0),
            "odd_ind_mean": getattr(cfg, "losses_odd_ind", 0),
            "ind_mean": getattr(cfg, "losses_ind", 0),
            "mean_mean": getattr(cfg, "losses_mean", 0),
            "even_mean": getattr(cfg, "losses_even", 0),
            "odd_mean": getattr(cfg, "losses_odd", 0),
            "integ_metrics_mean": getattr(cfg, "integ_metrics", 0),
        })

        out_path = cfg.output_path / f"result_grid_{dataset}.csv"
        one = pd.DataFrame([row], columns=all_columns)
        header_needed = not out_path.exists()
        one.to_csv(out_path, mode="a", header=header_needed, index=False, encoding="utf-8-sig")
        print(f"[saved] {out_path}")

        rows.append(row)

    df_all = pd.DataFrame(rows, columns=all_columns)
    return df_all

# run.py
from config.config import Config
from src.paths import CONFIG_DIR, OUTPUT_DIR, INPUT_DIR
    
if __name__ == "__main__":
    # 引数処理はここだけ（デフォルトは 0912）
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="0913")
    parser.add_argument("--use_csv", action="store_true", help="CSV由来のコンボを使わず、PARAM_GRIDのみで総当りする")
    args = parser.parse_args()

    # 出力先を決定
    output_path = OUTPUT_DIR / args.run_name

    # Config/Logger をここでだけ作成
    config = Config(output_path=output_path, input_path=INPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    logger.handlers.clear()  # 重複防止
    handler = FileHandler(filename=config.output_path / "result.log", encoding="utf-8")
    logger.addHandler(handler)

    # 実行
    df = run_grid(config, use_csv=(args.use_csv), logger_=logger)
    df.to_csv(config.output_path / "result_grid_all.csv", index=False, encoding="utf-8-sig")