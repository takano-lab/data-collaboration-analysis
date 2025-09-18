from __future__ import annotations
from itertools import product
from itertools import chain
import argparse
from logging import INFO, FileHandler, getLogger
import statistics
import pandas as pd
import numpy as np  # 追加
import yaml
from tqdm import tqdm 
from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.data_collaboration import DataCollaborationAnalysis
from src.institutional_analysis import (
    centralize_analysis,
    centralize_analysis_with_dimension_reduction,
    dca_analysis,
    individual_analysis,
    fl_analysis,
    individual_analysis_with_dimension_reduction,
)
from src.load_data import load_data
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

def run_onec_(visualize):
    logger.info(f"データセット: {config.dataset}")
    print(f"データセット:{config.dataset}")
    config.f_seed = 0
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)
    
    metrics_dict = {}
    
    if config.F_type == "kernel_pca" and config.G_type == "GEP_weighted":
        # GEP_weightedはUSE_KERNELがTrueのときのみ実行
        return
    #if config.F_type == "kernel_pca" and config.G_type == "GEP":
        # GEP_weightedはUSE_KERNELがTrueのときのみ実行
    #    return
    config.log(logger, exclude_keys=["output_path", "input_path", "name", "seed", "y_name"])
    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
    # データ分割 -> 統合表現の獲得まで一気に実行
    #data_collaboration.save_optimal_params()
    data_collaboration.run()
    if visualize:
        data_collaboration.visualize_representations()
    #data_collaboration.save_representations_to_csv()
        # 提案手法
    #record_config_to_cfg(config)
    if config.G_type == 'centralize':
                # 集中解析
        metrics_cen = centralize_analysis(config, logger, y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", metrics_cen)
        return metrics_cen
    
    elif config.G_type == 'centralize_dim':
        # 集中解析 with 次元削減
        metrics_cen_dim = centralize_analysis_with_dimension_reduction(config, logger, y_name=config.y_name)
        metrics_dict['centralize_dim'] = metrics_cen_dim
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", metrics_cen_dim)
        return metrics_cen_dim
    
    elif config.G_type == 'individual':
        # 個別解析
        metrics_ind = individual_analysis(
            config=config,
            logger=logger,
            Xs_train=data_collaboration.Xs_train,
            ys_train=data_collaboration.ys_train,
            Xs_test=data_collaboration.Xs_test,
            ys_test=data_collaboration.ys_test,
        )
        #metrics_dict['individual'] = metrics_ind
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", metrics_ind)
        return metrics_ind
    
    elif config.G_type == 'individual_dim':
        # 個別解析 with 次元削減
        metrics_ind_dim = individual_analysis_with_dimension_reduction(
            config=config,
            logger=logger,
            Xs_train=data_collaboration.Xs_train,
            ys_train=data_collaboration.ys_train,
            Xs_test=data_collaboration.Xs_test,
            ys_test=data_collaboration.ys_test,
        )
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", metrics_ind_dim)
        return metrics_ind_dim
    
    elif config.G_type == 'fl':
        metrics_fl = fl_analysis(
            config=config,
            logger=logger,
            Xs_train=data_collaboration.Xs_train,
            ys_train=data_collaboration.ys_train,
            Xs_test=data_collaboration.Xs_test,
            ys_test=data_collaboration.ys_test,
        )
        metrics_dict['fl'] = metrics_fl
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", metrics_fl)
        return metrics_fl
    else:
        config.f_seed = 0
        metrics_ind_dim = individual_analysis_with_dimension_reduction(
            config=config,
            logger=logger,
            Xs_train=data_collaboration.Xs_train,
            ys_train=data_collaboration.ys_train,
            Xs_test=data_collaboration.Xs_test,
            ys_test=data_collaboration.ys_test,
        )
        # metrics = dca_analysis(
        #                 X_train_integ=data_collaboration.X_train_integ,
        #                 X_test_integ=data_collaboration.X_test_integ,
        #                 y_train_integ=data_collaboration.y_train_integ,
        #                 y_test_integ=data_collaboration.y_test_integ,
        #                 config=config,
        #                 logger=logger,
        #             )
        # record_config_to_cfg(config)
        # record_value_to_cfg(config, "評価値", metrics)
        # print("評価値", metrics)
        #return metrics
        
        # --- ここから機関ごとの metrics を算出 ---
        # 各機関のサンプル数（元リスト）から、統合後配列のスライス境界を作る
        #train_counts = [len(y) for y in data_collaboration.ys_train]
        test_counts  = [len(y) for y in data_collaboration.ys_test]
        test_counts  = [config.num_institution_user for y in data_collaboration.ys_test]
        n_inst = config.num_institution

        #train_cum = np.concatenate(([0], np.cumsum(train_counts)))
        test_cum  = np.concatenate(([0], np.cumsum(test_counts)))

        inst_losses = []
        even_losses = []
        odd_losses = []
        
        config.f_seed = 0
        for i in range(n_inst):
            # 各機関の訓練・テストから num_institution_user 件だけ使用
            #tr_start, tr_end = int(train_cum[i]), int(train_cum[i+1])
            te_start, te_end = int(test_cum[i]),  int(test_cum[i+1])
            #tr_take = min(config.num_institution_user, tr_end - tr_start)
            #te_take = min(config.num_institution_user, te_end - te_start)
            te_take = config.num_institution_user
            #X_tr_i = data_collaboration.X_train_integ[tr_start: tr_start + tr_take, :]
            #y_tr_i = data_collaboration.y_train_integ[tr_start: tr_start + tr_take]
            X_te_i = data_collaboration.X_test_integ[te_start:  te_start  + te_take,  :]
            y_te_i = data_collaboration.y_test_integ[te_start:  te_start  + te_take]


            metric_i = dca_analysis(
                X_train_integ=data_collaboration.X_train_integ,
                X_test_integ=X_te_i,
                y_train_integ=data_collaboration.y_train_integ,
                y_test_integ=y_te_i,
                config=config,
                logger=logger,
            )
            inst_losses.append(metric_i)
            
            if i % 2 == 0:
                even_losses.append(metric_i)
            else:
                odd_losses.append(metric_i)

        # 平均・最小・最大を算出して出力
        inst_losses = np.array(inst_losses, dtype=float)
        mean_val = float(inst_losses.mean())
        min_val  = float(inst_losses.min())
        max_val  = float(inst_losses.max())
        
        config.losses_mean = round(mean_val, 4)
        config.losses_even =  round(sum(even_losses)/len(even_losses), 4)
        config.losses_odd = round(sum(odd_losses)/len(odd_losses), 4)
        #record_config_to_cfg(config)
        
        print("評価値2", mean_val)
        #record_config_to_cfg(config)
        #record_value_to_cfg(config, "評価値", mean_val)
        #print("評価値", mean_val)
        print("config.losses_mean", config.losses_mean)
        print(f"機関ごとの {config.metrics}: {np.round(inst_losses, 4).tolist()}")
        print(f"平均: {mean_val:.4f}, 最小: {min_val:.4f}, 最大: {max_val:.4f}")
        logger.info(f"機関ごとの {config.metrics}: {inst_losses.tolist()}")
        logger.info(f"平均: {mean_val:.6f}, 最小: {min_val:.6f}, 最大: {max_val:.6f}")

        # main_loop の集計用に平均値を返す
        return mean_val    
    
    
    # 個別解析
    # metrics_ind = individual_analysis_with_dimension_reduction(
    #     config=config,
    #     logger=logger,
    #     Xs_train=data_collaboration.Xs_train,
    #     ys_train=data_collaboration.ys_train,
    #     Xs_test=data_collaboration.Xs_test,
    #     ys_test=data_collaboration.ys_test,
    # )
    #metrics_dict['individual_dim'] = metrics_ind
    
        # 個別解析 2 
    # individual_analysis(
    #     config=config,
    #     logger=logger,
    #     Xs_train=data_collaboration.Xs_train_inter,
    #     ys_train=data_collaboration.ys_train,
    #     Xs_test=data_collaboration.Xs_test_inter,
    #     ys_test=data_collaboration.ys_test,
    # )
    #return metrics_dict 

def main_loop():
    LOADERS = [
    #    "concentric_three_circles",
    #    "mice",
    #  "statlog",
        'qsar',
    #   "breast_cancer",
    #    "adult",
    #    "digits",
    #    "concentric_circles",
    #    "har",
    #    "diabetes130",
    #    "bank_marketing", # 性能に変化でない
    #    "two_gaussian_distributions",
    #    '3D_gaussian_clusters',
    #    "3D_8_gaussian_clusters",
    #"digits_v2",
    #"housing",
    #"ames",
    #"tox21_sr_are",
    #"hiv",
    #"cyp3a4",
    #"cyp2d6",
    #"cyp1a2",
    #"mnist",
    #"fashion_mnist",
    ]
    MODELS = ["mlp"]#, "mlp"]#"random_forest"]#, "svm_linear_classifier", "mlp"]#"mlp"]#, "svm_linear_classifier"] #"svm_classifier"]#"random_forest"]#, _linear_
    gamma_types = ["X_tuning"] 
    F_types = ["kernel_pca_svd_mixed"]#"kernel_pca_svd_mixed", "svd", "kernel_pca_self_tuning"]#, "svd", "kernel_pca_self_tuning"]#"svd", "kernel_pca", "kernel_pca_self_tuning", ] # , "kernel_pca", "lpp" # "kernel_pca_self_tuning" "kernel_pca_svd_mixed",
   #G_types = ['centralize', "individual", "Imakura", "GEP",  "ODC", "nonlinear"]#, 'centralize', "individual", "Imakura", "GEP",  "ODC", "nonlinear"]# "nonlinear_tuning"#'centralize_dim', "nonlinear", "Imakura"]#"nonlinear_tuning"]#, "nonlinear", "nonlinear_tuning", "nonlinear_linear"]#["fl", 'centralize', 'individual', "Imakura", "ODC", "GEP", "nonlinear", "nonlinear_tuning", "nonlinear_linear"]#'centralize_dim', "nonlinear", "Imakura"]#
    G_types = ["nonlinear"] # "nonlinear", "Imakura", "GEP",  
    config.F_type = F_types[0]
    F_type = F_types[0]
    config.True_F_type = F_types[0]
    config.G_type = G_types[0]
    config.objective_direction_ratio = 0
    config.gamma_type = "X_tuning"
    config.lambda_pred = 0#10
    config.lambda_offdiag = 0#100000
    config.h_model = MODELS[0]
    config.nl_lambda = 0.1
    visualize = False
    data = {}
    model = MODELS[0]
    config.losses_even_ind = 0
    config.losses_odd_ind = 0
    config.losses_ind = 0
    config.losses_mean = 0
    config.losses_even = 0
    config.losses_odd = 0
    config.integ_metrics = 0
    for dataset in tqdm(LOADERS):
        config.now = "f"
        for met in ["auc"]:#, "accuracy"
            config.metrics = met
            config.h_model = model
            print(dataset)
            config.dataset = dataset
            for gamma_ratio in [1]:#0.1, 1, 5]: # [0.01, 0.1, 1, 10, 100]
                F_type = F_types[0]
                config.gamma_ratio = gamma_ratio
                config.F_type = F_type
                config.True_F_type = F_type
                for G_type in G_types:
                    config.G_type = G_type
                    for lw_alpha in [0]:
                        config.lw_alpha = lw_alpha
                        config.lb_beta = lw_alpha
                        semi_integ = False
                        orth = False
                        config.semi_integ = semi_integ
                        config.orth_ver = orth
                        metrics = []
                        losses_even_ind_list = []
                        losses_odd_ind_list = []
                        losses_ind_list = []
                        losses_mean_list = []
                        losses_even_list = []
                        losses_odd_list = []
                        integ_metrics_list = []
                        
                        if G_type == "centralize" or  "individual":
                            losses_even_ind_list = [0]
                            losses_odd_ind_list = [0]
                            losses_ind_list = [0]
                            losses_mean_list = [0]
                            losses_even_list = [0]
                            losses_odd_list = [0]
                            integ_metrics_list = [0]

                        for i in range(3, 4):
                            config.seed = i
                            #config.f_seed = i
                            config.plot_name = f"_0912_{dataset}_{G_type}.png" # {self.config.lambda_pred}_{self.config.dataset}
                            print("i", i, "G_type:", G_type)
                            metrics.append(run_once(visualize))
                            losses_even_ind_list.append(config.losses_even_ind)
                            losses_odd_ind_list.append(config.losses_odd_ind)
                            losses_ind_list.append(config.losses_ind)
                            losses_mean_list.append(config.losses_mean)
                            losses_even_list.append(config.losses_even)
                            losses_odd_list.append(config.losses_odd)
                            integ_metrics_list.append(config.integ_metrics)
                            config.F_type = config.True_F_type
                        # 平均値を計算
                        metrics_mean = sum(metrics) / len(metrics)
                        metrics_stdev = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
                        losses_even_ind_mean = sum(losses_even_ind_list) / len(losses_even_ind_list)
                        losses_odd_ind_mean = sum(losses_odd_ind_list) / len(losses_odd_ind_list)
                        losses_ind_mean = sum(losses_ind_list) / len(losses_ind_list)
                        losses_mean_mean = sum(losses_mean_list) / len(losses_mean_list)
                        losses_even_mean = sum(losses_even_list) / len(losses_even_list)
                        losses_odd_mean = sum(losses_odd_list) / len(losses_odd_list)
                        integ_metrics_mean = sum(integ_metrics_list) / len(integ_metrics_list)
                        losses_even_ind_stdev = statistics.stdev(losses_even_ind_list) if len(losses_even_ind_list) > 1 else 0.0
                        losses_odd_ind_stdev = statistics.stdev(losses_odd_ind_list) if len(losses_odd_ind_list) > 1 else 0.0
                        losses_ind_stdev = statistics.stdev(losses_ind_list) if len(losses_ind_list) > 1 else 0.0
                        losses_mean_stdev = statistics.stdev(losses_mean_list) if len(losses_mean_list) > 1 else 0.0
                        losses_even_stdev = statistics.stdev(losses_even_list) if len(losses_even_list) > 1 else 0.0
                        losses_odd_stdev = statistics.stdev(losses_odd_list) if len(losses_odd_list) > 1 else 0.0
                        integ_metrics_stdev = statistics.stdev(integ_metrics_list) if len(integ_metrics_list) > 1 else 0.0
                        data[f'{dataset}_{F_type}_{model}_{config.G_type}_{gamma_ratio}_{met}'] = [dataset, model, F_type, G_type, gamma_ratio, met, metrics_mean, metrics_stdev, losses_even_ind_mean, losses_even_ind_stdev, losses_odd_ind_mean, losses_odd_ind_stdev, losses_ind_mean, losses_ind_stdev, losses_mean_mean, losses_mean_stdev, losses_even_mean, losses_even_stdev, losses_odd_mean, losses_odd_stdev, integ_metrics_mean, integ_metrics_stdev]


        # DataFrameに変換
        df_all = pd.DataFrame.from_dict(data, orient="index", columns=["dataset", "model", "F_type", "G_type", "gamma_ratio", "metrics", "metrics_mean", "metrics_stdev", "even_ind_mean", "even_ind_stdev", "odd_ind_mean", "odd_ind_stdev", "ind_mean", "ind_stdev", "mean_mean", "mean_stdev", "even_mean", "even_stdev", "odd_mean", "odd_stdev", "integ_metrics_mean", "integ_metrics_stdev"])
        df_all.to_csv(output_path / f"result_{dataset}_0912.csv", index=True, encoding="utf-8-sig")

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
    "digits",
    "glass", "seeds", "letter_recognition",
    "wine_quality",
    "har",
    "diabetes130",
    "bank_marketing",
    "mnist",
    "fashion_mnist",
],#"wine_quality", "glass", "seeds", "letter_recognition"],#"wine_quality", #"qsar","mice", "statlog", "breast_cancer", "adult", "digits",],     # 例: ["qsar","mice"]
    "h_model": ["svm_linear_classifier"],             # 例: ["mlp","random_forest"] svm_linear_classifier
    "F_type": ["kernel_pca_svd_mixed"],
    "G_type": ["nonlinear", "Imakura", "ODC"], # 'centralize', "individual", "Imakura", "GEP",  "ODC",
    "gamma_type": ["X_tuning"],
    "gamma_ratio": [],#[0.1, 0.3, 1, 3, 10],             # 例: [0.1,1,5]
    "gamma_ratio_krr": [0.04, 0.2, 1],
    "num_anchor_data": [100],
    "nl_lambda": [0.1, 0.00001],        # LOCKで止められる
    "lw_alpha": [0],
    "lambda_pred": [0],
    "lambda_offdiag": [0],
    "metrics": ["auc"],
    "visualize": [False],
    "dim_intermediate": [],#[20, 10, 5, 2],
    "num_institution_user": [],#[50, 100, 200, 400],
    "num_institution": [2],
})

# 2) ループ回数（seed を 0..loop_num-1 で回します）
LOOP_NUM = 3

# 3) DataFrameに保持したい「パラメータ列」（順序もこの通り）
PARAM_COLUMNS: List[str] = [
    "dataset", "h_model", "F_type", "G_type", "gamma_type", "gamma_ratio", "gamma_ratio_krr",
    "num_anchor_data", "nl_lambda", "dim_intermediate", "num_institution_user"
]

# 4) 条件ルール
#    - LOCK: 条件一致時に指定パラメータを固定（そのキーは“ループしない”）
#    - SKIP: 条件一致の組合せを丸ごとスキップ
DEFAULTS = {
    "y_name": "target",
    "nl_lambda": 0.1,
    "gamma_ratio": 1,
    "gamma_ratio_krr": 1,
    "num_institution_user": 50,
    "feature_num": None,
    "dim_intermediate": None,
    "dim_integrate": None,
    "num_institution": None,
    "lambda_gen_eigen": 0,
    "orth_ver": False,
}

# --- 追加: dataset ごとのデフォルト適用（定数のみ。動的は未設定）---
_DATASET_DEFAULTS = {
    "qsar":                 {"feature_num": 41},#, "dim_intermediate": 37, "dim_integrate": 37, "num_institution_user": 25, "num_institution": 20},
    "adult":                {"feature_num": 51},#, "dim_intermediate": 50, "dim_integrate": 50, "num_institution_user": 150, "num_institution": 10},
    "diabetes130":          {"feature_num": 200},#, "dim_intermediate": 100, "dim_integrate": 100, "num_institution_user": 500, "num_institution": 10},
    "mice":                 {"feature_num": 77},#, "dim_intermediate": 46, "dim_integrate": 46, "num_institution_user": 50, "num_institution": 5},
    "breast_cancer":        {"feature_num": 15},#, "num_institution_user": 60},
    #"digits":               {"dim_intermediate": 15, "dim_integrate": 15, "num_institution_user": 100, "num_institution": 10},
    #"mnist":                {"dim_intermediate": 10, "dim_integrate": 10, "num_institution_user": 50, "num_institution": 10},
    #"fashion_mnist":        {"dim_intermediate": 10, "dim_integrate": 10, "num_institution_user": 50, "num_institution": 10},
    "concentric_circles":   {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
    "concentric_three_circles": {"feature_num": 2, "dim_intermediate": 2, "dim_integrate": 2, "num_institution": 2},
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

def run_grid(config: Config) -> pd.DataFrame:
    rows = []
    all_columns = PARAM_COLUMNS + [
        "loop_num", "score_mean", "score_stdev",
        "even_ind_mean", "odd_ind_mean", "ind_mean",
        "mean_mean", "even_mean", "odd_mean", "integ_metrics_mean"
    ]

    # 外で決めた固定値（ここだけ引き継ぐ）
    base_paths = dict(output_path=config.output_path, input_path=INPUT_DIR)
    
    # ↓ 変更: CSV 由来のコンボ + 通常グリッド（CSVで上書き対象G_typeは除外）
    csv_iter = _iter_csv_combos(PARAM_GRID, CSV_COMBO_PATH, CSV_OVERRIDE_MAP)
    def_iter = _iter_default_combos_excluding(PARAM_GRID, set(CSV_OVERRIDE_MAP.keys()))
    combos_iter = chain(csv_iter, def_iter)

    # ここを combos_iter に変更（CSV固定＋PARAM_GRID総当りを実行）
    for combo in combos_iter:
        dataset = combo["dataset"]
        metrics_name = combo["metrics"]

        # combo ごとに Config をリセット
        cfg = Config(**base_paths)

        vals = []
        print(f"[pattern] { {k: combo[k] for k in PARAM_COLUMNS if k in combo} }")

        for i in range(LOOP_NUM):
            # 以降は cfg を使用（元の config は触らない）
            cfg.seed = i
            cfg.dataset = dataset
            cfg.metrics = metrics_name
            cfg.plot_name = f"_0913_{dataset}_{combo.get('G_type','-')}_{metrics_name}.png"

            _set_config_from_combo(cfg, combo)
            _apply_defaults(cfg, dataset, combo)
            
            # val = run_once(cfg, logger)
            # vals.append(float(val))
            # record_config_to_cfg(cfg)
            # record_value_to_cfg(cfg, "評価値", val)

            try:
                val = run_once(cfg, logger)
                vals.append(float(val))
                record_config_to_cfg(cfg)
                record_value_to_cfg(cfg, "評価値", val)
            except Exception as e:
                msg = f"[skip] seed={i}, dataset={dataset}, G_type={combo.get('G_type')}, reason={e}"
                print(msg)
                try:
                    logger.exception(msg)
                except Exception:
                    pass
                try:
                    record_value_to_cfg(cfg, "error", str(e))
                except Exception:
                    pass
                continue

        mean_val = sum(vals) / len(vals) if vals else 0.0
        stdev_val = statistics.stdev(vals) if len(vals) > 1 else 0.0

        row = {k: combo.get(k, None) for k in PARAM_COLUMNS}
        row.update({
            "loop_num": LOOP_NUM,
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
import argparse, yaml
from config.config import Config
from src.paths import CONFIG_DIR, OUTPUT_DIR, INPUT_DIR
    
if __name__ == "__main__":
    # 引数処理はここだけ（デフォルトは 0912）
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="0913")
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
    df = run_grid(config)
    df.to_csv(config.output_path / "result_grid_all.csv", index=False, encoding="utf-8-sig")