from __future__ import annotations

import argparse
from logging import INFO, FileHandler, getLogger
import statistics
import pandas as pd
import yaml

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.data_collaboration import DataCollaborationAnalysis
from src.institutional_analysis import (
    centralize_analysis,
    centralize_analysis_with_dimension_reduction,
    dca_analysis,
    individual_analysis,
    individual_analysis_with_dimension_reduction,
)
from src.load_data import load_data
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, default="exp001")
args = parser.parse_args()

# yaml のパスと出力先パス
cfg_path    = CONFIG_DIR / f"{args.name}.yaml"
output_path = OUTPUT_DIR / args.name

# UTF-8 で読み込んで Config を生成
with cfg_path.open(encoding="utf-8") as f:
    cfg_dict = yaml.safe_load(f)

config = Config(**cfg_dict,
                output_path=output_path,
                input_path=INPUT_DIR)

# 出力ディレクトリ作成
output_path.mkdir(parents=True, exist_ok=True)

# ログの設定
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = FileHandler(filename=config.output_path / "result.log", encoding="utf-8")
logger.addHandler(handler)

def main():
    logger.info(f"データセット: {config.dataset}")
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
    #data_collaboration.visualize_representations()
    #data_collaboration.save_representations_to_csv()
        # 提案手法
    record_config_to_cfg(config)
    if config.G_type == 'centralize':
                # 集中解析
        metrics_cen = centralize_analysis(config, logger, y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        return metrics_cen
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
        return metrics_ind
    else:
        metrics = dca_analysis(
                        X_train_integ=data_collaboration.X_train_integ,
                        X_test_integ=data_collaboration.X_test_integ,
                        y_train_integ=data_collaboration.y_train_integ,
                        y_test_integ=data_collaboration.y_test_integ,
                        config=config,
                        logger=logger,
                    )
        #metrics_dict[f'{config.F_type}_{config.G_type}'] = 
        return metrics
    
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
    return metrics_dict 

def main_loop():
    LOADERS = [
    #   "statlog",
    #    'qsar',
    #   "breast_cancer",
    #    "har",
    #    "adult",
    #    "diabetes130",
    #    "bank_marketing", # 性能に変化でない
    #"digits",
    #"concentric_circles"
    "two_gaussian_distributions",
    #"mice",
    #"ames",
    #"tox21_sr_are",
    #"hiv",
    #"cyp3a4",
    #"cyp2d6",
    #"cyp1a2",
    #"mnist",
    #"fashion_mnist",
    ]
    MODELS = ["mlp"] #"svm_classifier"]#"random_forest"]#, _linear_
    
    F_types =["svd"]#["svd", "kernel_pca"]diffspan
    G_types = ['GEP']#'centralize', 'individual', "Imakura", "ODC", "GEP"]#"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]#, 'individual',"Imakura", "GEP"]#'centralize', 'individual', "Imakura", "ODC", "GEP", "GEP_weighted"]#"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]#, 'individual',"Imakura", "GEP", "GEP_weighted", "nonlinear"]#"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]'centralize', 'individual',
    config.metrics = "accuracy"
    #G_types = ["nonlinear"]
    config.F_type = F_types[0]
    config.G_type = G_types[0]
    
    data = {}
    config.nl_gamma = 0.002
    config.nl_lambda = 0.1
    config.h_model = MODELS[0]
    #for dim_m in [2]:
    #for dim_m in [36, 37, 38, 39]:
    for dataset in LOADERS:
        config.dataset = dataset
        #for num_inst in [5, 10]:#:, 15, 20]:
        for G_type in G_types:
            config.G_type = G_type
            #config.num_institution = num_inst
                #for lambda_ in [0, 0.0001, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 20, 40, 60, 80, 100, 1000]:
                #for lambda_ in [0.2, 0.4, 0.6, 0.8, 2, 4, 6, 8, 20, 40, 60, 80]:
                #for lambda_ in [1000, 100000]:
            #for lambda_ in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            metrics = []
            #config.lambda_gen_eigen = lambda_
            for i in range(1, 2):
                config.seed = i
                #config.nl_gamma = (lambda_+0.01) ** -1 
                metrics.append(main())
            # 平均値を計算
            metrics_mean = sum(metrics) / len(metrics)
            metrics_stdev = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
            data[f'{dataset}_{G_type}'] = [dataset, G_type, metrics_mean, metrics_stdev]

    # DataFrameに変換
    df_all = pd.DataFrame.from_dict(data, orient="index", columns=["dataset", "G_type", "metrics_mean", "metrics_stdev"])
    df_all.to_csv(output_path / "result.csv", index=True, encoding="utf-8-sig")
    
def partial_run():
    logger.info(f"データセット: {config.dataset}")
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)
    
    metrics_dict = {}
    # dim_intermediate,dim_integrate
    F_types =["kernel_pca"]#["svd", "kernel_pca"]
    G_types = ["Imakura", "nonlinear"]#, "targetvec", "GEP", "GEP_weighted"]

    #for F_type in F_types:
    dim_intermediate = config.dim_intermediate
    for dim_intermediate in range(1, dim_intermediate + 1, 5):
        config.dim_intermediate = dim_intermediate
        config.dim_integrate = config.dim_intermediate
        for G_type in G_types:
            config.F_type = F_types[0]
            config.G_type = G_type
            if config.F_type == "kernel_pca" and config.G_type == "GEP_weighted":
                # GEP_weightedはUSE_KERNELがTrueのときのみ実行
                return
            if config.F_type == "kernel_pca" and config.G_type == "GEP":
                # GEP_weightedはUSE_KERNELがTrueのときのみ実行
                return
            # インスタンスの生成
            data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
            # データ分割 -> 統合表現の獲得まで一気に実行
            data_collaboration.run()
    
if __name__ == "__main__":
    name = "main_loop"
    if name == "main_loop":
        main_loop()
    elif name == "partial_run":
        partial_run()
    else:
        main()