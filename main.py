from __future__ import annotations

import argparse
from logging import INFO, FileHandler, getLogger

import pandas as pd
import yaml

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.data_collaboration import DataCollaborationAnalysis
from src.institutional_analysis import (
    centralize_analysis,
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
    if config.F_type == "kernel_pca" and config.G_type == "GEP":
        # GEP_weightedはUSE_KERNELがTrueのときのみ実行
        return
    config.log(logger, exclude_keys=["output_path", "input_path", "name", "seed", "y_name"])
    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
    # データ分割 -> 統合表現の獲得まで一気に実行
    #data_collaboration.save_optimal_params()
    data_collaboration.run()

        # 提案手法
    record_config_to_cfg(config)
    metrics = dca_analysis(
                    X_train_integ=data_collaboration.X_train_integ,
                    X_test_integ=data_collaboration.X_test_integ,
                    y_train_integ=data_collaboration.y_train_integ,
                    y_test_integ=data_collaboration.y_test_integ,
                    config=config,
                    logger=logger,
                )
    metrics_dict[f'{config.F_type}_{config.G_type}'] = metrics
    
    # 集中解析
    metrics_cen = centralize_analysis(config, logger, y_name=config.y_name)
    
    metrics_dict['centralize'] = metrics_cen

    # 個別解析
    metrics_ind = individual_analysis(
        config=config,
        logger=logger,
        Xs_train=data_collaboration.Xs_train,
        ys_train=data_collaboration.ys_train,
        Xs_test=data_collaboration.Xs_test,
        ys_test=data_collaboration.ys_test,
    )
    metrics_dict['individual'] = metrics_ind
    
    # 個別解析
    metrics_ind = individual_analysis_with_dimension_reduction(
        config=config,
        logger=logger,
        Xs_train=data_collaboration.Xs_train,
        ys_train=data_collaboration.ys_train,
        Xs_test=data_collaboration.Xs_test,
        ys_test=data_collaboration.ys_test,
    )
    metrics_dict['individual_dim'] = metrics_ind
    
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
    #    "statlog",
        'qsar',
    #   "breast_cancer",
    #    "har",
    #    "adult",
    #    "diabetes130",
    #    "bank_marketing", # 性能に変化でない
    ]
    MODELS = ["svm_linear_classifier"]#"random_forest"]#, "svm_classifier"]
    
    F_types =["kernel_pca"]#["svd", "kernel_pca"]
    G_types = ["Imakura"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]
    #G_types = ["nonlinear"]
    config.F_type = F_types[0]
    config.G_type = G_types[0]
    
    data = {}
    for dataset in LOADERS:
        config.G_type = "Imakura"
        for model in MODELS:
            config.dataset = dataset
            config.h_model = model
            data[f'{dataset}_{model}'] = main()
            
        config.G_type = "nonlinear"
    
        for model in MODELS:
            for nl_lambda in [0.0000001]:
            #for nl_lambda in [0.0001, 0.01]:
                config.nl_lambda = nl_lambda
                config.dataset = dataset
                config.h_model = model
            #    data[f'{dataset}_{model}_{nl_lambda}'] = main()
                
                #for nl_gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                #for nl_gamma in [0.0001, 0.005, 0.05, 0.5]:
                for r in [10, 100, 1000, 3000]:
                    config.nl_gamma = 0.001# nl_gamma
                    config.num_anchor_data = r
                    #     config.dataset = dataset
                    #     config.h_model = model
                    data[f'{dataset}_{model}'] = main()

    # DataFrameに変換
    #df_all = pd.DataFrame.from_dict(data, orient="index")
    #df_all.to_csv(output_path / "result.csv", index=True, encoding="utf-8-sig")
    
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