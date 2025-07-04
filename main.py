from __future__ import annotations

import argparse
from logging import INFO, FileHandler, getLogger

import pandas as pd
import yaml

from config.config import Config
from src.data_collaboration import DataCollaborationAnalysis
from src.institutional_analysis import centralize_analysis, dca_analysis, individual_analysis
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
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)
    
    metrics_dict = {}
    F_types = {True: "Kernel_PCA", False: "SVD"}
    
    USE_KERNELS = [False]#, True]
    G_types = ["GEP_weighted"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]
    for USE_KERNEL in USE_KERNELS:
        for G_type in G_types:
            if USE_KERNEL and G_type == "GEP_weighted":
                # GEP_weightedはUSE_KERNELがTrueのときのみ実行
                continue
            if USE_KERNEL and G_type == "GEP":
                # GEP_weightedはUSE_KERNELがTrueのときのみ実行
                continue
            # インスタンスの生成
            data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
            # データ分割 -> 統合表現の獲得まで一気に実行
            data_collaboration.run(USE_KERNEL=USE_KERNEL, G_type=G_type)
            print(f"USE_KERNEL: {F_types[USE_KERNEL]}, G_type: {G_type}")
                # 提案手法
            metrics = dca_analysis(
                            X_train_integ=data_collaboration.X_train_integ,
                            X_test_integ=data_collaboration.X_test_integ,
                            y_train_integ=data_collaboration.y_train_integ,
                            y_test_integ=data_collaboration.y_test_integ,
                            config=config,
                            logger=logger,
                        )
            metrics_dict[f'{F_types[USE_KERNEL]}_{G_type}'] = metrics


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
        "statlog",
        'qsar',
        "breast_cancer",
        "bank_marketing",
        "har",
    #    "adult",
    #    "diabetes130",
    ]
    MODELS = ["svm_classifier"]#"random_forest"]#, "svm_classifier"]
    
    data = {}
    
    for dataset in LOADERS:
        for model in MODELS:
            config.dataset = dataset
            config.h_model = model
            data[f'{dataset}_{model}'] = main()
            
    # DataFrameに変換
    df_all = pd.DataFrame.from_dict(data, orient="index")
    df_all.to_csv(output_path / "result.csv", index=True, encoding="utf-8-sig")
    
def partial_run():
    logger.info(f"データセット: {config.dataset}")
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)
    
    metrics_dict = {}
    F_types = {True: "Kernel_PCA", False: "SVD"}
    USE_KERNELS = [False]#, True]
    USE_KERNEL = [0]
    G_types = ["GEP"]#"Imakura"]#, "targetvec", "GEP", "GEP_weighted"]
    G_type = G_types[0]
    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
    # データ分割 -> 統合表現の獲得まで一気に実行
    data_collaboration.run(USE_KERNEL=USE_KERNEL, G_type=G_type)
    print(f"USE_KERNEL: {F_types[USE_KERNEL]}, G_type: {G_type}")
        # 提案手法
    metrics = dca_analysis(
                    X_train_integ=data_collaboration.X_train_integ,
                    X_test_integ=data_collaboration.X_test_integ,
                    y_train_integ=data_collaboration.y_train_integ,
                    y_test_integ=data_collaboration.y_test_integ,
                    config=config,
                    logger=logger,
                )
    metrics_dict[f'{F_types[USE_KERNEL]}_{G_type}'] = metrics


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
    
if __name__ == "__main__":
    main_loop()