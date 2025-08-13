from __future__ import annotations

import argparse
from logging import INFO, FileHandler, getLogger
import statistics
import pandas as pd
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
    data_collaboration.visualize_representations()
    #data_collaboration.save_representations_to_csv()
        # 提案手法
    record_config_to_cfg(config)
    if config.G_type == 'centralize':
                # 集中解析
        metrics_cen = centralize_analysis(config, logger, y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        return metrics_cen
    elif config.G_type == 'centralize_dim':
        # 集中解析 with 次元削減
        metrics_cen_dim = centralize_analysis_with_dimension_reduction(config, logger, y_name=config.y_name)
        metrics_dict['centralize_dim'] = metrics_cen_dim
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
        return metrics_ind
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
        return metrics_fl
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
        "concentric_circles",
    #    "two_gaussian_distributions",
    #    '3D_gaussian_clusters',
    #    "mice",
    #   "statlog",
    #    'qsar',
    #   "breast_cancer",
    #   "har",
    #    "adult",
    #    "diabetes130",
    #    "bank_marketing", # 性能に変化でない
    #"digits",
    #"ames",
    #"tox21_sr_are",
    #"hiv",
    #"cyp3a4",
    #"cyp2d6",
    #"cyp1a2",
    #"mnist",
    #"fashion_mnist",
    ]
    MODELS = ["random_forest", "svm_linear_classifier", "mlp"]#"mlp"]#, "svm_linear_classifier"] #"svm_classifier"]#"random_forest"]#, _linear_
    gamma_types = ["auto", "X_tuning"]
    F_types =["kernel_pca_svd_mixed"]#"svd", "kernel_pca", "kernel_pca_self_tuning", ] # , "kernel_pca", "lpp" # "kernel_pca_self_tuning" "kernel_pca_svd_mixed",
    G_types = ["GEP"]#["fl", 'centralize', 'individual', "Imakura", "ODC", "GEP", "targetvec", "nonlinear", "nonlinear_tuning", "nonlinear_linear"]#'centralize_dim', "nonlinear", "Imakura"]#"nonlinear_tuning"]#, "nonlinear", "nonlinear_tuning", "nonlinear_linear"]#["fl", 'centralize', 'individual', "Imakura", "ODC", "GEP", "nonlinear", "nonlinear_tuning", "nonlinear_linear"]#'centralize_dim', "nonlinear", "Imakura"]#
    config.metrics = "auc"
    #G_types = ["nonlinear"]
    config.F_type = F_types[0]
    config.G_type = G_types[0]
    
    data = {}
    #config.nl_gamma = 1
    config.nl_lambda = 10
    config.h_model = MODELS[0]

    for dataset in tqdm(LOADERS):
        #for model in MODELS:
        for nl_lambda in [1]:
            config.nl_lambda = nl_lambda
            #config.h_model = model
            print(dataset)
            model = MODELS[0]
            config.dataset = dataset
            #for num_inst in [5, 10]:#:, 15, 20]:
            for F_type in F_types:
                config.F_type = F_type
                config.True_F_type = F_type
                for G_type in G_types:
                        config.G_type = G_type
                        for gamma_type in gamma_types:
                            config.gamma_type = gamma_type
                            if (G_type != "nonlinear_linear" and G_type != "nonlinear") and gamma_type == "X_tuning":
                                continue
                            metrics = []
                            for i in range(1, 2):
                                config.seed = i
                                metrics.append(main())
                                config.F_type = config.True_F_type
                            # 平均値を計算
                            metrics_mean = sum(metrics) / len(metrics)
                            metrics_stdev = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
                            data[f'{dataset}_{model}_{F_type}_{G_type}_{gamma_type}'] = [dataset, model, F_type, G_type, gamma_type, metrics_mean, metrics_stdev]


    # DataFrameに変換
    df_all = pd.DataFrame.from_dict(data, orient="index", columns=["dataset", "model", "F_type", "G_type", "gamma_type", "metrics_mean", "metrics_stdev"])
    df_all.to_csv(output_path / "result.csv", index=True, encoding="utf-8-sig")
    
def partial_run():
    logger.info(f"データセット: {config.dataset}")
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)
    
    metrics_dict = {}
    # dim_intermediate,dim_integrate
    F_types =["kernel_pca"]#["svd", "kernel_pca"]
    G_types = []#, "targetvec", "GEP", "GEP_weighted"]
    config.lambda_gen_eigen = 0.00001
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