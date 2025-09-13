from __future__ import annotations

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

# # 引数の設定
# parser = argparse.ArgumentParser()
# parser.add_argument("name", type=str, default="exp001")
# args = parser.parse_args()

# # yaml のパスと出力先パス
# cfg_path    = CONFIG_DIR / f"{args.name}.yaml"
# output_path = OUTPUT_DIR / args.name

# # UTF-8 で読み込んで Config を生成
# with cfg_path.open(encoding="utf-8") as f:
#     cfg_dict = yaml.safe_load(f)

# config = Config(**cfg_dict,
#                 output_path=output_path,
#                 input_path=INPUT_DIR)

# # 出力ディレクトリ作成
# output_path.mkdir(parents=True, exist_ok=True)

# # ログの設定
# logger = getLogger(__name__)
# logger.setLevel(INFO)
# handler = FileHandler(filename=config.output_path / "result.log", encoding="utf-8")
# logger.addHandler(handler)

def run_once(config, logger):
    #logger.info(f"データセット: {config.dataset}")
    print(f"データセット:{config.dataset}")
    
    
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
    if config.visualize:
        data_collaboration.visualize_representations()
        print(1111)
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
        
        config.losses_even =  round(sum(even_losses)/len(even_losses), 4)
        config.losses_odd = round(sum(odd_losses)/len(odd_losses), 4)
        config.losses_mean = round(mean_val, 4)
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

