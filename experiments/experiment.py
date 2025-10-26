from __future__ import annotations

import argparse
import statistics
from logging import INFO, FileHandler, getLogger

import numpy as np  # 追加
import pandas as pd
import yaml
from tqdm import tqdm

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.data_collaboration import DataCollaborationAnalysis
from src.institution_data import prepare_institutional_dataset  # 新しい機関データ生成
from src.institutional_analysis import (
    centralize_analysis,
    centralize_analysis_with_dimension_reduction,
    centralize_analysis_with_institution_dimension_reduction,
    dca_analysis,
    fl_analysis,
    individual_analysis,
    individual_analysis_with_dimension_reduction,
)
from src.load_data import load_data
from src.model import ModelRunner
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR
from src.visualization import DataCollabVisualizer

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
    # 1. 前処理まで（単一 df）
    df = load_data(config=config)
    # 2. 機関データへ変換 (内部で列制限/機関数補完/ train-test split / even|division)
    Xs_train, Xs_test, ys_train, ys_test, train_df, test_df = prepare_institutional_dataset(df, config)
    
    metrics_dict = {}
    
    if config.F_type == "kernel_pca" and config.G_type == "GEP_weighted":
        # GEP_weightedはUSE_KERNELがTrueのときのみ実行
        return
    #if config.F_type == "kernel_pca" and config.G_type == "GEP":
        # GEP_weightedはUSE_KERNELがTrueのときのみ実行
    #    return
    config.log(logger, exclude_keys=["output_path", "input_path", "name", "seed", "y_name"])
    if config.G_type != "centralize":
        # インスタンスの生成
        data_collaboration = DataCollaborationAnalysis(
            config=config,
            logger=logger,
            train_df=train_df,
            test_df=test_df,
            Xs_train=Xs_train,
            Xs_test=Xs_test,
            ys_train=ys_train,
            ys_test=ys_test,
        )
        # データ分割 -> 統合表現の獲得まで一気に実行
        #data_collaboration.save_optimal_params()
        data_collaboration.run()
        if config.visualize:
            # 新しい可視化クラス経由で表示/保存
            viz = DataCollabVisualizer(data_collaboration, logger)
            viz.visualize_representations()
            print(1111)
        #data_collaboration.save_representations_to_csv()
            # 提案手法
        #record_config_to_cfg(config)
    if config.G_type == 'centralize':
        # 集中解析
        print(22222222222)
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
        print(11111111111)
        config.f_seed = 0
        if getattr(config, "data_distribution", None) == "division":
            skip_individual = True
        else:
            skip_individual = False
        if not skip_individual:
            metrics_ind_dim = individual_analysis_with_dimension_reduction(
                config=config,
                logger=logger,
                Xs_train=data_collaboration.Xs_train,
                ys_train=data_collaboration.ys_train,
                Xs_test=data_collaboration.Xs_test,
                ys_test=data_collaboration.ys_test,
            )
        
        else:
            metrics_ind_dim = {config.metrics: np.nan}
            metrics_centralized_dims = []

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
        # --- ここから機関ごとの metrics を算出 ---
        n_inst = config.num_institution

        division_mode = getattr(config, "data_distribution", None) == "division"
        if division_mode:
            # division: テストは全機関共通セット（分割不要）
            test_cum = None
        else:
            # even: 各機関ごとに独立テストをスライス
            test_counts = [len(y) for y in data_collaboration.ys_test]
            test_cum = np.concatenate(([0], np.cumsum(test_counts)))
        print("division_mode", division_mode)
        if division_mode:
            pass
            # for i in range(n_inst):
            #     metrics_centralized_dim = centralize_analysis_with_institution_dimension_reduction(
            #                             X_train=data_collaboration.train_df.drop(config.y_name, axis=1),
            #                             X_test=data_collaboration.test_df.drop(config.y_name, axis=1),
            #                             y_train=data_collaboration.train_df[config.y_name],
            #                             y_test=data_collaboration.test_df[config.y_name],
            #                             X_train_reduction=data_collaboration.Xs_train[i],   
            #                             config=config,
            #                             logger=logger,
            #                         )
            #     metrics_centralized_dims.append(metrics_centralized_dim)
            # print("機関ごとの次元削減スコア")
            # print(metrics_centralized_dims)

        inst_losses = []
        even_losses = []
        odd_losses = []

        config.f_seed = 0
        for i in range(n_inst):
            if division_mode:
                # 全機関同一テストセットをそのまま使う
                X_te_i = data_collaboration.X_test_integ
                y_te_i = data_collaboration.y_test_integ
            
            else:
                # 機関 i のテスト範囲
                te_start, te_end = int(test_cum[i]), int(test_cum[i + 1])
                # 希望件数を超えないように調整
                te_take = min(config.num_institution_user, te_end - te_start)
                if te_take <= 0:
                    # データ不足 → NaN
                    inst_losses.append(np.nan)
                    (even_losses if i % 2 == 0 else odd_losses).append(np.nan)
                    continue
                X_te_i = data_collaboration.X_test_integ[te_start : te_start + te_take, :]
                y_te_i = data_collaboration.y_test_integ[te_start : te_start + te_take]

            try:
                metric_i = dca_analysis(
                    X_train_integ=data_collaboration.X_train_integ,
                    X_test_integ=X_te_i,
                    y_train_integ=data_collaboration.y_train_integ,
                    y_test_integ=y_te_i,
                    config=config,
                    logger=logger,
                )
            except ValueError as e:
                logger.warning(f"dca_analysis 失敗 inst={i}: {e} → NaN で継続")
                metric_i = np.nan
            except Exception as e:
                logger.exception(f"dca_analysis 予期せぬ例外 inst={i}: {e}")
                metric_i = np.nan

            inst_losses.append(metric_i)
            if i % 2 == 0:
                even_losses.append(metric_i)
            else:
                odd_losses.append(metric_i)

        # --- 集計 ---
        inst_losses_arr = np.array(inst_losses, dtype=float)
        valid_mask = ~np.isnan(inst_losses_arr)
        if valid_mask.any():
            mean_val = float(inst_losses_arr[valid_mask].mean())
            min_val = float(inst_losses_arr[valid_mask].min())
            max_val = float(inst_losses_arr[valid_mask].max())
        else:
            mean_val = min_val = max_val = float("nan")

        config.losses_even = round(float(np.nanmean(even_losses)), 4) if even_losses else np.nan
        config.losses_odd = round(float(np.nanmean(odd_losses)), 4) if odd_losses else np.nan
        config.losses_mean = round(mean_val, 4) if not np.isnan(mean_val) else np.nan

        print("評価値2", mean_val)
        print("config.losses_mean", config.losses_mean)
        print(f"機関ごとの {config.metrics}: {np.round(inst_losses_arr, 4).tolist()}")
        print(f"平均: {mean_val:.4f}, 最小: {min_val:.4f}, 最大: {max_val:.4f}")
        logger.info(f"機関ごとの {config.metrics}: {inst_losses_arr.tolist()}")
        logger.info(f"平均: {mean_val:.6f}, 最小: {min_val:.6f}, 最大: {max_val:.6f}")

        # --- 機関ごとの「学習データの最頻ラベルに基づく評価」リストを算出して表示 ---
        try:
            per_inst_major_label_scores = []
            runner = ModelRunner(config)
            for i in range(n_inst):
                # その機関の学習データで最頻ラベルを特定
                y_train_i = data_collaboration.ys_train[i]
                if len(y_train_i) == 0:
                    per_inst_major_label_scores.append(np.nan)
                    continue
                # 最頻値
                values, counts = np.unique(y_train_i, return_counts=True)
                major_label = values[np.argmax(counts)]

                # テストデータ（division/even）を取得
                if division_mode:
                    X_te_i = data_collaboration.X_test_integ
                    y_te_i = data_collaboration.y_test_integ
                else:
                    te_start, te_end = int(test_cum[i]), int(test_cum[i + 1])
                    te_take = min(config.num_institution_user, te_end - te_start)
                    if te_take <= 0:
                        per_inst_major_label_scores.append(np.nan)
                        continue
                    X_te_i = data_collaboration.X_test_integ[te_start : te_start + te_take, :]
                    y_te_i = data_collaboration.y_test_integ[te_start : te_start + te_take]

                # 学習済みモデルで予測（DCA と同じ学習データを用いる）
                y_pred_i, y_proba_i, classes_i = runner.predict_with_proba(
                    data_collaboration.X_train_integ,
                    data_collaboration.y_train_integ,
                    X_te_i,
                )

                metric_name = str(getattr(config, 'metrics', 'auc')).lower()
                if metric_name == 'auc':
                    # 1-vs-rest AUC をその最頻ラベルに対して計算
                    # 該当クラスの列を抽出
                    try:
                        # classes_i は元ラベルの配列
                        if major_label not in classes_i:
                            per_inst_major_label_scores.append(np.nan)
                            continue
                        cls_idx = int(np.where(classes_i == major_label)[0][0])
                        y_true_bin = (y_te_i == major_label).astype(int)
                        # 正例と負例の両方がないと AUC は定義されない
                        if len(np.unique(y_true_bin)) < 2:
                            per_inst_major_label_scores.append(np.nan)
                            continue
                        from sklearn.metrics import roc_auc_score
                        score_i = float(roc_auc_score(y_true_bin, y_proba_i[:, cls_idx]))
                    except Exception as e:
                        score_i = np.nan
                elif metric_name == 'accuracy':
                    # 「最頻ラベルに絞った」= そのラベルのテストサンプルに対する正解率（=再現率）
                    mask = (y_te_i == major_label)
                    if mask.sum() == 0:
                        score_i = np.nan
                    else:
                        score_i = float(np.mean(y_pred_i[mask] == y_te_i[mask]))
                else:
                    # 未対応の評価指標は簡易に再現率で代替
                    mask = (y_te_i == major_label)
                    if mask.sum() == 0:
                        score_i = np.nan
                    else:
                        score_i = float(np.mean(y_pred_i[mask] == y_te_i[mask]))

                per_inst_major_label_scores.append(score_i)

            print(f"各機関の最頻ラベルに基づくスコア: {np.round(per_inst_major_label_scores, 4).tolist()}")
            logger.info(f"各機関の最頻ラベルに基づくスコア: {per_inst_major_label_scores}")
        except Exception as e:
            # ここはログ/表示のみ（失敗しても主計算には影響させない）
            try:
                import traceback
                print(f"[WARN] 最頻ラベルスコア算出に失敗: {e}")
                traceback.print_exc()
            except Exception:
                pass

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

