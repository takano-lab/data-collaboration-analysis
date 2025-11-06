from __future__ import annotations

import argparse
import statistics
from logging import INFO, FileHandler, getLogger
from pathlib import Path

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


def _safe_preserve_name(name: object) -> str:
    if not name:
        return "unnamed"
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    slug = "_".join(filter(None, slug.split("_")))
    return slug or "unnamed"


def _preserved_df_path(config: Config) -> Path | None:
    name = getattr(config, "df_name", None)
    if not name:
        return None
    safe = _safe_preserve_name(name)
    return OUTPUT_DIR / "preserved_df" / "df" / f"{safe}.pkl"


def _has_preserved_df(config: Config) -> bool:
    path = _preserved_df_path(config)
    if not path:
        return False
    try:
        return path.exists()
    except Exception:
        return False

def run_once(config, logger):
    logger.info(f"データセット: {config.dataset}")
    
    load_preserve = bool(getattr(config, "load_df_data", False))
    preserved_available = _has_preserved_df(config) if load_preserve else False
    
    # 保存済みデータの読み込み or 新規データ読み込み＋機関データ変換
    if load_preserve and preserved_available:
        Xs_train, Xs_test, ys_train, ys_test, train_df, test_df = [], [], [], [], [], []
        
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
        data_collaboration.load_existing_df_data()
    # 機関データの生成
    else:
        # datasetの読み込み
        # 1. 前処理まで（単一 df）
        logger.info("データ新規読み込み中...")
        df = load_data(config=config)
        # 2. 機関データへ変換 (内部で列制限/機関数補完/ train-test split / even|division)
        Xs_train, Xs_test, ys_train, ys_test, train_df, test_df = prepare_institutional_dataset(df, config)
    
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
    
    metrics_dict = {}

    if config.G_type == 'centralize':
        # 集中解析
        metrics_cen = centralize_analysis(
            config=config, 
            logger=logger, 
            train_df=data_collaboration.train_df, 
            test_df=data_collaboration.test_df,
            y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        return metrics_cen
    
    elif config.G_type == 'centralize_dim':
        # 集中解析 with 次元削減
        metrics_cen_dim = centralize_analysis(
            config=config, 
            logger=logger, 
            train_df=data_collaboration.train_df, 
            test_df=data_collaboration.test_df,
            y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
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
        return metrics_fl
    
    # 統合解析
    else:
        data_collaboration.run()
        
        if config.visualize:
            # 新しい可視化クラス経由で表示/保存
            viz = DataCollabVisualizer(data_collaboration, logger)
            viz.visualize_representations()
            
        config.f_seed = 0

        n_inst = config.num_institution

        inst_losses = []
        
        config.f_seed = 0
        for i in range(n_inst):
            try:
                metric_i = dca_analysis(
                    X_train_integ=np.vstack(data_collaboration.Xs_train_integ), 
                    X_test_integ=data_collaboration.Xs_test_integ[i], 
                    y_train_integ=np.hstack(data_collaboration.ys_train_integ), 
                    y_test_integ=data_collaboration.ys_test_integ[i], 
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

        # --- 集計 ---
        inst_losses_arr = np.array(inst_losses, dtype=float)
        valid_mask = ~np.isnan(inst_losses_arr)
        if valid_mask.any():
            mean_val = float(inst_losses_arr[valid_mask].mean())
            min_val = float(inst_losses_arr[valid_mask].min())
            max_val = float(inst_losses_arr[valid_mask].max())
        else:
            mean_val = min_val = max_val = float("nan")

        logger.info(f"評価値: {mean_val}")
        logger.info(f"機関ごとの {config.metrics}: {np.round(inst_losses_arr, 4).tolist()}")
        logger.info(f"平均: {mean_val:.4f}, 最小: {min_val:.4f}, 最大: {max_val:.4f}")

        return mean_val
        # --- 機関ごとの「学習データの最頻ラベルに基づく評価」リストを算出して表示 ---
        """         
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


            logger.info(f"各機関の最頻ラベルに基づくスコア: {per_inst_major_label_scores}")
        except Exception as e:
            # ここはログ/表示のみ（失敗しても主計算には影響させない）
            try:
                import traceback

                traceback.print_exc()
            except Exception:
                pass """
