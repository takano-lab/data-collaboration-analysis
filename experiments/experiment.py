from __future__ import annotations

import argparse
import statistics
from logging import INFO, FileHandler, getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from config.config import Config
from config.config_logger import record_config_to_cfg, record_value_to_cfg
from src.institution_data_pipeline import InstitutionDatasetBuilder
from src.intermediate_expression import IntermediateExpressionBuilder
from src.integrated_expression import IntegratedExpressionBuilder
from src.institutional_analysis import (
    centralize_analysis,
    centralize_analysis_with_dimension_reduction,
    centralize_analysis_with_institution_dimension_reduction,
    dca_analysis,
    fl_analysis,
    individual_analysis,
    individual_analysis_with_dimension_reduction,
)
from src.model import ModelRunner
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR
from src.integrated_expression.visualization import DataCollabVisualizer


def run_once(config, logger):
    logger.info(f"データセット: {config.dataset}")

    dataset_builder = InstitutionDatasetBuilder(config=config, logger=logger)
    dataset_artifacts = dataset_builder.run()
    data_collaboration = dataset_builder

    metrics_dict = {}

    if config.G_type == 'centralize':
        metrics_cen = centralize_analysis(
            config=config, 
            logger=logger, 
            train_df=dataset_builder.train_df, 
            test_df=dataset_builder.test_df,
            y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        return metrics_cen
    
    elif config.G_type == 'centralize_dim':
        metrics_cen_dim = centralize_analysis(
            config=config, 
            logger=logger, 
            train_df=dataset_builder.train_df, 
            test_df=dataset_builder.test_df,
            y_name=config.y_name)
        metrics_dict['centralize'] = metrics_cen
        return metrics_cen_dim
    
    elif config.G_type == 'individual':
        metrics_ind = individual_analysis(
            config=config,
            logger=logger,
            Xs_train=dataset_builder.Xs_train,
            ys_train=dataset_builder.ys_train,
            Xs_test=dataset_builder.Xs_test,
            ys_test=dataset_builder.ys_test,
        )
        return metrics_ind
    
    elif config.G_type == 'individual_dim':
        metrics_ind_dim = individual_analysis_with_dimension_reduction(
            config=config,
            logger=logger,
            Xs_train=dataset_builder.Xs_train,
            ys_train=dataset_builder.ys_train,
            Xs_test=dataset_builder.Xs_test,
            ys_test=dataset_builder.ys_test,
        )
        return metrics_ind_dim
    
    elif config.G_type == 'fl':
        metrics_fl = fl_analysis(
            config=config,
            logger=logger,
            Xs_train=dataset_builder.Xs_train,
            ys_train=dataset_builder.ys_train,
            Xs_test=dataset_builder.Xs_test,
            ys_test=dataset_builder.ys_test,
        )
        metrics_dict['fl'] = metrics_fl["mean"]
        metrics_dict['institutions'] = metrics_fl["per_institution"]
        return metrics_fl["mean"]
    
    else:
        intermediate_builder = IntermediateExpressionBuilder(config=config, logger=logger)
        intermediate_artifacts = intermediate_builder.run(dataset_artifacts)

        data_collaboration = IntegratedExpressionBuilder(config=config, logger=logger)
        integrated_artifacts = data_collaboration.run(intermediate_artifacts)

        if config.visualize:
            viz = DataCollabVisualizer(config=config, artifacts=integrated_artifacts, logger=logger)
            viz.visualize_representations()
            if getattr(config, "visualize_for_presenations", False):
                viz.visualize_anchors_for_presenations()

        actual_inst = len(getattr(data_collaboration, "Xs_test_integ", []) or [])
        if actual_inst == 0:
            logger.warning("No integrated test splits were produced; skipping DCA evaluation.")
            return float("nan")
        if actual_inst != config.num_institution:
            logger.warning(
                "Configured num_institution=%s but integrated outputs have %s institutions. Using actual count.",
                config.num_institution,
                actual_inst,
            )

        inst_losses = []

        for i in range(actual_inst):
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
                logger.warning(f"dca_analysis 実行 inst={i}: {e} → NaN を採用")
                metric_i = np.nan
            except Exception as e:
                logger.exception(f"dca_analysis 失敗 inst={i}: {e}")
                metric_i = np.nan

            inst_losses.append(metric_i)

        inst_losses_arr = np.array(inst_losses, dtype=float)
        valid_mask = ~np.isnan(inst_losses_arr)
        if valid_mask.any():
            mean_val = float(inst_losses_arr[valid_mask].mean())
            min_val = float(inst_losses_arr[valid_mask].min())
            max_val = float(inst_losses_arr[valid_mask].max())
        else:
            mean_val = min_val = max_val = float("nan")

        logger.info(f"平均評価値: {mean_val}")
        logger.info(f"各機関の {config.metrics}: {np.round(inst_losses_arr, 4).tolist()}")
        logger.info(f"平均 {mean_val:.4f}, 最小 {min_val:.4f}, 最大 {max_val:.4f}")

        return mean_val
