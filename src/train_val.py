from __future__ import annotations

from typing import Optional, TypeVar

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from config.config import Config
from src.model import run_lgbm, run_surprise

# from src.model import run_fm

logger = TypeVar("logger")


# trainの1つをsvd
def svd(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int, anchor: Optional[np.ndarray] = None
) -> tuple[np.ndarray, ...]:
    """
    X_trainを基準にsvdを適用し、X_train, X_test, anchor(あれば)を次元削減する関数
    """
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X_train)
    X_train_svd = svd.transform(X_train)
    X_test_svd = svd.transform(X_test)

    if anchor is not None:
        anchor_svd = svd.transform(anchor)
        return X_train_svd, X_test_svd, anchor_svd
    else:
        return X_train_svd, X_test_svd


def centralize_analysis(config: Config, logger: logger) -> None:
    """
    集中解析を行う関数
    """
    # config['output_path']からtrain/testを読み込む
    train_df = pd.read_csv(config.output_path / "train.csv")
    test_df = pd.read_csv(config.output_path / "test.csv")

    # 先にsurpriseでの予測を行う
    rmse_surprise = run_surprise(
        dataset=config.dataset, train_df=train_df, test_df=test_df, neightbors=config.neighbors_centlize
    )
    logger.info(f"集中解析（協調フィルタリング）のRMSE: {rmse_surprise}")

    # ratingをyとし、削除
    y_train = train_df["rating"].values
    y_test = test_df["rating"].values
    train_df.drop("rating", axis=1, inplace=True)
    test_df.drop("rating", axis=1, inplace=True)

    # 因子分解機のデータ形式に変換
    target_cols = ["uid", "mid"]
    encoder = ce.OneHotEncoder(cols=target_cols)
    X_train = encoder.fit_transform(train_df)
    X_test = encoder.transform(test_df)

    # 集中解析（lightgbm）
    X_train_svd, X_test_svd = svd(X_train=X_train, X_test=X_test, n_components=config.dim_integrate)
    rmse_lgbm = run_lgbm(X_train=X_train_svd, y_train=y_train, X_test=X_test_svd, y_test=y_test, seed=config.seed)
    logger.info(f"集中解析（lightgbm）のRMSE: {rmse_lgbm}")

    # 集中解析（FM）
    # rmse = run_fm(config, X_train, y_train, X_test, y_test)
    # logger.info("集中解析（FM）のRMSE: {}".format(rmse))


def individual_analysis(
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_test: list[np.ndarray],
    ys_test: list[np.ndarray],
    config: Config,
    logger: logger,
) -> None:
    """
    個別解析を行う関数
    """
    # 個別解析（lightgbm）
    losses_lgbm: list[float] = []
    for X_train, X_test, y_train, y_test in zip(Xs_train, Xs_test, ys_train, ys_test):
        X_train_svd, X_test_svd = svd(X_train, X_test, config.dim_intermediate)
        rmse_lgbm = run_lgbm(X_train=X_train_svd, y_train=y_train, X_test=X_test_svd, y_test=y_test)
        losses_lgbm.append(rmse_lgbm)

    logger.info(f"個別解析（lightgbm）のRMSE: {np.mean(losses_lgbm)}")

    # 個別解析（FM）
    # loss_list = []
    # for i in range(len(train_x_list)):
    #     rmse = run_fm(config, train_x_list[i], train_y_list[i], test_x_list[i], test_y_list[i])
    #     loss_list.append(rmse)

    # logger.info("個別解析（FM）のRMSE: {}".format(np.mean(loss_list)))

    # 個別解析（協調フィルタリング）
    losses_sur: list[float] = []
    train_df = pd.read_csv(config.output_path / "train.csv")
    test_df = pd.read_csv(config.output_path / "test.csv")
    for institute in range(config.num_institution):
        # train/testをinstituteごとに抽出し、surpriseでの予測を行う
        train_df_institute = train_df.loc[
            (train_df["uid"].astype(int) > institute * config.num_institution_user)
            & (train_df["uid"].astype(int) <= (institute + 1) * config.num_institution_user),
            :,
        ]
        test_df_institute = test_df.loc[
            (test_df["uid"].astype(int) > institute * config.num_institution_user)
            & (test_df["uid"].astype(int) <= (institute + 1) * config.num_institution_user),
            :,
        ]
        rmse_sur = run_surprise(
            dataset=config.dataset,
            train_df=train_df_institute,
            test_df=test_df_institute,
            neightbors=config.neighbors_individual,
        )
        losses_sur.append(rmse_sur)

    logger.info(f"個別解析（協調フィルタリング）のRMSE: {np.mean(losses_sur)}")


def dca_analysis(
    X_train_integ: np.ndarray,
    X_test_integ: np.ndarray,
    y_train_integ: np.ndarray,
    y_test_integ: np.ndarray,
    logger: logger,
    seed: int = 42,
):
    """
    提案手法（データ統合解析）を行う関数
    """
    # 提案手法（lightgbm）
    rmse = run_lgbm(
        X_train=X_train_integ,
        y_train=y_train_integ,
        X_test=X_test_integ,
        y_test=y_test_integ,
        seed=seed,
    )
    logger.info(f"提案手法（lightgbm）のRMSE: {rmse}")

    # 提案手法（FM）
    # rmse = run_fm(config, integrate_train_x, integrate_train_y, integrate_test_x, integrate_test_y)
    # logger.info("提案手法（FM）のRMSE: {}".format(rmse))
