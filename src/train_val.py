from src.model import run_lgbm, run_surprise

# from src.model import run_fm
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np
import category_encoders as ce


# trainの1つをsvd
def svd(train_x, test_x, n_components):
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(train_x)
    svd_train_x = svd.transform(train_x)
    svd_test_x = svd.transform(test_x)
    return svd_train_x, svd_test_x


def centralize_analysis(config, logger):
    """
    集中解析を行う関数
    """
    # config['input_path']からtrain/testを読み込む
    train_df = pd.read_csv(config["output_path"] / "train.csv")
    test_df = pd.read_csv(config["output_path"] / "test.csv")

    # 先にsurpriseでの予測を行う
    rmse = run_surprise(
        config, train_df, test_df, neightbors=config["neighbors_centlize"]
    )
    logger.info("集中解析（協調フィルタリング）のRMSE: {}".format(rmse))

    # ratingをyとし、削除
    y_train = train_df["rating"]
    y_test = test_df["rating"]
    train_df.drop("rating", axis=1, inplace=True)
    test_df.drop("rating", axis=1, inplace=True)

    # 因子分解機のデータ形式に変換
    target_col = ["uid", "mid"]

    encoder = ce.OneHotEncoder(cols=target_col)

    X_train = encoder.fit_transform(train_df)
    X_test = encoder.transform(test_df)

    # 集中解析（lightgbm）
    svd_train_x, svd_test_x = svd(X_train, X_test, config["dim_integrate"])
    rmse = run_lgbm(config, svd_train_x, y_train, svd_test_x, y_test)
    logger.info("集中解析（lightgbm）のRMSE: {}".format(rmse))

    # 集中解析（FM）
    # rmse = run_fm(config, X_train, y_train, X_test, y_test)
    # logger.info("集中解析（FM）のRMSE: {}".format(rmse))


def individual_analysis(
    config, train_x_list, train_y_list, test_x_list, test_y_list, logger
):
    """
    個別解析を行う関数
    """
    # 個別解析（lightgbm）
    loss_list = []
    for i in range(config["num_institution"]):
        svd_train_x, svd_test_x = svd(
            train_x_list[i], test_x_list[i], config["dim_intermediate"]
        )
        rmse = run_lgbm(
            config, svd_train_x, train_y_list[i], svd_test_x, test_y_list[i]
        )
        loss_list.append(rmse)

    logger.info("個別解析（lightgbm）のRMSE: {}".format(np.mean(loss_list)))

    # 個別解析（FM）
    # loss_list = []
    # for i in range(len(train_x_list)):
    #     rmse = run_fm(config, train_x_list[i], train_y_list[i], test_x_list[i], test_y_list[i])
    #     loss_list.append(rmse)

    # logger.info("個別解析（FM）のRMSE: {}".format(np.mean(loss_list)))

    # 個別解析（協調フィルタリング）
    loss_list = []
    train_df = pd.read_csv(config["output_path"] / "train.csv")
    test_df = pd.read_csv(config["output_path"] / "test.csv")
    for institute in range(config["num_institution"]):
        # train/testをinstituteごとに抽出し、surpriseでの予測を行う
        train_df_institute = train_df.loc[
            (train_df["uid"].astype(int) > institute * config["num_institution_user"])
            & (
                train_df["uid"].astype(int)
                <= (institute + 1) * config["num_institution_user"]
            ),
            :,
        ]
        test_df_institute = test_df.loc[
            (test_df["uid"].astype(int) > institute * config["num_institution_user"])
            & (
                test_df["uid"].astype(int)
                <= (institute + 1) * config["num_institution_user"]
            ),
            :,
        ]
        rmse = run_surprise(
            config,
            train_df_institute,
            test_df_institute,
            neightbors=config["neighbors_individual"],
        )
        loss_list.append(rmse)

    logger.info("個別解析（協調フィルタリング）のRMSE: {}".format(np.mean(loss_list)))


def dca_analysis(
    config,
    integrate_train_x,
    integrate_test_x,
    integrate_train_y,
    integrate_test_y,
    logger,
):
    """
    提案手法（データ統合解析）を行う関数
    """
    # 提案手法（lightgbm）
    rmse = run_lgbm(
        config, integrate_train_x, integrate_train_y, integrate_test_x, integrate_test_y
    )
    logger.info("提案手法（lightgbm）のRMSE: {}".format(rmse))

    # 提案手法（FM）
    # rmse = run_fm(config, integrate_train_x, integrate_train_y, integrate_test_x, integrate_test_y)
    # logger.info("提案手法（FM）のRMSE: {}".format(rmse))
