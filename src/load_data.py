from __future__ import annotations

from pathlib import Path

import category_encoders as ce
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import Config


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("********************データの読み込み********************")
    if config.dataset == "movielens":
        # inputフォルダのmovielensデータを読み込み
        train_df = pd.read_csv(
            config.input_path / "ua.base",
            names=["uid", "mid", "rating", "timestamp"],
            sep="\t",
            dtype=int,
        )
        test_df = pd.read_csv(
            config.input_path / "ua.test",
            names=["uid", "mid", "rating", "timestamp"],
            sep="\t",
            dtype=int,
        )

        # timestampを削除
        train_df = train_df.drop(["timestamp"], axis=1)
        test_df = test_df.drop(["timestamp"], axis=1)

    elif config.dataset == "sushi":
        # inputフォルダのsushiデータを読み込み
        df = pd.read_csv(config.input_path / "osushi.csv")

        # trainとtestに分割
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=config.seed, stratify=df["uid"])
        # uidでソート
        train_df = train_df.sort_values(by="uid").reset_index(drop=True)
        test_df = test_df.sort_values(by="uid").reset_index(drop=True)

    # data保存
    save_data(
        dataset=config.dataset,
        num_institution=config.num_institution,
        num_institution_user=config.num_institution_user,
        output_path=config.output_path,
        train_df=train_df,
        test_df=test_df,
    )

    # ratingの列を取り出し
    train_rating_ser = train_df["rating"]
    test_rating_ser = test_df["rating"]

    # rating削除
    train_df = train_df.drop(["rating"], axis=1)
    test_df = test_df.drop(["rating"], axis=1)

    # onehotencoderを適用
    encoder = ce.OneHotEncoder(cols=["uid", "mid"])
    train_df = encoder.fit_transform(train_df)
    test_df = encoder.transform(test_df)

    # ratingを結合
    train_df = pd.concat([train_df, train_rating_ser], axis=1)
    test_df = pd.concat([test_df, test_rating_ser], axis=1)

    # 行数表示
    print("train test shape:", train_df.shape, test_df.shape)

    return train_df, test_df


def save_data(
    dataset: str,
    num_institution: int,
    num_institution_user: int,
    output_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    # configのnum_institutionとnum_institution_userの積を計算
    # (sushiはuidが0始まりなことに注意)
    if dataset == "movielens":
        max_uid = num_institution * num_institution_user
    elif dataset == "sushi":
        max_uid = num_institution * num_institution_user - 1

    # train, testそれぞれでuidがmax_uid以下のものだけを抽出
    train_df = train_df[train_df["uid"] <= max_uid].reset_index(drop=True)
    test_df = test_df[test_df["uid"] <= max_uid].reset_index(drop=True)

    # train, testをcsvに書き出し
    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
