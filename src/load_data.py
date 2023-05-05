import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce


def load_data(config):
    print("********************データの読み込み********************")
    if config["dataset"] == "movielens":
        # inputフォルダのmovielensデータを読み込み
        train = pd.read_csv(
            config["input_path"] / "ua.base",
            names=["uid", "mid", "rating", "timestamp"],
            sep="\t",
            dtype=int,
        )
        test = pd.read_csv(
            config["input_path"] / "ua.test",
            names=["uid", "mid", "rating", "timestamp"],
            sep="\t",
            dtype=int,
        )

        # timestampを削除
        train = train.drop(["timestamp"], axis=1)
        test = test.drop(["timestamp"], axis=1)

    elif config["dataset"] == "sushi":
        # inputフォルダのsushiデータを読み込み
        all = pd.read_csv(config["input_path"] / "osushi.csv")
        # trainとtestに分割
        train, test = train_test_split(
            all, test_size=0.2, random_state=config["seed"], stratify=all["uid"]
        )
        # uidでソート
        train = train.sort_values(by="uid").reset_index(drop=True)
        test = test.sort_values(by="uid").reset_index(drop=True)

    # data保存
    save_data(config, train, test)

    # ratingの列を取り出し
    train_rating = np.array(train["rating"], dtype=float)
    test_rating = np.array(test["rating"], dtype=float)

    # rating削除
    train = train.drop(["rating"], axis=1)
    test = test.drop(["rating"], axis=1)

    # onehotencoderを適用
    encoder = ce.OneHotEncoder(cols=["uid", "mid"])
    train = encoder.fit_transform(train)
    test = encoder.transform(test)

    # 行数表示
    print("train test shape:", train.shape, test.shape)

    return train, test, train_rating, test_rating


def save_data(config, train, test):
    # configのnum_institutionとnum_institution_userの積を計算
    # (sushiはuidが0始まりなことに注意)
    if config["dataset"] == "movielens":
        max_uid = config["num_institution"] * config["num_institution_user"]
    elif config["dataset"] == "sushi":
        max_uid = config["num_institution"] * config["num_institution_user"] - 1

    # train, testそれぞれでuidがmax_uid以下のものだけを抽出
    train = train[train["uid"] <= max_uid].reset_index(drop=True)
    test = test[test["uid"] <= max_uid].reset_index(drop=True)

    # train, testをcsvに書き出し
    train.to_csv(config["output_path"] / "train.csv", index=False)
    test.to_csv(config["output_path"] / "test.csv", index=False)
