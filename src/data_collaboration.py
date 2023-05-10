from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from config.config import Config

logger = TypeVar("logger")


class DataCollaborationAnalysis:
    def __init__(self, config: Config, logger: logger, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.config: Config = config
        self.logger: logger = logger

        # 本当はできるだけattributeを持たせない方が良い
        # 元データ
        self.train_df: pd.DataFrame = train_df
        self.test_df: pd.DataFrame = test_df
        self.anchor: np.ndarray = np.array([])

        # 機関ごとの分割データ
        self.Xs_train: list[np.ndarray] = []
        self.Xs_test: list[np.ndarray] = []
        self.ys_train: list[np.ndarray] = []
        self.ys_test: list[np.ndarray] = []

        # 中間表現
        self.anchors_inter: list[np.ndarray] = []
        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []
        self.ys_train_inter: list[np.ndarray] = []
        self.ys_test_inter: list[np.ndarray] = []

        # 統合表現
        self.X_train_integ: np.ndarray = np.array([])
        self.X_test_integ: np.ndarray = np.array([])
        self.y_train_integ: np.ndarray = np.array([])
        self.y_test_integ: np.ndarray = np.array([])

    def run(self) -> None:
        """
        データ分割、中間表現の生成、統合表現の生成を一気に行う関数
        """
        # データの分割
        self.Xs_train, self.Xs_test, self.ys_train, self.ys_test = self.train_test_split(
            train_df=self.train_df,
            test_df=self.test_df,
            num_institution=self.config.num_institution,
            num_institution_user=self.config.num_institution_user,
        )

        # アンカーデータの生成
        self.anchor = self.produce_anchor(
            num_row=self.config.num_anchor_data, num_col=self.Xs_train[0].shape[1], seed=self.config.seed
        )

        # 中間表現の生成
        self.make_intermediate_expression()

        # 統合表現の生成
        self.make_integrate_expression()

    @staticmethod
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, num_institution: int, num_institution_user: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        print("********************データの分割********************")
        """
        複数機関を想定してデータセットを分割する関数
        """
        y_train_ser = train_df["rating"]
        X_train_df = train_df.drop("rating", axis=1)
        y_test_ser = test_df["rating"]
        X_test_df = test_df.drop("rating", axis=1)

        # 格納しておくリスト
        Xs_train, Xs_test = [], []
        ys_train, ys_test = [], []

        # データセットを分割する
        for institute_start in tqdm(
            range(
                0,
                num_institution * num_institution_user,
                num_institution_user,
            )
        ):
            # 一時的に格納する変数
            X_train_df_tmp, X_test_df_tmp = pd.DataFrame(), pd.DataFrame()

            for user_id in range(institute_start + 1, institute_start + num_institution_user + 1):  # user_idは1から始まる
                # trainのuser_idが一致する箇所のindexを抽出
                index_train = X_train_df[f"uid_{user_id}"] == 1
                # testのuser_idが一致する箇所のindexを抽出
                index_test = X_test_df[f"uid_{user_id}"] == 1

                # train_index, test_indexに一致するデータを抽出
                X_train_df_tmp = pd.concat([X_train_df_tmp, X_train_df[index_train]], axis=0)
                X_test_df_tmp = pd.concat([X_test_df_tmp, X_test_df[index_test]], axis=0)

            # tempを1つのarrayに変換し、リストに格納
            Xs_train.append(X_train_df_tmp.values)
            Xs_test.append(X_test_df_tmp.values)

            # yはtemp_train_xに対応するratingを格納
            ys_train.append(y_train_ser[X_train_df_tmp.index].values)
            ys_test.append(y_test_ser[X_test_df_tmp.index].values)

        print("機関の数: ", len(Xs_train))

        return Xs_train, Xs_test, ys_train, ys_test

    @staticmethod
    def produce_anchor(num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
        """
        アンカーデータを生成する関数
        """
        np.random.seed(seed=seed)
        anchor = np.random.rand(num_row, num_col)
        return anchor

    def make_intermediate_expression(self) -> None:
        print("********************中間表現の生成********************")
        """
        中間表現を生成する関数
        """
        for institute in tqdm(range(self.config.num_institution)):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用
            X_train_svd, X_test_svd, anchor_svd = DataCollaborationAnalysis.svd(
                self.Xs_train[institute],
                self.Xs_test[institute],
                self.anchor,
                self.config.dim_intermediate,
            )
            # svdを適用したデータをリストに格納
            self.Xs_train_inter.append(X_train_svd)
            self.Xs_test_inter.append(X_test_svd)
            self.anchors_inter.append(anchor_svd)

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])

    def make_integrate_expression(self) -> None:
        print("********************統合表現の生成********************")
        """
        統合表現を生成する関数
        """
        # アンカーデータを水平方向に開く（アンカーデータ数 × 各機関の中間表現次元の合計）
        centralized_anchor = np.hstack(self.anchors_inter)  # \hat{X}^{anc}

        # 特異値分解（Uはアンカーデータ数 × 統合表現の次元数）
        U, _, _ = np.linalg.svd(centralized_anchor)
        U = U[:, : self.config.dim_integrate]  # 固有値の大きい順に統合表現の次元数だけ取得

        # Zは統合表現の次元数 × アンカーデータ数
        Z = U.T

        # 各機関の統合関数を求め、統合表現を生成
        Xs_train_integrate, Xs_test_integrate = [], []
        for institute in tqdm(range(self.config.num_institution)):
            # 各機関のアンカーデータの中間表現を転置して、擬似逆行列を求める
            pseudo_inverse = np.linalg.pinv(self.anchors_inter[institute].T)  # \hat{X}^{anc}+

            # 各機関の統合関数を求める
            integrate_function = np.dot(Z, pseudo_inverse)  # G^{(i)}

            # 統合関数で各機関の中間表現を統合表現に変換
            X_train_integrate = np.dot(integrate_function, self.Xs_train_inter[institute].T)
            X_test_integrate = np.dot(integrate_function, self.Xs_test_inter[institute].T)

            # 統合表現をリストに格納
            Xs_train_integrate.append(X_train_integrate.T)
            Xs_test_integrate.append(X_test_integrate.T)

        print("統合表現の次元数: ", Xs_train_integrate[0].shape[1])

        # 全ての機関の統合表現をくっつけ、1つのarrayに変換
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ = np.vstack(Xs_test_integrate)

        # yもくっつける
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)

        # logにも出力
        self.logger.info(f"統合表現（訓練データ）の数と次元数: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テストデータ）の数と次元数: {self.X_test_integ.shape}")
        self.logger.info(f"統合表現（訓練データの正解）の数と次元数: {self.y_train_integ.shape}")
        self.logger.info(f"統合表現（テストデータの正解）の数と次元数: {self.y_test_integ.shape}")

    @staticmethod
    def svd(
        X_train: np.ndarray, X_test: np.ndarray, anchor: np.ndarray, n_components: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        X_trainを基準にsvdを適用し、X_train, X_test, anchorを次元削減する関数
        """
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X_train)
        X_train_svd = svd.transform(X_train)
        X_test_svd = svd.transform(X_test)
        anchor_svd = svd.transform(anchor)
        return X_train_svd, X_test_svd, anchor_svd
