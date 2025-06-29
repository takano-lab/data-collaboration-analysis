from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from config.config import Config
from src.utils import reduce_dimensions_with_svd

logger = TypeVar("logger")


class DataCollaborationAnalysis:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: Config, logger: logger) -> None:
        self.config: Config = config
        self.logger = logger

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
            y_name=self.config.y_name,
        )

        # アンカーデータの生成
        self.anchor = self.produce_anchor(
            num_row=self.config.num_anchor_data, num_col=self.Xs_train[0].shape[1], seed=self.config.seed
        )
        print("num_row", self.config.num_anchor_data, "num_col", self.Xs_train[0].shape[1])
        print("anchor", self.anchor)
        print("Xs_train[0].shape", self.Xs_train[0].shape, "Xs_test[0].shape", self.Xs_test[0].shape)
        # 中間表現の生成
        self.make_intermediate_expression()

        # 統合表現の生成
        self.make_integrate_expression()
        #self.make_integrate_expression_kawakami_suetake()

    @staticmethod
    # この関数外に出したい
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, num_institution: int, num_institution_user: int, y_name: str = "target"
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        print("********************データの分割********************")
        """
        複数機関を想定してデータセットを分割する関数
        """
        
        train_df = train_df.copy()[:num_institution * num_institution_user]
        test_df = test_df.copy()[:num_institution * num_institution_user]
        y_train_ser = train_df[y_name]
        X_train_df = train_df.drop(y_name, axis=1)
        y_test_ser = test_df[y_name]
        X_test_df = test_df.drop(y_name, axis=1)

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
            # tempを1つのarrayに変換し、リストに格納
            Xs_train.append(X_train_df[institute_start:institute_start + num_institution_user].values)
            Xs_test.append(X_test_df[institute_start:institute_start + num_institution_user].values)

            # yはtemp_train_xに対応するratingを格納
            ys_train.append(y_train_ser[institute_start:institute_start + num_institution_user].values)
            ys_test.append(y_test_ser[institute_start:institute_start + num_institution_user].values)

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
        print(self.config)
        print("self.config.dim_intermediate", self.config.dim_intermediate)
        print()
        for X_train, X_test in zip(tqdm(self.Xs_train), self.Xs_test):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用
            X_train_svd, X_test_svd, anchor_svd = reduce_dimensions_with_svd(
               X_train=X_train,
               X_test=X_test,
               n_components=self.config.dim_intermediate,
               anchor=self.anchor,
            )
            
            # そのままで実験
            # X_train_svd = X_train
            # X_test_svd = X_test
            # anchor_svd = self.anchor
            
            # svdを適用したデータをリストに格納
            self.Xs_train_inter.append(X_train_svd)
            self.Xs_test_inter.append(X_test_svd)
            self.anchors_inter.append(anchor_svd)

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])
        
        self.logger.info(f"中間表現（訓練データ）の数と次元数: {self.Xs_train_inter[0].shape}")
        self.logger.info(f"中間表現（テストデータ）の数と次元数: {self.Xs_test_inter[0].shape}")


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
        for X_train_inter, X_test_inter, anchor_inter in zip(
            tqdm(self.Xs_train_inter), self.Xs_test_inter, self.anchors_inter
        ):
            # 各機関のアンカーデータの中間表現を転置して、擬似逆行列を求める
            pseudo_inverse = np.linalg.pinv(anchor_inter.T)  # \hat{X}^{anc}+

            # 各機関の統合関数を求める
            integrate_function = np.dot(Z, pseudo_inverse)  # G^{(i)}

            # 統合関数で各機関の中間表現を統合表現に変換
            X_train_integrate = np.dot(integrate_function, X_train_inter.T)
            X_test_integrate = np.dot(integrate_function, X_test_inter.T)

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

    def make_integrate_expression_2(self) -> None:
        print("********************統合表現の生成********************")
        """
        統合表現を生成する関数
        """
        from scipy.linalg import pinv, qr, solve_triangular, svd
        from sklearn.utils.extmath import randomized_svd
        U, _, _ =  randomized_svd(np.hstack(self.anchors_inter), n_components=self.config.dim_intermediate)
        # Compute Intermediate Representations of the anchor data

        G_list_Imakura = [pinv(A) @ U for A in self.anchors_inter]
        # アンカーデータを水平方向に開く（アンカーデータ数 × 各機関の中間表現次元の合計）
        self.X_train_integ = np.hstack([self.Xs_train_inter @ G_list_Imakura[i] for i in range(self.config.num_institution)])
        self.X_test_integ = np.hstack([self.Xs_test_inter @ G_list_Imakura[i] for i in range(self.config.num_institution)])

        # yもくっつける
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)

        # logにも出力
        self.logger.info(f"統合表現（訓練データ）の数と次元数: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テストデータ）の数と次元数: {self.X_test_integ.shape}")

    def make_integrate_expression_kawakami_yanagi(self) -> None:
        print("********************統合表現の生成********************")
        # 解析対象の行列構築
        N = self.config.dim_intermediate * (self.config.num_institution)
        Q = np.zeros((self.config.num_anchor_data, N))
        R = np.zeros((N, N))
        for institution in tqdm(range(self.config.num_institution)):
            # ある機関のアンカーデータを取得
            anchor = self.anchors_inter[institution]
            # QR分解
            q, r = linalg.qr(anchor, mode="economic")
            # 解析対象の行列に代入
            base = self.config.dim_intermediate * institution
            Q[:, base : base + self.config.dim_intermediate] = q
            R[base : base + self.config.dim_intermediate, base : base + self.config.dim_intermediate] = linalg.inv(r)

        # start = time.time()  # 測定開始
        try:
            U, s, V = linalg.svd(Q, full_matrices=False)
        except linalg.LinAlgError as e:
            U, s, V = linalg.svd(Q, full_matrices=False, lapack_driver="gesvd")
        l = (-2 * np.square(s[: self.config.dim_intermediate])) + (2 * self.config.num_institution)
        v = R @ V.T[:, : self.config.dim_intermediate]

        # 各機関の統合関数を求め、統合表現を生成
        Xs_train_integrate, Xs_test_integrate = [], []
        for (
            i,
            X_train_inter,
            X_test_inter,
        ) in zip(tqdm(range(self.config.num_institution)), self.Xs_train_inter, self.Xs_test_inter):
            # 統合関数はvから取ってくる
            integrate_function = v[self.config.dim_intermediate * i : self.config.dim_intermediate * (i + 1), :]

            # 統合関数で各機関の中間表現を統合表現に変換
            print("X_train_inter shape:", X_train_inter.shape)
            X_train_integrate = np.dot(X_train_inter, integrate_function)
            X_test_integrate = np.dot(X_test_inter, integrate_function)

            # 統合表現をリストに格納
            Xs_train_integrate.append(X_train_integrate)
            Xs_test_integrate.append(X_test_integrate)

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

    def make_integrate_expression_kawakami_suetake(self) -> None:
        import numpy as np
        from numpy.linalg import eigh  # 対称行列用の固有分解
        print("********************統合表現の生成********************")

        # ===================================================
        # 1. C_{~S} を作る
        # ===================================================
        #   行  : アンカーデータ（r 個）
        #   列  : 各機関の中間表現を横に並べたもの
        #         └ shape = (r, dim_intermediate * num_institution)
        # ---------------------------------------------------
        r  = self.config.num_anchor_data
        dI = self.config.dim_intermediate
        c  = self.config.num_institution
        N  = dI * c                          # 全体の特徴次元

        Q = np.zeros((r, N))                 # r×N 行列
        for inst_idx, anchor in enumerate(self.anchors_inter):
            # anchor : shape = (r, dim_intermediate)
            col_from = inst_idx * dI
            col_to   = (inst_idx + 1) * dI
            Q[:, col_from:col_to] = anchor

        # （必要なら）中心化しておく
        Q -= Q.mean(axis=0, keepdims=True)

        # Kawakami-Suetake の元論文では R を使った一般化固有値問題
        # が出てきますが，R=I と置くと C_{~S}=Q^T Q だけで十分
        C_tildeS = Q.T @ Q                  # N×N の対称正定値行列

        # ===================================================
        # 2. 固有値問題 C_{~S} z = λ z を解く
        # ===================================================
        eigvals, eigvecs = eigh(C_tildeS)    # 昇順で返る
        idx = eigvals.argsort()[::-1]        # 降順に並び替え
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]            # shape = (N, N)

        # 必要な次元数だけ取り出す（例：共通表現の次元 m̂）
        m_hat = self.config.dim_integrate
        v = eigvecs[:, :m_hat]               # shape = (N, m̂)
        # eigh で得た固有ベクトルは既に ||z||_2=1 に正規化済み

        # ===================================================
        # 3. 各機関ごとの統合関数を切り出して統合表現を生成
        # ===================================================
        Xs_train_integrate, Xs_test_integrate = [], []

        for i, X_train_inter, X_test_inter in zip(
                range(c), self.Xs_train_inter, self.Xs_test_inter):

            integrate_function = v[dI * i : dI * (i + 1), :]   # (dI, m̂)

            X_train_integrate = X_train_inter @ integrate_function
            X_test_integrate  = X_test_inter  @ integrate_function

            Xs_train_integrate.append(X_train_integrate)
            Xs_test_integrate.append(X_test_integrate)

        # スタックして共通表現を完成
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)
        
        # そのままで実験
        # self.X_train_integ = np.vstack(self.Xs_train)
        # self.X_test_integ  = np.vstack(self.Xs_test)
        # self.y_train_integ = np.hstack(self.ys_train)
        # self.y_test_integ  = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")
