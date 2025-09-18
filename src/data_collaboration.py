from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh, inv, norm, pinv
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config.config import Config
from src.utils import reduce_dimensions, self_tuning_gamma

logger = TypeVar("logger")
import csv
from pathlib import Path

from config.timing import timed

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
        self.anchor_y: np.ndarray = np.array([])
        self.anchor_test: np.ndarray = np.array([])

        # 機関ごとの分割データ
        self.Xs_train: list[np.ndarray] = []
        self.Xs_test: list[np.ndarray] = []
        self.ys_train: list[np.ndarray] = []
        self.ys_test: list[np.ndarray] = []

        # 中間表現
        self.anchors_inter: list[np.ndarray] = []
        self.anchors_test_inter: list[np.ndarray] = []
        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []
        #self.ys_train_inter: list[np.ndarray] = []
        #self.ys_test_inter: list[np.ndarray] = []

        # 統合表現
        self.anchors_integ: list[np.ndarray] = []
        self.anchors_test_integ: list[np.ndarray] = []
        self.X_train_integ: np.ndarray = np.array([])
        self.X_test_integ: np.ndarray = np.array([])
        self.y_train_integ: np.ndarray = np.array([])
        self.y_test_integ: np.ndarray = np.array([])


        self.make_integrate_expression_gen_eig = timed(config=self.config)(
            self.make_integrate_expression_gen_eig
        )
        self.make_integrate_expression = timed(config=self.config)(
            self.make_integrate_expression
        )

    def save_optimal_params(self) -> None:
        """
        データ分割、中間表現の生成、統合表現の生成を一気に行う関数。
        各機関ごとに最適なparamをグリッドサーチし、CSVに保存する。
        """
        # データの分割
        self.Xs_train, self.Xs_test, self.ys_train, self.ys_test = self.train_test_split(
            train_df=self.train_df,
            test_df=self.test_df,
            num_institution=self.config.num_institution,
            num_institution_user=self.config.num_institution_user,
            y_name=self.config.y_name,
        )

        best_params = {}

        # 各機関に対してグリッドサーチ
        for i, (X_tr, X_te, y_tr, y_te) in enumerate(zip(self.Xs_train, self.Xs_test, self.ys_train, self.ys_test)):
            best_score = -float("inf")
            best_param = None

            for param in [1, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
                X_tr_svd, X_te_svd = reduce_dimensions(X_tr, X_te, n_components=self.config.dim_intermediate, param=param)
                score = h_ml_model(X_tr_svd, y_tr, X_te_svd, y_te, self.config)
                print(score, param)
                if score > best_score:  # 指標が大きいほど良い場合（例：ROC-AUC）
                    best_score = score
                    best_param = param

            best_params[i] = best_param
            print(f"Institution {i}: Best param = {best_param:.1e}, score = {best_score:.4f}")

        # 保存パスの作成
        out_path = Path(self.config.output_path)
        out_path.mkdir(parents=True, exist_ok=True)

        save_path = out_path / "best_param.csv"

        # CSV形式で保存
        with open(save_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["institution", "best_param"])
            for k, v in best_params.items():
                writer.writerow([k, v])

        print(f"✅ 最適パラメータ saved to: {save_path}")
        
    def assign_anchor_labels(self, k=5): # リークしてる
        """
        self.anchor に対して、self.Xs_train, self.ys_train を使い
        k-NN多数決でラベルを付与し self.anchor_y に格納する
        """
        # 全機関の訓練データとラベルを結合
        X_train_all = np.vstack(self.Xs_train)
        y_train_all = np.hstack(self.ys_train)

        # k-NNでラベル推定
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_all, y_train_all)
        self.anchor_y = knn.predict(self.anchor)
        
        knn_test = KNeighborsClassifier(n_neighbors=k)
        knn_test.fit(X_train_all, y_train_all)
        self.anchor_y_test = knn_test.predict(self.anchor_test)


    def build_laplacians_from_anchor_labels(self, gamma: Optional[float] = None) -> None:
        """
        アンカーデータとそのラベルを用いて、
        同ラベル近接ラプラシアン(L_within)と異ラベル分離ラプラシアン(L_between)を構築する。
        結果は self.L_within と self.L_between に保存される。

        Args:
            gamma (Optional[float]): RBFカーネルのガンマ値。Noneの場合、1/n_features を使用。
        """
        if self.anchor.size == 0 or self.anchor_y.size == 0:
            self.logger.warning("アンカーデータまたはアンカーラベルが未生成のため、ラプラシアンを構築できません。")
            return

        print("******************** ラプラシアン行列の構築 ********************")
        from sklearn.metrics.pairwise import rbf_kernel

        n_anchors = self.anchor.shape[0]
        y = self.anchor_y

        # 1. アンカー間の類似度行列 W を計算 (RBFカーネルを使用)
        if gamma is None:
            gamma = 1.0 / self.anchor.shape[1]  # デフォルトのガンマ値
        
        W = rbf_kernel(self.anchor, gamma=gamma)
        np.fill_diagonal(W, 0) # 対角成分は0にする

        # 2. ラベル情報に基づいて、同ラベルペアと異ラベルペアのマスクを作成
        # y.reshape(-1, 1) == y.reshape(1, -1) は、(i,j)成分が y_i == y_j かどうかを示すブール行列
        same_label_mask = (y.reshape(-1, 1) == y.reshape(1, -1))
        diff_label_mask = ~same_label_mask

        # 3. 同ラベル近接ラプラシアン L_within (L_w) の構築
        W_within = W * same_label_mask
        D_within = np.diag(W_within.sum(axis=1))
        self.L_within = D_within - W_within
        
        # 4. 異ラベル分離ラプラシアン L_between (L_b) の構築
        W_between = W * diff_label_mask
        D_between = np.diag(W_between.sum(axis=1))
        self.L_between = D_between - W_between

        # ★★★ ここから追加 ★★★
        # 5. トレースで正規化
        trace_Lw = np.trace(self.L_within)
        if trace_Lw > 1e-9:
            self.L_within /= trace_Lw
            self.logger.info(f"L_within をトレース ({trace_Lw:.4g}) で正規化しました。")

        trace_Lb = np.trace(self.L_between)
        if trace_Lb > 1e-9:
            self.L_between /= trace_Lb
            self.logger.info(f"L_between をトレース ({trace_Lb:.4g}) で正規化しました。")
        # ★★★ ここまで追加 ★★★

        self.logger.info(f"同ラベル近接ラプラシアン (L_within) を構築しました。Shape: {self.L_within.shape}")
        self.logger.info(f"異ラベル分離ラプラシアン (L_between) を構築しました。Shape: {self.L_between.shape}")

        self.logger.info(f"同ラベル近接ラプラシアン (L_within) を構築しました。Shape: {self.L_within.shape}")
        self.logger.info(f"異ラベル分離ラプラシアン (L_between) を構築しました。Shape: {self.L_between.shape}")


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
        self.logger.info(f"各機関（訓練データ）の数と次元数: {self.Xs_train[0].shape}")
        # アンカーデータの生成
        self.anchor = self.produce_anchor(
            num_row=self.config.num_anchor_data, num_col=self.Xs_train[0].shape[1], seed=self.config.seed
        )
        
        
        # アンカーデータの生成
        self.anchor_test = self.produce_anchor(
            num_row=self.config.num_anchor_data, num_col=self.Xs_train[0].shape[1], seed=self.config.seed+1
        )
        print("num_row", self.config.num_anchor_data, "num_col", self.Xs_train[0].shape[1])
        print("Xs_train[0].shape", self.Xs_train[0].shape, "Xs_test[0].shape", self.Xs_test[0].shape)
        # 中間表現の生成
        self.make_intermediate_expression()
        #self.make_intermediate_expression(USE_KERNEL=True)
        self.config.now = "g"
        # 統合表現の生成
        if self.config.G_type == "Imakura":
            self.make_integrate_expression()
        elif self.config.G_type  == "targetvec":
            self.make_integrate_expression_targetvec()
        elif self.config.G_type  == "GEP":
            if self.config.semi_integ:
                self.make_semi_integrate_expression()
            self.make_integrate_expression_gen_eig(use_eigen_weighting=False)
        elif self.config.G_type  == "GEP_weighted":
            self.make_integrate_expression_gen_eig(use_eigen_weighting=True)
        elif self.config.G_type == "ODC": # この分岐を追加
            self.make_integrate_expression_odc()
        elif self.config.G_type  == "nonlinear":
            #self.assign_anchor_labels(k=5)
            #self.build_laplacians_from_anchor_labels()
            self.make_integrate_nonlinear_expression()
        elif self.config.G_type  == "nonlinear_tuning":
            self.make_integrate_nonlinear_expression_tuning()
        elif self.config.G_type == "nonlinear_linear":
            self.make_integrate_nonlinear_linear()
        elif self.config.G_type == "mlp_objective":
            self.build_init_from_gen_eig()   # ← 上で追加した関数
            self.make_integrate_gen_eig_fitting_objective()       
        else:
            print(f"Unknown G_type: {self.config.G_type}")

        self.logger.info(f"{self.config.dim_integrate}:次元")
        self.logger.info(f"{self.config.num_institution_user} 機関人数")
        self.logger.info(f"{self.config.num_institution} 機関数")
        
        self.integrate_metrics()


    @staticmethod
    # この関数外に出したい
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, num_institution: int, num_institution_user: int, y_name: str = "target"
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        print("********************データの分割********************")
        """
        複数機関を想定してデータセットを分割する関数
        """

        train_df = train_df.copy()
        test_df = test_df.copy()
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

    def produce_anchor(self, num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
        """
        アンカーデータを生成する関数
        """
        if  self.config.anchor_method == "gaussian":
            np.random.seed(seed=seed)
            anchor = np.random.randn(num_row, num_col)
            return anchor
        
        elif  self.config.anchor_method == "uniform":
            """
            train_df の各特徴量の [min, max] から一様乱数でアンカーを生成する。
            y 列（config.y_name）は除外。
            """
            rng = np.random.default_rng(seed)
            y_name = getattr(self.config, "y_name", "target")

            # 特徴量行列の取得（y を除外）
            if y_name in self.train_df.columns:
                X_df = self.train_df.drop(columns=[y_name])
            else:
                # フォールバック（分割済みがある場合）
                if self.Xs_train:
                    X_df = pd.DataFrame(np.vstack(self.Xs_train))
                else:
                    # 何も無ければ [-1,1] の一様
                    return rng.uniform(-1.0, 1.0, size=(num_row, num_col))

            X_vals = X_df.values
            # 列数は num_col に合わせる（超過分は切り詰め）
            if X_vals.shape[1] < num_col:
                num_col = X_vals.shape[1]
            X_vals = X_vals[:, :num_col]

            # 列ごとの min/max（NaN 無視）
            col_min = np.nanmin(X_vals, axis=0)
            col_max = np.nanmax(X_vals, axis=0)

            # 無効値はデフォルト [-1,1] に置換
            invalid = ~np.isfinite(col_min) | ~np.isfinite(col_max)
            col_min = np.where(invalid, -1.0, col_min)
            col_max = np.where(invalid,  1.0, col_max)

            # 一様サンプリング（幅0の列は定数になる）
            width = np.clip(col_max - col_min, 0.0, None)
            U = rng.random((num_row, num_col))
            anchor = col_min + U * width
            return anchor

    def make_intermediate_expression(self) -> None:
        print("********************中間表現の生成********************")
        """
        中間表現を生成する関数
        """
        print(self.config)
        print("self.config.dim_intermediate", self.config.dim_intermediate)
        print()
        self.config.f_seed = 0
        self.config.f_seed_2 = 0
        mixed = False
        unfixed_mixed = False
        if self.config.True_F_type == "kernel_pca_svd_mixed": #
            mixed = True
        elif self.config.True_F_type == "kernel_pca_unfixed_mixed":
            unfixed_mixed = True
            # kernel_pca_unfixed_gamma
        for X_train, X_test in zip(tqdm(self.Xs_train), self.Xs_test):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用
            if mixed:
                if self.config.f_seed_2 % 2 == 0:
                    self.config.F_type = "kernel_pca_self_tuning"
                    #print("svd")
                else:
                    self.config.F_type = "svd"
                    #print("kpca")
                self.config.f_seed_2 += 1
            elif unfixed_mixed:
                self.config.f_seed_2 += 1
                if self.config.f_seed_2 % 6 == 0:
                    self.config.F_type = "svd"
                else:
                    self.config.F_type = "kernel_pca_unfixed_gamma"
            #print(self.config.F_type)
            X_train_svd, X_test_svd, anchor_svd, anchor_test_svd = reduce_dimensions(
               X_train=X_train,
               X_test=X_test,
               n_components=self.config.dim_intermediate,
               anchor=self.anchor,
               anchor_test=self.anchor_test,
               F_type=self.config.F_type,
               seed=self.config.f_seed,
               config=self.config,)
            self.config.f_seed += 1


            # そのままで実験  ##########################################
            #X_train_svd = X_train
            #X_test_svd = X_test
            #anchor_svd = self.anchor

            # svdを適用したデータをリストに格納
            self.Xs_train_inter.append(X_train_svd)
            self.Xs_test_inter.append(X_test_svd)
            self.anchors_inter.append(anchor_svd)
            self.anchors_test_inter.append(anchor_test_svd)
            
            
            # 標準化 # qsar だと欠損になる
            
            # # SVDを適用したデータをリストに格納
            # scaler = StandardScaler()

            # # 訓練データの標準化
            # X_train_svd = scaler.fit_transform(X_train_svd)
            # self.Xs_train_inter.append(X_train_svd)

            # # テストデータの標準化
            # X_test_svd = scaler.transform(X_test_svd)
            # self.Xs_test_inter.append(X_test_svd)

            # # アンカーデータの標準化
            # anchor_svd = scaler.fit_transform(anchor_svd)
            # self.anchors_inter.append(anchor_svd)

            # # テスト用アンカーデータの標準化
            # anchor_test_svd = scaler.transform(anchor_test_svd)
            # self.anchors_test_inter.append(anchor_test_svd)

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])

        self.logger.info(f"中間表現（訓練データ）の数と次元数: {self.Xs_train_inter[0].shape}")

    def make_semi_integrate_expression(self) -> None:
        print("********************中間表現の生成********************")
        """
        中間統合表現を生成する関数
        """
        self.Xs_train_inter_copy = self.Xs_train_inter.copy()
        self.Xs_test_inter_copy = self.Xs_test_inter.copy()
        self.anchors_inter_copy = self.anchors_inter.copy()
        # 中間表現
        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []
        self.anchors_inter: list[np.ndarray] = []

        for X_train, X_test, y_train, anchor_inter in zip(tqdm(self.Xs_train_inter_copy), self.Xs_test_inter_copy, self.ys_train, self.anchors_inter_copy):
            X_train_svd, X_test_svd, anchor_svd, anchor_test_svd = reduce_dimensions(
               X_train=X_train,
               X_test=X_test,
               n_components=self.config.dim_intermediate,
               y_train=y_train,
               anchor=anchor_inter,
               seed=self.config.f_seed,
               F_type="kcca", 
               config=self.config,)
            self.config.f_seed += 1

            # svdを適用したデータをリストに格納
            self.Xs_train_inter.append(X_train_svd)
            self.Xs_test_inter.append(X_test_svd)
            self.anchors_inter.append(anchor_svd)
            self.anchors_test_inter.append(anchor_test_svd)

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])

        self.logger.info(f"中間表現（訓練データ）の数と次元数: {self.Xs_train_inter[0].shape}")

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
        # 擬似逆行列の絶対値総和を計算するための変数を初期化
        total_g_abs_sum = 0.0

        for X_train_inter, X_test_inter, anchor_inter, anchor_test_inter in zip(
            tqdm(self.Xs_train_inter), self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter
        ):
            # 各機関のアンカーデータの中間表現を転置して、擬似逆行列を求める
            pseudo_inverse = np.linalg.pinv(anchor_inter.T)  # \hat{X}^{anc}+


            # 各機関の統合関数を求める
            integrate_function = np.dot(Z, pseudo_inverse)  # G^{(i)}

            # 擬似逆行列の絶対値の総和を計算して加算
            total_g_abs_sum += np.sum(np.abs(integrate_function))

            # 統合関数で各機関の中間表現を統合表現に変換
            X_train_integrate = np.dot(integrate_function, X_train_inter.T)
            X_test_integrate = np.dot(integrate_function, X_test_inter.T)
            anchor_integ = np.dot(integrate_function, anchor_inter.T)
            anchor_test_integ = np.dot(integrate_function, anchor_test_inter.T)
            # そのままで実験 ##########################################
            # X_train _integrate = X_train_inter.T
            # X_test_integrate = X_test_inter.T

            # 統合表現をリストに格納
            Xs_train_integrate.append(X_train_integrate.T)
            Xs_test_integrate.append(X_test_integrate.T)
            
            self.anchors_integ.append(anchor_integ.T)
            self.anchors_test_integ.append(anchor_test_integ.T)

        # 計算した総和をconfigに保存
        self.config.g_abs_sum = total_g_abs_sum
        print(f"擬似逆行列の絶対値の総和: {self.config.g_abs_sum}")

        print("統合表現の次元数: ", Xs_train_integrate[0].shape[1])

        # 全ての機関の統合表現をくっつけ、1つのarrayに変換
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ = np.vstack(Xs_test_integrate)

        # yもくっつける
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)
        
        self.Z = Z

        # logにも出力
        self.logger.info(f"統合表現（訓練データ）の数と次元数: {self.X_train_integ.shape}")

    def make_integrate_expression_targetvec(self) -> None:
        """
        固有値問題 (16) に基づき統合関数 G^(k) を求め，
        各機関の中間表現を共通表現へ射影する。
        前提: self.anchors_inter          : list[np.ndarray]  r × d_I
            self.Xs_train_inter/test_inter : list[np.ndarray] n_k × d_I
            self.config.dim_common        : 共通表現次元 p̂
            self.config.num_institution   : 機関数 m
            self.config.num_anchor_data   : アンカー数 r
        """
        print("********************統合表現の生成 (目標ベクトル型) ********************")
        from numpy.linalg import eigh, pinv

        # --------------------------------------------------
        # 1. C_s̃ = m I_r - Σ_k S̃^(k) (S̃^(k))^†   （r×r）
        # --------------------------------------------------
        m = self.config.num_institution
        r = self.config.num_anchor_data
        I_r = np.eye(r)

        C_tildeS = m * I_r
        for S_tilde in self.anchors_inter:                # S_tilde : (r, d_I)
            C_tildeS -= S_tilde @ pinv(S_tilde)           # r×r
            
        # 固有値を計算
        eigvals = np.linalg.eigvals(C_tildeS)

        # すべての固有値が正か確認

        # --------------------------------------------------
        # 2. 固有値問題  C_s̃ z = λ z  を解く（昇順）
        # --------------------------------------------------
        #eigvals, eigvecs = eigh(C_tildeS)                 # 昇順で返る
        p_hat = self.config.dim_integrate
        #Z = eigvecs[:, :p_hat]                            # r×p̂  —— 目標行列 Z
        
        # 目的関数が向上するようなZを選ぶ
        #objective_direction_ratio = getattr(self.config, "objective_direction_ratio", 0.1)
        #if objective_direction_ratio < 0:
            # すべての固有値が正か確認
        #    is_positive_definite = np.all(eigvals > 0)
        #    print(f"C_tildeS is positive definite: {is_positive_definite}")
            
        #    selected_idx, Z, eigvals_centered, eigvecs, coef = self.select_eigvecs_linear_hybrid(C_tildeS, self.anchor_y, p_hat=p_hat, objective_direction_ratio=objective_direction_ratio)
            #is_positive_definite = np.all(eigvals > 0)
            #print(f"zzzC_tildeS is positive definite: {is_positive_definite}")
        #else:
        # ❷ 実対称用の固有値分解を使う
        eigvals, eigvecs = np.linalg.eigh(C_tildeS)
        # ❸ 念のため負の丸め誤差を 0 に
        eigvals[eigvals < 0] = 0.0
        Z = eigvecs[:, :p_hat]
        
        # --------------------------------------------------
        # 3. 各機関ごとに  g^(k) = (S̃^(k))^† Z   を計算
        #    → 係数行列 G^(k)（d_I × p̂）
        # --------------------------------------------------
        Gs = []            # 係数行列 G^(k) を保存（デバッグ用）
        Xs_train_integrate = []
        Xs_test_integrate  = []

        for S_tilde_k, X_tr_k, X_te_k in zip(
                self.anchors_inter, self.Xs_train_inter, self.Xs_test_inter):

            Gk = pinv(S_tilde_k) @ Z                      # (d_I, p̂)
            Gs.append(Gk)

            Xs_train_integrate.append(X_tr_k @ Gk)        # 射影
            Xs_test_integrate.append( X_te_k @ Gk)        # 射影

        # --------------------------------------------------
        # 4. スタックして最終データを保持
        # --------------------------------------------------
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")
        
        self.Z = Z

        # 必要なら self.Gs = Gs などで保存しておくと解析に便利
        
        # すべての固有値が正か確認
        
        reg = LinearRegression()
        reg.fit(Z, self.anchor_y)
        y_pred = reg.predict(Z)
        mse = mean_squared_error(self.anchor_y, y_pred)
        print(f"平均二乗誤差 (MSE)最小: {mse:.4g}")
        #print(eigvals > 0)
        reg = LinearRegression()
        Z = eigvecs[:, eigvals.argsort()[:p_hat]]
        reg.fit(Z, self.anchor_y)
        y_pred = reg.predict(Z)
        mse = mean_squared_error(self.anchor_y, y_pred)
        print(f"平均二乗誤差 (MSE) 普通: {mse:.4g}")
        
        reg = LinearRegression()
        Z_ = eigvecs[:, eigvals.argsort()[:p_hat]]
        reg.fit(Z_, self.anchor_y)
        y_pred = reg.predict(Z_)
        mse = mean_squared_error(self.anchor_y, y_pred)
        print(f"平均二乗誤差 (MSE) 直後2: {mse:.4g}")
        #print(eigvals > 0)

    # ============================================================
    # 〈統合関数の最適化〉§3 一般化固有値問題 (8) ベース
    #   A_s̃ v = λ B_s̃ v ,  vᵀ B_s̃ v = 1
    # ============================================================
    def make_integrate_expression_gen_eig(self, use_eigen_weighting=False) -> None:
        """
        川上・高野 (2024) §3   一般化固有値問題による統合関数
        + オプションで λ に基づくウェイト付け   (exp(-(λ_j-λ1)/(λ_max-λ1)))
        ------------------------------------------------------------
        追加設定:
            self.config.use_eigen_weighting : bool  ← デフォルト False
        追加出力:
            self.lambda_selected : ndarray (p̂,)     ← 選択した λ_j
            self.weights_selected: ndarray (p̂,)     ← w(λ_j)  (use_eigen_weighting=True のとき)
        """
        print("********************統合表現の生成 (一般化固有値型) ********************")
        from functools import reduce

        import numpy as np
        from scipy.linalg import block_diag, eigh

        # --------------------------------------------------
        # 0. 各種設定・寸法
        # --------------------------------------------------
        m       = self.config.num_institution
        p_hat   = self.config.dim_integrate           # ← 共通表現次元
        r       = self.config.num_anchor_data
        lambda_gen = getattr(self.config, 'lambda_gen_eigen', 0)
        print("lambda_gen", lambda_gen)
        orth_ver = bool(getattr(self.config, "orth_ver", None) or False)
        #use_w   = getattr(self.config, "use_eigen_weighting", False)   # ★


        # --------------------------------------------------
        # 1. W̃_s  と  B̃_s  を構築
        # --------------------------------------------------
        W_s_tilde = np.hstack(self.anchors_inter)                     # r × Σd_k
        blocks    = [S.T @ S for S in self.anchors_inter]             # 各 d_k × d_k
        epsilon = 1e-6
        B_s_tilde = reduce(lambda a, b: block_diag(a, b), blocks) + epsilon * np.eye(sum(S.shape[1] for S in self.anchors_inter))

        # --------------------------------------------------
        # 2. Ã_s = 2m B̃_s - 2 WᵀW
        # --------------------------------------------------
        print("W_s_tilde.shape", W_s_tilde.shape, "B_s_tilde.shape", B_s_tilde.shape)
        print("lambda_gen", lambda_gen)
        A_s_tilde = 2 * m * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde) + lambda_gen* np.eye(W_s_tilde.shape[1])  # 正則化項を追加

        # --------------------------------------------------
        # 3. 一般化固有値問題  A v = λ B v
        # --------------------------------------------------
        if orth_ver:
            eigvals, eigvecs = eigh(A_s_tilde)                 # SciPy の一般化固有分解
        else:
            eigvals, eigvecs = eigh(A_s_tilde, B_s_tilde)                 # SciPy の一般化固有分解
        order   = np.argsort(eigvals)                                 # 昇順
        lambdas = eigvals[order][:p_hat]                              # ★ λ_1 … λ_p̂
        print(lambdas)
        V_sel   = eigvecs[:, order[:p_hat]]
        cum_dims = np.cumsum([0] + [S.shape[1] for S in self.anchors_inter])

        # Jreg (目的関数第2項) の値を計算して記録
        jreg_val = 0.0
        for j in range(p_hat):
            gj = V_sel[:, j]
            term1 = 0.0
            sum_Sgj = np.zeros(self.anchors_inter[0].shape[0]) # r次元ベクトル
            for k in range(m):
                gjk = gj[cum_dims[k]:cum_dims[k+1]]
                Sk = self.anchors_inter[k]
                term1 += gjk.T @ (Sk.T @ Sk) @ gjk
                sum_Sgj += Sk @ gjk
            jreg_val += (2.0 * m * term1 - 2.0 * (sum_Sgj @ sum_Sgj))
        self.config.jreg_gep = f"{jreg_val:.6g}"
        print(f"Jreg (GEP) = {self.config.jreg_gep}")
        
        # --- ノルム値の計算 ---
        norm_val_sum = 0.0
        for j in range(p_hat):
            gj = V_sel[:, j]
            for k in range(m):
                gjk = gj[cum_dims[k]:cum_dims[k+1]]
                Sk = self.anchors_inter[k]
                norm_vec = Sk @ gjk
                norm_val_sum += norm_vec @ norm_vec
        
        avg_norm_val = norm_val_sum / p_hat
        self.config.g_norm_val_gep = f"{avg_norm_val:.6g}"
        print(f"norm (GEP) = {self.config.g_norm_val_gep}")

        # λ の総和を計算して記録
        self.config.sum_objective_function = f"{np.sum(lambdas):.4g}"
        print(f"λ の総和 (sum_objective_function): {self.config.sum_objective_function}")

        self.config.g_abs_sum = f"{np.sum(np.abs(V_sel)):.4g}"  # Σd_k × p̂
        print(f"V_selの絶対値の総和: {self.config.g_abs_sum}")

        mean_vars = []
        for k in range(len(self.anchors_inter)):
            V_k = V_sel[cum_dims[k]:cum_dims[k + 1], :]               # 機関 k の部分
            var_k = np.var(V_k, axis=0)                               # 列ごとの分散
            mean_vars.append(np.mean(var_k))                         # 分散の平均を計算
        self.config.g_mean_var = f"{np.mean(mean_vars):.4g}"         # 全機関の平均を格納
        print(f"機関ごとのベクトル分散の平均: {self.config.g_mean_var}")

        # 条件数を計算
        lambda_min, lambda_max = lambdas[0], lambdas[-1]
        print(lambda_min, lambda_max)
        print(lambda_min, lambda_max)
        print(lambda_max / lambda_min)
        self.config.g_condition_number = f"{lambda_max / lambda_min:.4g}" if lambda_min > 0 else "inf"
        print(f"条件数: {self.config.g_condition_number}")
        
        # --------------------------------------------------
        # 5. 機関ごとの G^(k) 抽出と射影
        # --------------------------------------------------
        # ベクトル vj の分散を計算し、機関ごとに平均を取る
        Xs_train_integrate, Xs_test_integrate = [], []

        for k, (d_k, X_tr_k, X_te_k) in enumerate(
                zip(np.diff(cum_dims), self.Xs_train_inter, self.Xs_test_inter)):

            Gk = V_sel[cum_dims[k]:cum_dims[k + 1], :]               # d_k × p̂
            Xs_train_integrate.append(X_tr_k @ Gk)
            Xs_test_integrate.append(X_te_k @ Gk)

        # --------------------------------------------------
        # 6. スタック & 保存
        # --------------------------------------------------
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

        # 解析用に λ とウェイトも保持 ★
        if use_eigen_weighting:

            self.config.eigenvalues  = lambdas

    def make_integrate_expression_odc(self) -> None:
        """
        Orthogonal Procrustes Problem (OPP) に基づく統合表現を生成する。
        G_k = U_k V_k^T  where  anchor_k^T @ anchor_1 = U_k S_k V_k^T
        """
        print("********************統合表現の生成 (Orthogonal Procrustes) ********************")

        if not self.anchors_inter:
            self.logger.error("アンカーの中間表現が生成されていません。")
            return

        # 1. 基準となるアンカー (A_1) を設定
        anchor_1 = self.anchors_inter[0]

        Xs_train_integrate = []
        Xs_test_integrate = []

        # 2. 各機関 k についてループ
        for anchor_k, X_tr_k, X_te_k, anchor_inter, anchor_test_inter in zip(
            self.anchors_inter, self.Xs_train_inter, self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter
        ):
            # 3. M_k = A_k^T @ A_1 を計算　Oはなし
            M_k = anchor_k.T @ anchor_1

            # 4. M_k を特異値分解(SVD)
            # full_matrices=False にして、計算結果の行列サイズを揃える
            U_k, _, Vh_k = np.linalg.svd(M_k, full_matrices=False)

            # 5. 変換行列 G_k = U_k @ Vh_k を計算 (Vh_k は V_k^T)
            G_k = U_k @ Vh_k

            # 6. G_k を用いて中間表現を射影
            # これにより、全機関の表現が anchor_1 と同じ次元数に変換される
            Xs_train_integrate.append(X_tr_k @ G_k)
            Xs_test_integrate.append(X_te_k @ G_k)
            self.anchors_integ.append(anchor_inter @ G_k)
            self.anchors_test_integ.append(anchor_test_inter @ G_k)

        self.Z = anchor_1

        # 7. スタックして最終データを保持
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def _one_hot(self, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        # classes の順に one-hot を作る（列順が常に一定）
        return (y.reshape(-1, 1) == classes.reshape(1, -1)).astype(float)

    def select_eigvecs_multilogit_hybrid(
        self,
        M: np.ndarray,
        y: np.ndarray,
        p_hat: int,
        objective_direction_ratio: float = 0.0,
        C: float = 1.0,
        random_state: int = 0,
    ):
        y = y - np.mean(y)
        eigvals, eigvecs = np.linalg.eigh(M)
        n = M.shape[0]

        # ① クラス配列を固定（列順の基準）
        classes = np.unique(y)           # 例: array([0,1,2,3])
        K = len(classes)
        Y = self._one_hot(y, classes)         # (n, K)

        # ② 初期確率：事前確率で (n,K)
        class_priors = Y.mean(axis=0)    # (K,)
        P = np.tile(class_priors, (n, 1))

        # 1) 固有値で前半を先取り
        m1 = int(p_hat * (1 - objective_direction_ratio))
        eig_order = np.argsort(eigvals)
        selected_idx = list(eig_order[:m1])
        remaining = [i for i in range(n) if i not in selected_idx]

        # 先取り分で一度フィット
        if m1 > 0:
            Z_sel = eigvecs[:, selected_idx]
            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                C=C, max_iter=1000, random_state=random_state
            )
            clf.fit(Z_sel, y)
            P_raw = clf.predict_proba(Z_sel)   # 列順は clf.classes_
            # ③ 列順を classes に合わせて並べ替え
            order = [np.where(clf.classes_ == c)[0][0] for c in classes]
            P = P_raw[:, order]
        else:
            Z_sel = None

        # 2) 残りをスコアで前向き選択
        m2 = p_hat - m1
        for _ in range(m2):
            # 安全確認（デバッグ用）
            # assert Y.shape[1] == P.shape[1] == K

            R = Y - P
            W_diag_list = [P[:, c] * (1.0 - P[:, c]) + 1e-12 for c in range(K)]

            best_j, best_score = None, -np.inf
            for j in remaining:
                u = eigvecs[:, j]
                num = 0.0
                for c in range(K):
                    r_c = R[:, c]
                    W_c = W_diag_list[c]
                    uTr = float(u @ r_c)
                    uWu = float(u @ (W_c * u))
                    num += (uTr ** 2) / uWu
                score = num  # 固有値ペナルティを入れるなら: - gamma * eigvals[j]

                if score > best_score:
                    best_score = score
                    best_j = j

            selected_idx.append(best_j)
            remaining.remove(best_j)
            Z_sel = eigvecs[:, selected_idx]

            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                C=C, max_iter=1000, random_state=random_state
            )
            clf.fit(Z_sel, y)
            P_raw = clf.predict_proba(Z_sel)
            order = [np.where(clf.classes_ == c)[0][0] for c in classes]
            P = P_raw[:, order]

        Z = eigvecs[:, selected_idx]
        
        return selected_idx, Z, eigvals, eigvecs
    
    def select_eigvecs_linear_hybrid(
        self,
        M: np.ndarray,
        y: np.ndarray,
        p_hat: int,
        objective_direction_ratio: float = 0.0,
    ):
        """
        線形回帰（最小二乗）用の固有ベクトル選択（ハイブリッド：後ろ向き選択版）。
        - 前半 m1 本: 固有値が小さい順に先取り（固定）
        - 後半 m2 本: 中心化スコアに基づく「後ろ向き選択」
        - 候補は固有値小さい順に p_hat の3倍までに制限
        """
        # 1) 固有分解（対称化しておくと数値安定）
        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)  # 各列は直交・ノルム1
        n = M.shape[0]

        # 固有値小さい順に p_hat の3倍までに制限
        max_candidates = min(3 * p_hat, n)
        candidate_indices = np.argsort(eigvals)[:max_candidates]
        eigvals = eigvals[candidate_indices]
        eigvecs = eigvecs[:, candidate_indices]

        # 2) スコア用に中心化（切片ありOLSと等価）
        y_c = y - np.mean(y)
        U = eigvecs
        U_c = U - U.mean(axis=0)  # 列中心化（直交性は壊れる）

        # 3) 前半: 固有値の小さい順で m1 本（固定セット）
        m1 = int(p_hat * (1.0 - objective_direction_ratio))
        m1 = max(0, min(p_hat, m1))
        order_small = np.argsort(eigvals)  # 昇順
        fixed_idx = list(order_small[:m1])
        fixed_idx_set = set(fixed_idx)

        # 4) 後半: 残り候補から m2 本を「後ろ向き選択」で決める
        m2 = p_hat - m1
        if m2 < 0:
            m2 = 0

        remaining_pool = [j for j in range(max_candidates) if j not in fixed_idx_set]
        if m2 == 0:
            selected_idx = fixed_idx
        else:
            # 後ろ向き選択：残り候補を最終的に m2 本まで「削って」絞り込む
            chosen = remaining_pool.copy()  # 現在の採用集合（ここから削る）
            A_fixed = U_c[:, fixed_idx] if len(fixed_idx) > 0 else None

            def rss_with_columns(cols: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
                """固定列 + 指定列で OLS を解き、RSSを返す（中心化・切片なし）"""
                A_cols = U_c[:, cols] if len(cols) > 0 else None
                if A_fixed is None and A_cols is None:
                    r = y_c
                    return float(r @ r), np.empty((0,)), np.empty((0, 0))
                elif A_fixed is None:
                    A = A_cols
                elif A_cols is None:
                    A = A_fixed
                else:
                    A = np.hstack([A_fixed, A_cols])

                coef, *_ = np.linalg.lstsq(A, y_c, rcond=None)
                r = y_c - A @ coef
                return float(r @ r), coef, A

            base_rss, _, _ = rss_with_columns(chosen)

            while len(chosen) > m2:
                best_drop = None
                best_increase = np.inf

                for j in chosen:
                    trial = [c for c in chosen if c != j]
                    rss_j, _, _ = rss_with_columns(trial)
                    increase = rss_j - base_rss
                    if increase < best_increase:
                        best_increase = increase
                        best_drop = j

                if best_drop is None:
                    break
                chosen.remove(best_drop)
                base_rss, _, _ = rss_with_columns(chosen)

            selected_idx = fixed_idx + chosen

        # 5) 最終 Z は「元の固有ベクトル列」を返す（評価は切片ありでOK）
        Z = U[:, selected_idx]

        # 係数（参考）：中心化問題の最小二乗解（スコア計算と同条件）
        A_c = U_c[:, selected_idx]
        coef_centered, *_ = np.linalg.lstsq(A_c, y_c, rcond=None)

        # デバッグ出力（切片ありで評価）
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        reg = LinearRegression()  # intercept=True
        reg.fit(Z, y)
        mse_hybrid = mean_squared_error(y, reg.predict(Z))
        print(f"[hybrid-backward] MSE (with intercept) = {mse_hybrid:.6g}")

        Z_small = U[:, order_small[:p_hat]]
        reg2 = LinearRegression()
        reg2.fit(Z_small, y)
        mse_small = mean_squared_error(y, reg2.predict(Z_small))
        print(f"[small-eigs]     MSE (with intercept) = {mse_small:.6g}")

        return selected_idx, Z, eigvals, eigvecs, coef_centered
    # ------------------------------------------------------------------
    # 〈非線形統合〉　射影行列 P^(k) で Z を最適化する ２段階アルゴリズム
    # ------------------------------------------------------------------
    def make_integrate_nonlinear_expression(self) -> None:
        """
        非線形（カーネル）版：アンカー同士の射影行列で共通ターゲット Z を導き，
        各機関データを同じ次元 p̂ へ写像する。
        """
        import numpy as np
        from numpy.linalg import eig, inv, norm
        from sklearn.metrics.pairwise import rbf_kernel

        m  = len(self.anchors_inter)              # 機関数
        r  = self.anchors_inter[0].shape[0]       # アンカー行数
        p̂  = self.config.dim_integrate           # 統合表現次元
        lw_alpha     = float(getattr(self.config, "lw_alpha", None) or 0) # 同ラベル近接ラプラシアンの重み
        lb_beta      = float(getattr(self.config, "lb_beta", None) or 0) # 異ラベル分離ラプラシアンの重み

        Ks, Ps, gammas = [], [], []
        I_r = np.eye(r)
        
        if self.config.gamma_type == "auto":
            for S̃ in self.anchors_inter:             # S̃ : r×d̃_k
                γ = 1.0 / S̃.shape[1]                # γ = 1/d̃_k
                gammas.append(γ)

        elif self.config.gamma_type == "X_tuning":
            for X_train_inter in self.Xs_train_inter:
                # gamma を計算
                # gamma を計算
                gamma = self_tuning_gamma(X_train_inter, standardize=False, k=3, summary='median')
                gamma *= self.config.gamma_ratio_krr
                gammas.append(gamma)
        
        elif self.config.gamma_type == "same_as_f":
            gammas = self.config.nl_gammas
            print("ggggggggggggggggggggggggggggg")
            print(len(self.Xs_train_inter))
            print(len(gammas))
            # svd だと記録されないためバグる
        print(gammas)

        if hasattr(self.config, "nl_lambda"):
            lam = self.config.nl_lambda
        else:
            lam = 1e-2
        #gammas = [11, 15.5, 1000]
        #k = 1
        # --- 1. Gram 行列と射影行列 ---
        for i, S̃ in enumerate(self.anchors_inter):             # S̃ : r×d̃_k
            K = rbf_kernel(S̃, S̃, gamma=gammas[i])       # r×r
            # (a) カーネル行列（先に作って正規化）
            if self.config.K_normalization:
                mu_max = max(eigvalsh(K).max(), 1e-12)            # スペクトル半径
                K = K / mu_max                                # ||K||_2 = 1
            
            Ks.append(K)
            Ps.append(K @ inv(K + lam * I_r))     # 射影
        
        M = sum((P - I_r).T @ (P - I_r) for P in Ps)
        ## M 正規化 ラプラシアンなしならしなくてよい
        trace_M = np.trace(M)
        if trace_M > 1e-9:
            M /= trace_M
        
        # --- 2. 固有値問題 → Z (r×p̂ , ‖Z‖_F=1) --- 近接ラプラシアンの重みも加える
        Q = M #+ lw_alpha * self.L_within - lb_beta * self.L_between

        # ❶ ほんのわずかな非対称を切り落とす
        Q = (Q + Q.T) * 0.5
        
        # 目的関数が向上するzを選択
        #objective_direction_ratio = getattr(self.config, "objective_direction_ratio", 0)
        #if objective_direction_ratio < 0:
        #    print(1)
        #    idx, Z, eigvals, eigvecs = self.select_eigvecs_linear_hybrid(Q, self.anchor_y, p_hat=p̂, objective_direction_ratio=objective_direction_ratio)
        #    print(2)
        #else:
        # ❷ 実対称用の固有値分解を使う
        eigvals, eigvecs = np.linalg.eigh(Q)
        # ❸ 念のため負の丸め誤差を 0 に
        eigvals[eigvals < 0] = 0.0
        Z = eigvecs[:, eigvals.argsort()[:p̂]]
            
        # 列ごとに ||z_j||_2 = 1 へ
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz
        
        # S_hat_list (S_hat_k = P_k @ Z) の計算
        #S_hat_list = []
        #for P in Ps:
        #    S_hat_list.append(P @ Z)
        #self.anchors_integ = S_hat_list
        #self.logger.info(f"S_hat_list を計算しました。要素数: {len(self.anchors_integ)}, 各要素のShape: {self.anchors_integ[0].shape}")
        
        cul_test = False
        if cul_test:
            Ks_test, Ps_test, gammas = [], [], []
                        # --- 1. Gram 行列と射影行列 ---
            for i, S̃_test in enumerate(self.anchors_inter_test):             # S̃ : r×d̃_k
                K_test = rbf_kernel(S̃_test, S̃_test, gamma=gammas[i])       # r×r
                # (a) カーネル行列（先に作って正規化）
                mu_max_test = max(eigvalsh(K_test).max(), 1e-12)            # スペクトル半径
                K_test = K_test / mu_max_test                                # ||K||_2 = 1
                
                Ks_test.append(K_test)
                Ps_test.append(K_test @ inv(K_test + lam * I_r))     # 射影

            M_test = sum((P - I_r).T @ (P - I_r) for P in Ps_test)
            ## M 正規化 ラプラシアンなしならしなくてよい
            trace_M = np.trace(M_test)
            if trace_M > 1e-9:
                M_test /= trace_M

            # --- 2. 固有値問題 → Z (r×p̂ , ‖Z‖_F=1) --- 近接ラプラシアンの重みも加える
            Q_test = M_test #+ lw_alpha * self.L_within - lb_beta * self.L_between

            # ❶ ほんのわずかな非対称を切り落とす
            Q_test = (Q_test + Q_test.T) * 0.5
            
            objective_direction_ratio = getattr(self.config, "objective_direction_ratio", 0)
            if objective_direction_ratio < 0:
                print(1)
                idx, Z_test, eigvals_test, eigvecs_test = self.select_eigvecs_linear_hybrid(Q_test, self.anchor_y_test, p_hat=p̂, objective_direction_ratio=objective_direction_ratio)
                print(2)
            else:
                # ❷ 実対称用の固有値分解を使う
                eigvals_test, eigvecs_test = np.linalg.eigh(Q_test)
                # ❸ 念のため負の丸め誤差を 0 に
                eigvals_test[eigvals_test < 0] = 0.0
                Z_test = eigvecs_test[:, eigvals_test.argsort()[:p̂]]

            # 列ごとに ||z_j||_2 = 1 へ
            for j in range(Z_test.shape[1]):
                nz = np.linalg.norm(Z_test[:, j])
                if nz > 0:
                    Z_test[:, j] /= nz

            # S_hat_list (S_hat_k = P_k @ Z) の計算
            S_hat_test_list = []
            for P in Ps_test:
                S_hat_test_list.append(P @ Z_test)
            self.anchors_test_integ = S_hat_test_list
            self.logger.info(f"S_hat_list を計算しました。要素数: {len(self.anchors_test_integ)}, 各要素のShape: {self.anchors_test_integ[0].shape}")

        # --- 3. 各機関の係数 B^(k) とデータ射影 ---
        Xs_train_intg, Xs_test_intg = [], []
        # zipに self.anchors_test_inter を追加
        for K, S̃_train, S̃_test, γ, X_tr, X_te in zip(
            Ks, self.anchors_inter, self.anchors_test_inter, gammas,
            self.Xs_train_inter, self.Xs_test_inter
        ):
            # 学習データから係数 Bk を計算
            mu_max = max(eigvalsh(K).max(), 1e-12)            # スペクトル半径
            
            Bk  = inv(K + lam * I_r) @ Z          # r×p̂
            
            # (a) 学習データの射影
            K_tr = rbf_kernel(X_tr, S̃_train, gamma=γ)  # n_k×r

            # (b) テストデータの射影
            K_te = rbf_kernel(X_te, S̃_train, gamma=γ)  # t_k×r

            # (c) 学習アンカーの射影結果 S_hat (P @ Z と等価)

            # (d) ★★★ テストアンカーの射影結果 S_hat_test ★★★
            K_anchor_test = rbf_kernel(S̃_test, S̃_train, gamma=γ) # (r_test, r_train)
            
            if self.config.K_normalization:
                #s = np.linalg.svd(K_tr, compute_uv=False)
                #mu_max = s.max()
                K_tr = K_tr / mu_max
                    
                #s = np.linalg.svd(K_te, compute_uv=False)
                #mu_max = s.max()
                K_te = K_te / mu_max
                # ||K||_2 = 1
                mu_max = max(eigvalsh(K_anchor_test).max(), 1e-12)            # スペクトル半径
                K_anchor_test = K_anchor_test / mu_max                             # ||K||_2 = 1
            
            Xs_train_intg.append(K_tr @ Bk)       # n_k×p̂
            Xs_test_intg.append(K_te @ Bk)        # t_k×p̂
            self.anchors_integ.append(K @ Bk)
            self.anchors_test_integ.append(K_anchor_test @ Bk)

                
        # --- 4. スタック & 保存 ---
        self.X_train_integ = np.vstack(Xs_train_intg)
        self.X_test_integ  = np.vstack(Xs_test_intg)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        self.logger.info(f"nonlinear integrate: X_train {self.X_train_integ.shape}")
        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")
        
        # 固有値の小さい順に p_hat 個選択
        lambdas = eigvals[:p̂]  # 固有値の上位 p_hat 個

        # 固有値の総和を計算
        sum_lambdas = np.sum(lambdas)

        # 結果を self.config.g_abs_sum に格納
        self.config.g_abs_sum = f"{sum_lambdas:.4g}"
        
        self.Z = Z

        # デバッグ用出力
        print(f"固有値 λ の上位 {p̂} 個の総和: {self.config.g_abs_sum}")
        #print(f"固有値 λ の目的関数減少 {p̂} 個の総和: {np.sum(eigvals[idx])}")
        
    """
    #バイアス項あり
    # ---------------------------------------------------------------
    # 〈非線形統合〉  RBF ⊕ Linear  ―  λ は RBF 部分だけを罰する
    #     g(x)=αᵀ k_rbf(x,·) + β₀ + βᵀx
    #     ───────────────────────────────
    #     零空間   = { β₀+βᵀx }     ← 無罰則
    #     有罰則   = α             ← λ‖α‖²
    # ---------------------------------------------------------------
    def make_integrate_nonlinear_linear(self) -> None:

        m   = len(self.anchors_inter)
        r   = self.anchors_inter[0].shape[0]
        p_hat = self.config.dim_integrate
        lam = getattr(self.config, "nl_lambda", 1e-2)

        Ks, Ps, gammas, Ps_lambda = [], [], [], []
        K_scales = []                           # K の正規化係数（埋め込み時に使う）
        I_r = np.eye(r)

        # ---- gamma の用意（省略：あなたのまま） ----
        gammas = []
        if self.config.gamma_type == "auto":
            for S_tilde in self.anchors_inter:
                gammas.append(1.0 / S_tilde.shape[1])
        elif self.config.gamma_type == "X_tuning":
            for X_train_inter in self.Xs_train_inter:
                gamma = self_tuning_gamma(X_train_inter, standardize=False, k=3, summary='median')
                gammas.append(gamma)
        print(gammas)

        # ---- 1. K と P_λ（厳密 or 一次近似） ----
        USE_FIRST_ORDER = (lam >= 10.0)

        for i, S_tilde in enumerate(self.anchors_inter):
            gamma = gammas[i]

            # (a) カーネル行列（先に作って正規化）
            K_rbf_raw = rbf_kernel(S_tilde, S_tilde, gamma=gamma)     # (r,r) 対称PSD
            mu_max = max(eigvalsh(K_rbf_raw).max(), 1e-12)            # スペクトル半径
            K_rbf = K_rbf_raw / mu_max                                # ||K||_2 = 1
            K_scales.append(mu_max)                                   # 未来の K(x,S) にも適用

            # (b-1) 線形基底（バイアスなし）
            #P_lin = S_tilde 
            # (b-2) 線形基底（バイアス込み）
            P_lin = np.hstack((np.ones((r, 1)), S_tilde))             # (r, d_k+1)

            if USE_FIRST_ORDER:
                # ---- 一次近似：P_λ ≈ P_linProj + (1/λ) P^(1) ----
                # 射影 P_linProj = P (PᵀP)^† Pᵀ
                G = P_lin.T @ P_lin
                G_inv = pinv(G)
                P_linProj = P_lin @ G_inv @ P_lin.T                   # (r,r)

                # P^(1) = K - K P - P K + P K P   （K は正規化済）
                P1 = K_rbf - K_rbf @ P_linProj - P_linProj @ K_rbf + P_linProj @ K_rbf @ P_linProj
                P_lambda = P_linProj + (1.0/lam) * P1                 # (r,r)

                # 係数の近似も用意（後段の embed 用）
                # beta0 = (PᵀP)^† Pᵀ Z,  r0 = Z - P beta0
                # beta ≈ beta0 + (1/λ) (PᵀP)^† Pᵀ K r0
                # alpha ≈ (1/λ) r0
                coeff_mode = "first_order"
                coeff_pack = (G_inv, P_linProj)                       # 係数再計算用
            else:
                # ---- 厳密計算（従来どおり） ----
                A_inv = inv(K_rbf + lam * I_r)                        # (K + λI)^(-1)
                try:
                    M = inv(P_lin.T @ A_inv @ P_lin)
                except np.linalg.LinAlgError:
                    M = pinv(P_lin.T @ A_inv @ P_lin)

                P_lambda = (K_rbf @ A_inv
                            + (P_lin - K_rbf @ A_inv @ P_lin) @ M @ (P_lin.T @ A_inv))
                coeff_mode = "exact"
                coeff_pack = (A_inv, M)

            Ks.append((K_rbf, P_lin, coeff_mode, coeff_pack, mu_max))
            Ps_lambda.append(P_lambda)

        # ---- 2. 共通ターゲット Z（そのまま）----
        M_tot = sum((P_l - I_r).T @ (P_l - I_r) for P_l in Ps_lambda)
        M_sym = 0.5 * (M_tot + M_tot.T)
        eigvals, eigvecs = np.linalg.eigh(M_sym)

        Z = eigvecs[:, eigvals.argsort()[:p_hat]]
        # 列ごとに ||z_j||_2 = 1 へ
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz
        # ---- 3. 各機関データを写像 ----
        Xs_train_intg, Xs_test_intg = [], []

        for (K_rbf, P_lin, coeff_mode, coeff_pack, mu_max), S_tilde, gamma, X_tr, X_te in zip(
                Ks, self.anchors_inter, gammas, self.Xs_train_inter, self.Xs_test_inter):

            if coeff_mode == "exact":
                A_inv, M = coeff_pack
                beta  = M @ (P_lin.T @ A_inv @ Z)                     # (d_k+1, p̂)
                alpha = A_inv @ (Z - P_lin @ beta)                    # (r, p̂)
            else:
                # 一次近似：beta0, r0, beta1, alpha
                G_inv, P_linProj = coeff_pack
                beta0 = G_inv @ (P_lin.T @ Z)
                r0 = Z - P_lin @ beta0
                beta1 = G_inv @ (P_lin.T @ (K_rbf @ r0))
                beta  = beta0 + (1.0/lam) * beta1
                alpha = (1.0/lam) * r0

            # --- 埋め込み関数（K(x,S) も同じスケールで正規化！）---
            def embed(X):
                Kx_raw = rbf_kernel(X, S_tilde, gamma=gamma)          # (n,r)
                Kx = Kx_raw / mu_max                                  # K と同じ正規化
                Px  = np.hstack((np.ones((len(X), 1)), X))            # (n,d_k+1)
                return Kx @ alpha + Px @ beta

            Xs_train_intg.append(embed(X_tr))
            Xs_test_intg.append(embed(X_te))

        # ---- 4. スタック & 保存 ----
        self.X_train_integ = np.vstack(Xs_train_intg)
        self.X_test_integ  = np.vstack(Xs_test_intg)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        self.logger.info(f"nonlinear integrate (RBF+Linear): "
                        f"X_train {self.X_train_integ.shape}, "
                        f"X_test {self.X_test_integ.shape}, "
                        f"lambda={lam}, approx={'1st' if USE_FIRST_ORDER else 'exact'}")
        print("統合表現の次元数:", self.X_train_integ.shape[1])
    """
    # ---------------------------------------------------------------
    # 〈非線形統合〉  RBF ⊕ Linear（※線形のバイアス項なし）
    #     g(x)= αᵀ k_rbf(x,·) + βᵀx
    #     零空間 = { βᵀx }（無罰則）, 有罰則 = α（λ‖α‖²）
    # ---------------------------------------------------------------
    def make_integrate_nonlinear_linear(self) -> None:

        import numpy as np
        from numpy.linalg import eigvalsh, inv, pinv
        from sklearn.metrics.pairwise import rbf_kernel

        m     = len(self.anchors_inter)
        r     = self.anchors_inter[0].shape[0]
        p_hat = self.config.dim_integrate
        lam   = getattr(self.config, "nl_lambda", 1e-2)

        Ks, gammas, Ps_lambda = [], [], []
        K_scales = []
        I_r = np.eye(r)

        # ---- gamma の用意（既存ロジックのまま）----
        if self.config.gamma_type == "auto":
            for S_tilde in self.anchors_inter:
                gammas.append(1.0 / S_tilde.shape[1])
        elif self.config.gamma_type == "X_tuning":
            for X_train_inter in self.Xs_train_inter:
                gamma = self_tuning_gamma(X_train_inter, standardize=False, k=3, summary='median')
                gammas.append(gamma)
        else:
            # フォールバック
            for S_tilde in self.anchors_inter:
                gammas.append(1.0 / S_tilde.shape[1])

        # ---- 1. K と P_λ（厳密 or 一次近似） ----
        # λ大きいことに寄る誤差はほとんどないが、一次近似すると計算時間が短縮される
        USE_FIRST_ORDER = (lam >= 10.0)

        for i, S_tilde in enumerate(self.anchors_inter):
            gamma = gammas[i]

            # (a) RBF カーネル行列と正規化
            K_rbf_raw = rbf_kernel(S_tilde, S_tilde, gamma=gamma)        # (r,r)
            mu_max = max(eigvalsh(K_rbf_raw).max(), 1e-12)               # スペクトル半径
            K_rbf = K_rbf_raw / mu_max                                   # ||K||_2 = 1
            K_scales.append(mu_max)

            # (b) ★ 線形基底（バイアス無し）
            P_lin = S_tilde                                              # ★ (r, d_k)

            if USE_FIRST_ORDER:
                # 一次近似：P_λ ≈ P_linProj + (1/λ) P^(1)
                G = P_lin.T @ P_lin
                G_inv = pinv(G)
                P_linProj = P_lin @ G_inv @ P_lin.T                      # (r,r)

                # P^(1) = K - K P - P K + P K P   （K は正規化済）
                P1 = K_rbf - K_rbf @ P_linProj - P_linProj @ K_rbf + P_linProj @ K_rbf @ P_linProj
                P_lambda = P_linProj + (1.0/lam) * P1

                coeff_mode = "first_order"
                coeff_pack = (G_inv, P_linProj)                          # 係数再計算用
            else:
                # 厳密計算
                A_inv = inv(K_rbf + lam * I_r)                           # (K + λI)^(-1)
                try:
                    M = inv(P_lin.T @ A_inv @ P_lin)                     # ★ (d_k,d_k)
                except np.linalg.LinAlgError:
                    M = pinv(P_lin.T @ A_inv @ P_lin)

                # P_λ = K A^{-1} + (P - K A^{-1} P) M Pᵀ A^{-1}
                P_lambda = (K_rbf @ A_inv
                            + (P_lin - K_rbf @ A_inv @ P_lin) @ M @ (P_lin.T @ A_inv))
                coeff_mode = "exact"
                coeff_pack = (A_inv, M)

            Ks.append((K_rbf, P_lin, coeff_mode, coeff_pack, mu_max))
            Ps_lambda.append(P_lambda)

        # ---- 2. 共通ターゲット Z（固有値問題）----
        M_tot = sum((P_l - I_r).T @ (P_l - I_r) for P_l in Ps_lambda)
        M_sym = 0.5 * (M_tot + M_tot.T)
        eigvals, eigvecs = np.linalg.eigh(M_sym)

        Z = eigvecs[:, eigvals.argsort()[:p_hat]]
        # 列正規化（任意）
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz

        # ---- 3. 各機関データを写像 ----
        Xs_train_intg, Xs_test_intg = [], []

        for (K_rbf, P_lin, coeff_mode, coeff_pack, mu_max), S_tilde, gamma, X_tr, X_te in zip(
                Ks, self.anchors_inter, gammas, self.Xs_train_inter, self.Xs_test_inter):

            if coeff_mode == "exact":
                A_inv, M = coeff_pack
                beta  = M @ (P_lin.T @ A_inv @ Z)                          # ★ (d_k, p̂)
                alpha = A_inv @ (Z - P_lin @ beta)                         # (r, p̂)
            else:
                # 一次近似
                G_inv, P_linProj = coeff_pack
                beta0 = G_inv @ (P_lin.T @ Z)                              # ★ (d_k, p̂)
                r0    = Z - P_lin @ beta0
                beta1 = G_inv @ (P_lin.T @ (K_rbf @ r0))                   # ★ (d_k, p̂)
                beta  = beta0 + (1.0/lam) * beta1
                alpha = (1.0/lam) * r0

            # --- 埋め込み（K(x,S) も同じスケールで正規化）---
            def embed(X):
                Kx_raw = rbf_kernel(X, S_tilde, gamma=gamma)               # (n,r)
                Kx = Kx_raw / mu_max
                Px = X                                                     # ★ (n, d_k)  ← バイアス無し
                return Kx @ alpha + Px @ beta

            Xs_train_intg.append(embed(X_tr))
            Xs_test_intg .append(embed(X_te))

        # ---- 4. スタック & 保存 ----
        self.X_train_integ = np.vstack(Xs_train_intg)
        self.X_test_integ  = np.vstack(Xs_test_intg)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        self.logger.info(
            f"nonlinear integrate (RBF + Linear[no-bias]): "
            f"X_train {self.X_train_integ.shape}, X_test {self.X_test_integ.shape}, "
            f"lambda={lam}, approx={'1st' if USE_FIRST_ORDER else 'exact'}"
        )
        print("統合表現の次元数:", self.X_train_integ.shape[1])

 
        
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 〈非線形統合〉 10× / 0.1× ステップで λ・γ_k を探索する座標勾配アルゴリズム
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 〈非形統合〉 γ_k のみ 10× / 0.1× 探索（λ=1e-9 固定）
    # ------------------------------------------------------------------
    def make_integrate_nonlinear_expression_tuning(self) -> None:
        """
        * λ は 1e-9 に固定。
        * 機関別カーネル幅 γ_k のみを勾配符号に沿って 1 桁スケール移動で最適化。
        * 改善なければ方向反転しステップ幅を対数で 1/2 に縮小。
        * 同じパラメータで 2 回縮小が起これば収束とみなす。
        """
        import math

        import numpy as np
        from numpy.linalg import eigh, inv, norm
        from sklearn.metrics.pairwise import rbf_kernel

        # ------------------ 基本設定 ------------------------------------------
        LAMBDA_FIXED = 1e-9                               # λ を固定
        m = len(self.anchors_inter)                       # 機関数
        r = self.anchors_inter[0].shape[0]                # アンカー行数
        p_hat = self.config.dim_integrate                 # 統合表現次元

        # 5:5 split ------------------------------------------------------------
        r_tr = r // 2
        tr_idx = np.arange(r_tr)
        val_idx = np.arange(r_tr, r)
        I_tr = np.eye(r_tr)
        I_r = np.eye(r)

        # 距離^2 行列をキャッシュ ----------------------------------------------
        sq_dists = []
        for S in self.anchors_inter:
            G = S @ S.T
            diag = np.diag(G)
            sq_dists.append(diag[:, None] + diag[None, :] - 2 * G)

        # -------------------- val loss 関数 (λ 固定) --------------------------
        def val_loss(log_gammas: np.ndarray) -> float:
            gammas = np.exp(log_gammas)

            P_tr_list, P_val_list = [], []
            for dist, gamma in zip(sq_dists, gammas):
                K_tr = np.exp(-gamma * dist[np.ix_(tr_idx, tr_idx)])
                K_val_tr = np.exp(-gamma * dist[np.ix_(val_idx, tr_idx)])
                A = inv(K_tr + LAMBDA_FIXED * I_tr)
                P_tr_list.append(K_tr @ A)
                P_val_list.append(K_val_tr @ A)

            # --- 固有値問題 (train) -----------------------------------------
            M_tr = np.zeros((r_tr, r_tr))
            for P_tr in P_tr_list:
                M_tr += (I_tr - P_tr).T @ (I_tr - P_tr)
            M_tr = 0.5 * (M_tr + M_tr.T)
            eigvals, eigvecs = eigh(M_tr)
            Z_tr = eigvecs[:, np.argsort(eigvals)[:p_hat]]
            Z_tr /= norm(Z_tr, 'fro')

            # val 側の予測平均
            Z_val_pred = sum(P_val @ Z_tr for P_val in P_val_list) / m

            loss = 0.0
            for P_val in P_val_list:
                E = P_val @ Z_tr - Z_val_pred
                loss += np.sum(E ** 2)
            return loss

        # --------------------- 勾配 (有限差分) -------------------------------
        def finite_grad(fun, x, eps=1e-4):
            g = np.zeros_like(x)
            for i in range(len(x)):
                e = np.zeros_like(x)
                e[i] = eps
                g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
            return g

        # ------------------- 初期値とステップ幅 ------------------------------
        gammas0 = [1.0 / S.shape[1] for S in self.anchors_inter]
        log_gammas = np.log(np.array(gammas0, dtype=float))
        print(gammas0)
        step_sizes = np.full_like(log_gammas, math.log(10.0))   # ln10
        shrink_counts = np.zeros_like(log_gammas, dtype=int)
        min_step = math.log(1.5)                                # ≈0.405
        max_iter = 300

        curr_loss = val_loss(log_gammas)

        for it in range(max_iter):
            grad = finite_grad(val_loss, log_gammas)
            improved_any = False
            for i in range(len(log_gammas)):
                if shrink_counts[i] >= 2:
                    continue   # この γ_k は収束済み

                direction = -np.sign(grad[i])
                if direction == 0:
                    shrink_counts[i] = 2  # フラット => 収束
                    continue

                # 1 桁移動 (×10 or ÷10)
                trial = log_gammas.copy()
                trial[i] += direction * step_sizes[i]
                loss1 = val_loss(trial)

                if loss1 < curr_loss:     # 改善
                    log_gammas = trial
                    curr_loss = loss1
                    improved_any = True
                    continue

                # 方向反転
                trial2 = log_gammas.copy()
                trial2[i] -= direction * step_sizes[i]
                loss2 = val_loss(trial2)
                if loss2 < curr_loss:
                    log_gammas = trial2
                    curr_loss = loss2
                    improved_any = True
                else:
                    # ステップ幅半減 (対数的に 1/2) and count ++
                    step_sizes[i] *= 0.5
                    shrink_counts[i] += 1
                print(log_gammas)
            # 全 γ_k が 2 回縮小済みなら収束
            if np.all(shrink_counts >= 2):
                break
            # 1 イテレーションで何も改善しなければ再度勾配計算で続行
            if not improved_any and np.all(step_sizes < min_step):
                break

        # ------------------ 最終統合（固定 λ, 最適 γ） -----------------------
        gammas_opt = np.exp(log_gammas)
        print("最適化された γ_k:", gammas_opt)
        Ks, Ps = [], []
        for S, gamma in zip(self.anchors_inter, gammas_opt):
            K = rbf_kernel(S, S, gamma=gamma)
            Ks.append(K)
            Ps.append(K @ inv(K + LAMBDA_FIXED * I_r))

        M_full = np.zeros((r, r))
        for P in Ps:
            M_full += (I_r - P).T @ (I_r - P)
        M_full = 0.5 * (M_full + M_full.T)
        eigvals_full, eigvecs_full = eigh(M_full)
        Z = eigvecs_full[:, np.argsort(eigvals_full)[:p_hat]]
        Z /= norm(Z, 'fro')
        
        # ---- 各機関を統合空間に射影 ----------------------------------------
        Xs_train_intg, Xs_test_intg = [], []
        for S, K, gamma, X_tr, X_te in zip(self.anchors_inter, Ks, gammas_opt,
                                           self.Xs_train_inter, self.Xs_test_inter):
            Bk = inv(K + LAMBDA_FIXED * I_r) @ Z
            Xs_train_intg.append(rbf_kernel(X_tr, S, gamma=gamma) @ Bk)
            Xs_test_intg.append(rbf_kernel(X_te, S, gamma=gamma) @ Bk)

        self.X_train_integ = np.vstack(Xs_train_intg)
        self.X_test_integ = np.vstack(Xs_test_intg)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)

        # --------------- 保存 & ログ -----------------------------------------
        eig_top = eigvals_full[np.argsort(eigvals_full)[:p_hat]].sum()
        self.config.g_abs_sum = f"{eig_top:.4g}"
        self.config.nl_lambda_opt = LAMBDA_FIXED
        self.config.nl_gamma_opt = [float(g) for g in gammas_opt]
        self.logger.info(f"[nonlinear integrate] λ fixed = {LAMBDA_FIXED}")
        self.logger.info(f"[nonlinear integrate] opt γ = {[round(g,5) for g in gammas_opt]}")
        self.logger.info(f"[nonlinear integrate] X_train {self.X_train_integ.shape}")
        self.logger.info(f"[nonlinear integrate] X_test  {self.X_test_integ.shape}")
        print("統合表現の次元数:", self.X_train_integ.shape[1])
        print(gammas_opt)

        # 固有値の小さい順に p_hat 個選択
        lambdas = eigvals_full[:p_hat]  # 固有値の上位 p_hat 個

        # 固有値の総和を計算
        sum_lambdas = np.sum(lambdas)

        # 結果を self.config.g_abs_sum に格納
        self.config.g_abs_sum = f"{sum_lambdas:.4g}"

        # デバッグ用出力
        print(f"固有値 λ の上位 {p_hat} 個の総和: {self.config.g_abs_sum}")
        

    def build_init_from_gen_eig(self):
        """
        make_integrate_expression_gen_eig() のロジックに基づき、
        交互最適化の初期解 G_init と beta_init を生成して self に保存する。
        """
        print("******************** 初期解生成 (from GEP) ********************")
        from functools import reduce
        from scipy.linalg import block_diag, eigh

        # --- 0. 設定とデータ ---
        m     = self.config.num_institution
        p_hat = self.config.dim_integrate
        Ss    = self.anchors_inter
        Xs    = self.Xs_train_inter
        ys    = self.ys_train

        # --- 1. GEPを解いて V_sel を得る ---
        W_s_tilde = np.hstack(Ss)
        blocks    = [S.T @ S for S in Ss]
        epsilon   = 1e-6
        B_s_tilde = reduce(lambda a, b: block_diag(a, b), blocks) + epsilon * np.eye(sum(S.shape[1] for S in Ss))
        A_s_tilde = 2 * m * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde)
        
        if self.config.orth_ver:
            eigvals, eigvecs = eigh(A_s_tilde)
        else:
            eigvals, eigvecs = eigh(A_s_tilde, B_s_tilde)
        order = np.argsort(eigvals)
        V_sel = eigvecs[:, order[:p_hat]]  # (Σd_k, p_hat)

        # --- 2. V_sel を G_init として整形し、制約を満たすように正規化 ---
        dims = [S.shape[1] for S in Ss]
        cum  = np.cumsum([0] + dims)
        G_init = V_sel.copy()

        # 制約: Σ_k ||S_k g_jk||^2 = 1 に合わせて各列を正規化
        for j in range(p_hat):
            vj = G_init[:, j]
            norm_sq = 0.0
            for k in range(m):
                vjk = vj[cum[k]:cum[k+1]]
                norm_sq += vjk.T @ blocks[k] @ vjk
            
            if norm_sq > 1e-9:
                G_init[:, j] /= np.sqrt(norm_sq)

        # --- 3. 生成した G_init を使って beta_init を最小二乗法で計算 ---
        X_blocks = []
        for k in range(m):
            Gk = G_init[cum[k]:cum[k+1], :]
            X_blocks.append(Xs[k] @ Gk)
        X_all = np.vstack(X_blocks)
        y_all = np.hstack(ys)

        try:
            beta_init, _, _, _ = np.linalg.lstsq(X_all, y_all, rcond=None)
        except np.linalg.LinAlgError:
            beta_init = np.zeros(p_hat)

        # --- 4. 初期解をインスタンス変数として保存 ---
        self.G_init    = G_init
        self.beta_init = beta_init
        print("✅ 初期解 G_init と beta_init を生成しました。")
        print(f"  G_init.shape: {self.G_init.shape}, beta_init.shape: {self.beta_init.shape}")
        
    def make_integrate_gen_eig_linear_fitting_objective(self) -> None:
        """
        Gurobiなし・勾配法(Adam)で手法2を最小化:
        min_{β,G} Σ_k ||y^(k) - X'^(k) G^(k) β||^2
                    + λ Σ_j Σ_{k,k'} ||S'^(k) g_j^(k) - S'^(k') g_j^(k')||^2
        s.t.      Σ_k ||S'^(k) g_j^(k)||^2 = 1 (j=1..p̂)
        制約は各ステップ後の列正規化で満たす（投影付き最適化）。
        """
        import numpy as np
        import torch

        print("******************** 統合表現 (GD / Adam) ********************")

        # -------------------- 0) 設定 --------------------
        m       = int(self.config.num_institution)
        p_hat   = int(self.config.dim_integrate)
        lam     = float(getattr(self.config, "lambda_gurobi", 1.0))

        # 勾配法ハイパーパラメータ（必要ならconfigから拾ってください）
        max_it = int(getattr(self.config, "gd_max_iter", None) or 2000)
        lr = float(getattr(self.config, "gd_lr", None) or 1e-2)
        betas = getattr(self.config, "gd_betas", None) or (0.9, 0.999)
        weight_decay = float(getattr(self.config, "gd_weight_decay", None) or 0.0)
        grad_clip = float(getattr(self.config, "gd_grad_clip", None) or 0.0)  # 0 なら無効
        print_every = int(getattr(self.config, "gd_print_every", None) or 100)
        max_it = int(getattr(self.config, "gd_max_iter", None) or 2000)
        seed    = getattr(self.config, "gd_seed", None) or 0
        torch.manual_seed(seed)
        np.random.seed(seed)

        Xs = self.Xs_train_inter  # list of (n_k × d_k)
        Ss = self.anchors_inter   # list of (r   × d_k)
        ys = self.ys_train        # list of (n_k,)

        # 形状など
        assert len(Xs) == m and len(Ss) == m and len(ys) == m, "Xs/Ss/ys の長さ m が一致しません。"
        dims = [S.shape[1] for S in Ss]      # d_k
        dtot = int(sum(dims))
        cum  = np.cumsum([0] + dims)

        # -------------------- 1) Tensor化（double精度） --------------------
        device = torch.device("cpu")
        dtype  = torch.double

        X_list = [torch.tensor(X, dtype=dtype, device=device) for X in Xs]
        S_list = [torch.tensor(S, dtype=dtype, device=device) for S in Ss]
        y_list = [torch.tensor(y, dtype=dtype, device=device) for y in ys]
        STS_list = [S.T @ S for S in S_list]  # d_k×d_k

        # -------------------- 2) 初期値 --------------------
        # 既に自前の初期値があれば利用
        G_init = getattr(self, "G_init", None)
        beta_init = getattr(self, "beta_init", None)

        if G_init is None or G_init.shape != (dtot, p_hat):
            G_np = 0.01 * np.random.randn(dtot, p_hat)
        else:
            G_np = np.array(G_init, copy=True)

        # 制約に合わせて列ごとに正規化: Σ_k ||S_k g_jk||^2 = 1
        for j in range(p_hat):
            vj = G_np[:, j].copy()
            norm2 = 0.0
            for k in range(m):
                vjk = vj[cum[k]:cum[k+1]]
                norm2 += float(vjk.T @ (STS_list[k].cpu().numpy()) @ vjk)
            if norm2 > 0:
                G_np[:, j] = vj / np.sqrt(norm2)

        if beta_init is None or beta_init.shape != (p_hat,):
            # LS で初期化
            X_blocks = [Xs[k] @ G_np[cum[k]:cum[k+1], :] for k in range(m)]
            X_all = np.vstack(X_blocks)
            y_all = np.hstack(ys)
            try:
                beta_np = np.linalg.lstsq(X_all, y_all, rcond=None)[0]
            except Exception:
                beta_np = np.zeros(p_hat)
        else:
            beta_np = np.array(beta_init, copy=True)

        # 学習変数
        G = torch.tensor(G_np, dtype=dtype, device=device, requires_grad=True)        # (dtot × p_hat)
        beta = torch.tensor(beta_np, dtype=dtype, device=device, requires_grad=True)  # (p_hat,)

        opt = torch.optim.Adam([{"params": [G], "lr": lr},
                                {"params": [beta], "lr": lr}],
                            betas=betas, weight_decay=weight_decay)

        eps = 1e-12

        def loss_fn(G, beta):
            # 第1項: Σ_k || y^(k) - X^(k) G^(k) β ||^2
            se = 0.0 * beta.sum()  # dummy to keep torch type
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]                # d_k × p_hat
                pred_k = X_list[k] @ (Gk @ beta)          # n_k
                res_k  = y_list[k] - pred_k               # n_k
                se = se + (res_k @ res_k)

            # 第2項: λ Σ_j ( 2m Σ_k g_jk^T S_k^T S_k g_jk - 2 || Σ_k S_k g_jk ||^2 )
            align = 0.0 * beta.sum()
            for j in range(p_hat):
                gj = G[:, j]                              # (dtot,)
                term1 = 0.0 * beta.sum()
                sumS  = 0.0
                # sumS は r 次元テンソルを使う必要があるので、一旦 torchベクトルで保持
                # r は各S_kで同一前提
                rdim = S_list[0].shape[0]
                sumS_vec = torch.zeros(rdim, dtype=dtype, device=device)
                for k in range(m):
                    gjk = gj[cum[k]:cum[k+1]]            # (d_k,)
                    term1 = term1 + (gjk @ (STS_list[k] @ gjk))
                    sumS_vec = sumS_vec + (S_list[k] @ gjk)
                align = align + (2.0 * m) * term1 - 2.0 * (sumS_vec @ sumS_vec)

            return se + lam * align

        # -------------------- 3) 最適化ループ（投影付き） --------------------
        last_val = None
        for it in range(1, max_it + 1):
            opt.zero_grad()
            val = loss_fn(G, beta)
            val.backward()

            # 勾配クリップ（必要なら）
            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([G, beta], max_norm=grad_clip)

            opt.step()

            # 射影: 各列 j について Σ_k ||S_k g_jk||^2 = 1 に正規化
            with torch.no_grad():
                for j in range(p_hat):
                    gj = G[:, j]
                    norm2 = torch.tensor(0.0, dtype=dtype, device=device)
                    for k in range(m):
                        gjk = gj[cum[k]:cum[k+1]]
                        norm2 = norm2 + (gjk @ (STS_list[k] @ gjk))
                    scale = torch.sqrt(torch.clamp(norm2, min=eps))
                    G[:, j] = gj / scale

            if print_every and (it % print_every == 0 or it == 1):
                v = float(val.detach().cpu().numpy())
                if last_val is None:
                    print(f"iter {it:5d} | obj={v:.6g}")
                else:
                    rel = abs(last_val - v) / (1.0 + abs(last_val))
                    print(f"iter {it:5d} | obj={v:.6g} | relΔ={rel:.3e}")
                    # 収束判定（任意）：tol が config にあれば利用
                    tol = float(getattr(self.config, "gd_tol", None) or 1e-6)
                    if rel < tol:
                        print("Converged by tolerance.")
                        # 早期停止
                        break
                last_val = v

        # -------------------- 4) 結果の取り出し＆保存 --------------------
        G_opt = G.detach().cpu().numpy()
        beta_opt = beta.detach().cpu().numpy()

        # 埋め込み計算
        Xs_train_integrate, Xs_test_integrate = [], []
        for k in range(m):
            Gk = G_opt[cum[k]:cum[k+1], :]
            Xs_train_integrate.append(self.Xs_train_inter[k] @ Gk)
            Xs_test_integrate.append(self.Xs_test_inter[k]  @ Gk)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

        # オプション：初期値として保持（次回のウォームスタートに使える）
        self.G_init = G_opt
        self.beta_init = beta_opt


    def make_integrate_gen_eig_fitting_objective(self) -> None:
        """
        NN(θ) + 統合行列 G を同時最適化（Adam）。
        目的:  λ*J_pred(θ,G) + J_reg(G) + μ*J_B-ortho(G)
        制約:  diag(G^T B G) = 1  （列ごと B-ノルム=1 を射影で課す）
        罰則:  J_B-ortho = Σ_{i≠j} (v_i^T B v_j)^2
        """
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim.lr_scheduler as lr_scheduler

        # ---------- config ----------
        m      = int(self.config.num_institution)
        p_hat  = int(self.config.dim_integrate)

        # パラメータでバージョンを切り替え
        orth_ver = bool(getattr(self.config, "orth_ver", None) or False)

        if orth_ver:
            print("***** 統合表現 (NN + ユークリッドノルム制約 + 直交罰則) *****")
        else:
            print("***** 統合表現 (NN + Bノルム制約 + B直交罰則) *****")


        lam_pred   = float(getattr(self.config, "lambda_pred", None) or 0)
        lam_reg    = float(getattr(self.config, "lambda_reg", None) or 1.0) # ★ Jregの重みを追加
        mu_off     = float(getattr(self.config, "lambda_offdiag", None) or 0.0)

        max_it      = int(getattr(self.config, "gd_max_iter", None) or 1000)
        lr_theta    = float(getattr(self.config, "gd_lr_theta", None) or getattr(self.config, "gd_lr", None) or 1e-3)
        lr_G        = float(getattr(self.config, "gd_lr_G", None) or getattr(self.config, "gd_lr", None) or 1e-2)
        betas       = getattr(self.config, "gd_betas", None) or (0.9, 0.999)
        weight_decay= float(getattr(self.config, "gd_weight_decay", None) or 0.0)
        grad_clip   = float(getattr(self.config, "gd_grad_clip", None) or 0.0)
        tol         = float(getattr(self.config, "gd_tol", None) or 1e-6)
        print_every = int(getattr(self.config, "gd_print_every", None) or 100)
        seed        = int(getattr(self.config, "gd_seed", None) or 0)
        
        # ★ 学習率スケジューラの設定
        lr_step_size = int(getattr(self.config, "lr_step_size", None) or 500)
        lr_gamma     = float(getattr(self.config, "lr_gamma", None) or 0.5)

        torch.manual_seed(seed); np.random.seed(seed)

        # ---------- data (inter を優先) ----------
        S_list   = getattr(self, "anchors_inter", None) or self.anchors
        Xtr_list = getattr(self, "Xs_train_inter", None) or self.Xs_train
        Xte_list = getattr(self, "Xs_test_inter",  None) or self.Xs_test
        ytr_list = self.ys_train

        dims = [S.shape[1] for S in S_list]; dtot = int(sum(dims)); cum = np.cumsum([0]+dims)
        rdim = S_list[0].shape[0]
        STS_np = [S.T @ S for S in S_list]

        num_classes = int(max(int(y.max()) for y in ytr_list) + 1)

        # ---------- tensors ----------
        device = torch.device("cpu")
        dtype  = torch.double
        Xtr = [torch.tensor(X, dtype=dtype, device=device) for X in Xtr_list]
        ytr = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in ytr_list]
        S   = [torch.tensor(Si, dtype=dtype, device=device) for Si in S_list]
        STS = [torch.tensor(M,  dtype=dtype, device=device) for M in STS_np]

        # ---------- NN model ----------
        hidden = getattr(self.config, "nn_hidden", None) or [256]
        layers, in_dim = [], p_hat
        for h in hidden:
            layers += [nn.Linear(in_dim, h, bias=True), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes, bias=True)]
        model = nn.Sequential(*layers).to(device).double()

        # ---------- init G & projection (Bノルム=1 列ごと) ----------
        G_init = getattr(self, "G_init", None)
        if G_init is None or G_init.shape != (dtot, p_hat):
            G_np = 0.01 * np.random.randn(dtot, p_hat)
        else:
            G_np = np.array(G_init, copy=True)

        def project_columns(G_np: np.ndarray) -> np.ndarray:
            for j in range(p_hat):
                vj = G_np[:, j].copy()
                norm2 = 0.0
                for k in range(m):
                    vjk = vj[cum[k]:cum[k+1]]
                    if orth_ver:
                        norm2 += float(vjk.T @ vjk)
                    else:
                        norm2 += float(vjk.T @ STS_np[k] @ vjk)
                if norm2 > 1e-9:
                    G_np[:, j] = vj / np.sqrt(norm2)
            #print("cj=", np.sqrt(norm2), "1に近いか？")
            return G_np

        G_np = project_columns(G_np)
        G = torch.tensor(G_np, dtype=dtype, device=device, requires_grad=True)

        # ---------- helpers ----------
        ce = nn.CrossEntropyLoss(reduction="sum")

        def reg_term(G: torch.Tensor) -> torch.Tensor:
            tot = torch.tensor(0.0, dtype=dtype, device=device)
            for j in range(p_hat):
                gj   = G[:, j]
                term1 = torch.tensor(0.0, dtype=dtype, device=device)
                sumS  = torch.zeros(rdim, dtype=dtype, device=device)
                for k in range(m):
                    gjk = gj[cum[k]:cum[k+1]]
                    term1 = term1 + (gjk @ (STS[k] @ gjk))
                    sumS  = sumS  + (S[k] @ gjk)
                tot = tot + (2.0*m)*term1 - 2.0*(sumS @ sumS)
            return tot

        def B_times_G(G: torch.Tensor) -> torch.Tensor:
            blocks = []
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                blocks.append(STS[k] @ Gk)
            return torch.vstack(blocks)

        def offdiag_penalty(G: torch.Tensor) -> torch.Tensor:
            if mu_off == 0.0:
                return torch.tensor(0.0, dtype=dtype, device=device)
            
            if orth_ver:
                M = G.T @ G
            else:
                BG = B_times_G(G)
                M  = G.T @ BG
            
            off = M - torch.diag(torch.diag(M))
            return mu_off * torch.sum(off**2)


        # ---------- optimizer & scheduler ★ ----------
        opt = torch.optim.Adam(
            [{"params": model.parameters(), "lr": lr_theta, "weight_decay": weight_decay},
             {"params": [G], "lr": lr_G, "weight_decay": 0.0}],
            betas=betas
        )
        scheduler = lr_scheduler.StepLR(opt, step_size=lr_step_size, gamma=lr_gamma)

        # ---------- training loop ----------
        last_obj = None
        for it in range(1, max_it + 1):
            opt.zero_grad()

            Jpred = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                logits = model(Xtr[k] @ Gk)
                Jpred += ce(logits, ytr[k])

            Jreg  = reg_term(G)
            Joff  = offdiag_penalty(G)

            obj = lam_pred * Jpred + lam_reg * Jreg + Joff
            obj.backward()

            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [G], max_norm=grad_clip)

            opt.step()
            scheduler.step() # ★ スケジューラを更新

            with torch.no_grad():
                G_np = project_columns(G.detach().cpu().numpy())
                G.data[:] = torch.tensor(G_np, dtype=dtype, device=device)

            if print_every and (it % print_every == 0 or it == 1 or it == max_it):
                val = float(obj.detach().cpu().numpy())
                # ★ 勾配ノルムと学習率をログに追加
                g_norm_G = torch.norm(G.grad).item() if G.grad is not None else 0.0
                g_norm_theta_list = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
                g_norm_theta = torch.norm(torch.cat(g_norm_theta_list)).item() if g_norm_theta_list else 0.0
                current_lr = scheduler.get_last_lr()[0]

                if last_obj is None:
                    print(f"iter {it:5d} | J={val:.6g} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(lam_reg*Jreg):.6g} | Joff={float(Joff):.6g} | ∇G={g_norm_G:.2e} | ∇θ={g_norm_theta:.2e} | lr={current_lr:.2e}")
                else:
                    rel = abs(last_obj - val) / (1.0 + abs(last_obj))
                    print(f"iter {it:5d} | J={val:.6g} | relΔ={rel:.3e} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(lam_reg*Jreg):.6g} | Joff={float(Joff):.6g} | ∇G={g_norm_G:.2e} | ∇θ={g_norm_theta:.2e} | lr={current_lr:.2e}")
                    if rel < tol:
                        print("Converged by tolerance."); break
                last_obj = val

        # ---------- build embeddings & save results ----------
        G_opt = G.detach().cpu().numpy()
        Xs_train_integrate, Xs_test_integrate = [], []
        for k in range(m):
            Gk = G_opt[cum[k]:cum[k+1], :]
            Xs_train_integrate.append(self.Xs_train_inter[k] @ Gk)
            Xs_test_integrate.append(self.Xs_test_inter[k]  @ Gk)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)
        self.G_init = G_opt

        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")
        
                # --- [検証] G_optがGEPの解の性質を満たすか ---
        print("\n--- [検証] G_opt がGEPの解の性質を満たすか ---")
        from scipy.linalg import block_diag
        from numpy.linalg import norm

        # 1. 行列 A, B を構築
        Ss_np = self.anchors_inter
        SkT_Sk_np = [S.T @ S for S in Ss_np]
        
        # B = blkdiag(S_k^T S_k)
        B_s_tilde = block_diag(*SkT_Sk_np)

        # --- [検証] B_s_tilde の正定値性 ---
        try:
            eigvals_B = np.linalg.eigvalsh(B_s_tilde)
            min_eig_B = np.min(eigvals_B)
            print(f"\n[検証] B_s_tilde の最小固有値: {min_eig_B:.6g}")
            if min_eig_B > 0:
                print("[検証] B_s_tilde は正定値です。\n")
            else:
                print("[検証] B_s_tilde は正定値ではありません。\n")
        except np.linalg.LinAlgError:
            print("[検証] B_s_tilde の固有値計算に失敗しました。\n")
        
        # W = hstack(S_k)
        W_s_tilde = np.hstack(Ss_np)
        
        # A = 2m B - 2 W^T W
        A_s_tilde = 2 * m * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde)
        #print("A_s_tilde", A_s_tilde)
        #print("B_s_tilde", B_s_tilde)

        # 2. G_opt の各列 v_j について検証
        eigenvalues_from_v = []
        for j in range(p_hat):
            v_j = G_opt[:, j]
            
            if j > 0:
                v_prev = G_opt[:, j-1]
                # ユークリッド内積ベースのコサイン類似度
                cos_sim_vs = (v_j @ v_prev) / (norm(v_j) * norm(v_prev))
                print(f"v_{j} vs v_{j-1}: ユークリッド コサイン類似度 = {cos_sim_vs:.6f}")
                
                # B-内積 (罰則項が0にしようとする値)
                b_inner_product = v_j.T @ B_s_tilde @ v_prev
                print(f"v_{j} vs v_{j-1}: B-内積 v_j^T B v_{j-1} = {b_inner_product:.6f} (→0に近いか？)")

            # (7)式: v^T B v = 1 の検証
            norm_val = v_j.T @ B_s_tilde @ v_j
            print(f"v_{j}: 制約 v^T B v = {norm_val:.6f} (→1に近いか？)")

            # (8)式: Av = λBv の検証
            Av = A_s_tilde @ v_j
            Bv = B_s_tilde @ v_j
            
            # Av と Bv のコサイン類似度で平行か確認
            cos_sim = (Av @ Bv) / (norm(Av) * norm(Bv))
            print(f"v_{j}: Av と Bv のコサイン類似度 = {cos_sim:.6f} (→1に近いか？)")

            # 対応する固有値 λ を計算
            lambda_j = v_j.T @ A_s_tilde @ v_j
            eigenvalues_from_v.append(lambda_j)
            print(f"v_{j}: 対応する固有値 λ = {lambda_j:.6f}")
            print("-" * 20)

        # 3. 固有値の和を計算
        sum_lambda = np.sum(eigenvalues_from_v)
        print(f"計算された固有値の総和: {sum_lambda:.6f}")
        
        # ★【追加】Av - λBv の差のノルムを計算
        diff_norm = norm(Av - lambda_j * Bv)
        print(f"v_{j}: ||Av - λBv|| のノルム = {diff_norm:.6f} (→0に近いか？)")
        
        # Jregの値を取得して比較
        jreg_val_str = getattr(self.config, 'jreg_gep', 'N/A')
        print(f"目的関数の正則化項 Jreg: {jreg_val_str} (→固有値の総和と近いか？)")
        print("--- [検証] 終了 ---\n")

    def make_integrate_gen_eig_logi_fitting_objective(self) -> None:

        """
        線形モデル + 統合行列 G を同時最適化（Adam）。
        目的:  J_pred(β,G) + J_reg(G) + μ*J_ortho(G)
        モデル: J_predはロジスティック回帰(分類) or 線形回帰(回帰)の損失
        制約:  列ごとのノルム=1 を射影で課す
        """
        import numpy as np
        import torch
        import torch.nn as nn

        # ---------- config ----------
        m      = int(self.config.num_institution)
        p_hat  = int(self.config.dim_integrate)
        
        orth_ver = bool(getattr(self.config, "orth_ver", None) or False)

        if orth_ver:
            print("***** 統合表現 (線形モデル + ユークリッドノルム制約 + 直交罰則) *****")
        else:
            print("***** 統合表現 (線形モデル + Bノルム制約 + B直交罰則) *****")

        lam_pred   = float(getattr(self.config, "lambda_pred", None) or 1.0)
        mu_off     = float(getattr(self.config, "lambda_b_offdiag", None) or 10000.0)

        max_it      = int(getattr(self.config, "gd_max_iter", None) or 2000)
        lr_beta     = float(getattr(self.config, "gd_lr_beta", None) or getattr(self.config, "gd_lr", None) or 1e-3)
        lr_G        = float(getattr(self.config, "gd_lr_G", None) or getattr(self.config, "gd_lr", None) or 1e-2)
        betas       = getattr(self.config, "gd_betas", None) or (0.9, 0.999)
        weight_decay= float(getattr(self.config, "gd_weight_decay", None) or 0.0)
        grad_clip   = float(getattr(self.config, "gd_grad_clip", None) or 0.0)
        tol         = float(getattr(self.config, "gd_tol", None) or 1e-6)
        print_every = int(getattr(self.config, "gd_print_every", None) or 100)
        seed        = int(getattr(self.config, "gd_seed", None) or 0)
        
        early_stopping_patience = int(getattr(self.config, "early_stopping_patience", None) or 10)
        use_early_stopping = early_stopping_patience > 0

        torch.manual_seed(seed); np.random.seed(seed)

        # ---------- data (inter を優先) ----------
        S_list   = getattr(self, "anchors_inter", None) or self.anchors
        Xtr_list = getattr(self, "Xs_train_inter", None) or self.Xs_train
        Xte_list = getattr(self, "Xs_test_inter",  None) or self.Xs_test
        ytr_list = self.ys_train
        yte_list = self.ys_test

        dims = [S.shape[1] for S in S_list]; dtot = int(sum(dims)); cum = np.cumsum([0]+dims)
        rdim = S_list[0].shape[0]
        STS_np = [S.T @ S for S in S_list]

        # --- ★ 問題の種類を判別 (分類 or 回帰) ---
        is_classification = np.issubdtype(ytr_list[0].dtype, np.integer)
        if is_classification:
            print("問題を「分類」として扱います。")
            num_classes = int(max(int(y.max()) for y in ytr_list) + 1)
            # 2クラス問題も多クラスとして扱う（CrossEntropyLossが対応）
            if num_classes < 2: num_classes = 2 
        else:
            print("問題を「回帰」として扱います。")
            num_classes = 1

        # ---------- tensors ----------
        device = torch.device("cpu")
        dtype  = torch.double
        Xtr = [torch.tensor(X, dtype=dtype, device=device) for X in Xtr_list]
        Xte = [torch.tensor(X, dtype=dtype, device=device) for X in Xte_list]
        if is_classification:
            ytr = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in ytr_list]
            yte = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in yte_list]
        else: # 回帰
            ytr = [torch.tensor(y, dtype=dtype, device=device) for y in ytr_list]
            yte = [torch.tensor(y, dtype=dtype, device=device) for y in yte_list]
        S   = [torch.tensor(Si, dtype=dtype, device=device) for Si in S_list]
        STS = [torch.tensor(M,  dtype=dtype, device=device) for M in STS_np]

        # ---------- ★ 線形モデルのパラメータ beta を初期化 ----------
        beta_init = getattr(self, "beta_init", None)
        beta_shape = (p_hat, num_classes) if num_classes > 1 else (p_hat,)
        
        if beta_init is not None and beta_init.shape == beta_shape:
            beta_np = np.array(beta_init, copy=True)
        else:
            beta_np = 0.01 * np.random.randn(*beta_shape)
        
        beta = torch.tensor(beta_np, dtype=dtype, device=device, requires_grad=True)

        # ---------- init G & projection ----------
        G_init = getattr(self, "G_init", None)
        if G_init is None or G_init.shape != (dtot, p_hat):
            G_np = 0.01 * np.random.randn(dtot, p_hat)
        else:
            G_np = np.array(G_init, copy=True)

        def project_columns(G_np: np.ndarray) -> np.ndarray:
            for j in range(p_hat):
                vj = G_np[:, j].copy()
                norm2 = 0.0
                for k in range(m):
                    vjk = vj[cum[k]:cum[k+1]]
                    if orth_ver:
                        norm2 += float(vjk.T @ vjk)
                    else:
                        norm2 += float(vjk.T @ STS_np[k] @ vjk)
                if norm2 > 1e-9:
                    G_np[:, j] = vj / np.sqrt(norm2)
            return G_np

        G_np = project_columns(G_np)
        G = torch.tensor(G_np, dtype=dtype, device=device, requires_grad=True)

        # ---------- helpers ----------
        loss_func = nn.CrossEntropyLoss(reduction="sum") if is_classification else nn.MSELoss(reduction="sum")

        def reg_term(G: torch.Tensor) -> torch.Tensor:
            tot = torch.tensor(0.0, dtype=dtype, device=device)
            for j in range(p_hat):
                gj   = G[:, j]
                term1 = torch.tensor(0.0, dtype=dtype, device=device)
                sumS  = torch.zeros(rdim, dtype=dtype, device=device)
                for k in range(m):
                    gjk = gj[cum[k]:cum[k+1]]
                    if orth_ver:
                        term1 = term1 + (gjk @ gjk)
                    else:
                        term1 = term1 + (gjk @ (STS[k] @ gjk))
                    sumS  = sumS  + (S[k] @ gjk)
                tot = tot + (2.0*m)*term1 - 2.0*(sumS @ sumS)
            return tot

        def B_times_G(G: torch.Tensor) -> torch.Tensor:
            blocks = []
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                blocks.append(STS[k] @ Gk)
            return torch.vstack(blocks)

        def offdiag_penalty(G: torch.Tensor) -> torch.Tensor:
            if mu_off == 0.0: return torch.tensor(0.0, dtype=dtype, device=device)
            M = G.T @ G if orth_ver else G.T @ B_times_G(G)
            off = M - torch.diag(torch.diag(M))
            return mu_off * torch.sum(off**2)

        # ---------- optimizer ----------
        opt = torch.optim.Adam(
            [{"params": [beta], "lr": lr_beta, "weight_decay": weight_decay},
             {"params": [G], "lr": lr_G, "weight_decay": 0.0}],
            betas=betas
        )

        # ---------- training loop ----------
        last_obj = None
        if use_early_stopping:
            best_val_loss = float('inf')
            patience_counter = 0
            best_beta_state = None
            best_G_state = None
            print(f"Early stopping is enabled with patience = {early_stopping_patience}.")

        for it in range(1, max_it + 1):
            opt.zero_grad()

            Jpred = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                Zk = Xtr[k] @ Gk
                pred = Zk @ beta
                if not is_classification: pred = pred.squeeze(-1)
                Jpred += loss_func(pred, ytr[k])

            Jreg  = reg_term(G)
            Joff  = offdiag_penalty(G)

            obj = lam_pred * Jpred + Jreg + Joff
            obj.backward()

            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([beta, G], max_norm=grad_clip)

            opt.step()

            with torch.no_grad():
                G_np = project_columns(G.detach().cpu().numpy())
                G.data[:] = torch.tensor(G_np, dtype=dtype, device=device)

            if print_every and (it % print_every == 0 or it == 1 or it == max_it):
                val = float(obj.detach().cpu().numpy())
                log_msg = ""
                if last_obj is None:
                    log_msg = f"iter {it:5d} | J(train)={val:.6g} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(Jreg):.6g} | Joff={float(Joff):.6g}"
                else:
                    rel = abs(last_obj - val) / (1.0 + abs(last_obj))
                    log_msg = f"iter {it:5d} | J(train)={val:.6g} | relΔ={rel:.3e} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(Jreg):.6g} | Joff={float(Joff):.6g}"
                    if not use_early_stopping and rel < tol:
                        print(log_msg); print("Converged by training objective tolerance."); break
                last_obj = val
                
                if use_early_stopping:
                    with torch.no_grad():
                        val_loss = torch.tensor(0.0, dtype=dtype, device=device)
                        for k in range(m):
                            Gk = G[cum[k]:cum[k+1], :]
                            Zk_val = Xte[k] @ Gk
                            pred_val = Zk_val @ beta
                            if not is_classification: pred_val = pred_val.squeeze(-1)
                            val_loss += loss_func(pred_val, yte[k])
                    
                    val_loss_item = val_loss.item()
                    log_msg += f" | J(val)={val_loss_item:.6g}"
                    
                    if val_loss_item < best_val_loss:
                        best_val_loss = val_loss_item
                        patience_counter = 0
                        best_beta_state = beta.clone().detach()
                        best_G_state = G.clone().detach()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(log_msg); print(f"Early stopping triggered after {patience_counter} checks."); break
                
                print(log_msg)

        if use_early_stopping and best_beta_state is not None:
            print(f"\nRestoring model to the best state with validation loss: {best_val_loss:.6g}")
            beta.data[:] = best_beta_state
            G.data[:] = best_G_state

        # ---------- build embeddings ----------
        G_opt = G.detach().cpu().numpy()
        Xs_train_integrate, Xs_test_integrate = [], []
        for k in range(m):
            Gk = G_opt[cum[k]:cum[k+1], :]
            Xs_train_integrate.append((self.Xs_train_inter if hasattr(self, "Xs_train_inter") and self.Xs_train_inter else self.Xs_train)[k] @ Gk)
            Xs_test_integrate.append((self.Xs_test_inter if hasattr(self, "Xs_test_inter") and self.Xs_test_inter else self.Xs_test)[k]   @ Gk)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)
        self.G_init = G_opt
        self.beta_init = beta.detach().cpu().numpy()

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")
        
                # --- [検証] G_optがGEPの解の性質を満たすか ---
        print("\n--- [検証] G_opt がGEPの解の性質を満たすか ---")
        from scipy.linalg import block_diag
        from numpy.linalg import norm

        # 1. 行列 A, B を構築
        Ss_np = self.anchors_inter
        SkT_Sk_np = [S.T @ S for S in Ss_np]
        
        # B = blkdiag(S_k^T S_k)
        B_s_tilde = block_diag(*SkT_Sk_np)

        # --- [検証] B_s_tilde の正定値性 ---
        try:
            eigvals_B = np.linalg.eigvalsh(B_s_tilde)
            min_eig_B = np.min(eigvals_B)
            print(f"\n[検証] B_s_tilde の最小固有値: {min_eig_B:.6g}")
            if min_eig_B > 0:
                print("[検証] B_s_tilde は正定値です。\n")
            else:
                print("[検証] B_s_tilde は正定値ではありません。\n")
        except np.linalg.LinAlgError:
            print("[検証] B_s_tilde の固有値計算に失敗しました。\n")
        
        # W = hstack(S_k)
        W_s_tilde = np.hstack(Ss_np)
        
        # A = 2m B - 2 W^T W
        A_s_tilde = 2 * m * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde)
        #print("A_s_tilde", A_s_tilde)
        #print("B_s_tilde", B_s_tilde)

        # 2. G_opt の各列 v_j について検証
        eigenvalues_from_v = []
        for j in range(p_hat):
            v_j = G_opt[:, j]
            print(f"v_{j}", v_j)
            
            if j > 0:
                v_prev = G_opt[:, j-1]
                # ユークリッド内積ベースのコサイン類似度
                cos_sim_vs = (v_j @ v_prev) / (norm(v_j) * norm(v_prev))
                print(f"v_{j} vs v_{j-1}: ユークリッド コサイン類似度 = {cos_sim_vs:.6f}")
                
                # B-内積 (罰則項が0にしようとする値)
                b_inner_product = v_j.T @ B_s_tilde @ v_prev
                print(f"v_{j} vs v_{j-1}: B-内積 v_j^T B v_{j-1} = {b_inner_product:.6f} (→0に近いか？)")

            # (7)式: v^T B v = 1 の検証
            norm_val = v_j.T @ B_s_tilde @ v_j
            print(f"v_{j}: 制約 v^T B v = {norm_val:.6f} (→1に近いか？)")

            # (8)式: Av = λBv の検証
            Av = A_s_tilde @ v_j
            Bv = B_s_tilde @ v_j
            
            # Av と Bv のコサイン類似度で平行か確認
            cos_sim = (Av @ Bv) / (norm(Av) * norm(Bv))
            print(f"v_{j}: Av と Bv のコサイン類似度 = {cos_sim:.6f} (→1に近いか？)")

            # 対応する固有値 λ を計算
            lambda_j = v_j.T @ A_s_tilde @ v_j
            eigenvalues_from_v.append(lambda_j)
            print(f"v_{j}: 対応する固有値 λ = {lambda_j:.6f}")
            print("-" * 20)

        # 3. 固有値の和を計算
        sum_lambda = np.sum(eigenvalues_from_v)
        print(f"計算された固有値の総和: {sum_lambda:.6f}")
        
        # ★【追加】Av - λBv の差のノルムを計算
        diff_norm = norm(Av - lambda_j * Bv)
        print(f"v_{j}: ||Av - λBv|| のノルム = {diff_norm:.6f} (→0に近いか？)")
        
        # Jregの値を取得して比較
        jreg_val_str = getattr(self.config, 'jreg_gep', 'N/A')
        print(f"目的関数の正則化項 Jreg: {jreg_val_str} (→固有値の総和と近いか？)")
        print("--- [検証] 終了 ---\n")
        
    def make_integrate_gen_eig_fitting_objective_ortho(self) -> None:
        """
        分類版 + 直交「制約」：G^T B G = I を毎ステップのリトラクションで厳密に満たす。
        J(θ,G) = Σ_k CE(y^(k), NN(X^(k) G^(k);θ))
                + λ * Σ_j [ 2m Σ_k g_jk^T S_k^T S_k g_jk - 2 ||Σ_k S_k g_jk||^2 ]
                (+ ρ ||G||_F^2 は任意、既定は 0)

        制約:
        G^T B G = I  （B = blkdiag(S_k^T S_k)）
        実装:
        勾配更新 → B-計量 Gram–Schmidt で列直交化（リトラクション）
        保存:
        self.X_train_integ / self.X_test_integ / self.y_*_integ, self.G_init
        """
        import numpy as np
        import torch
        import torch.nn as nn

        print("******************** 統合表現 (NN + B-直交制約) ********************")

        # -------------------- 0) 設定とデータ --------------------
        m        = int(self.config.num_institution)
        p_hat    = int(self.config.dim_integrate)
        lam      = float(getattr(self.config, "lambda_nn_reg", None)
                        or getattr(self.config, "lambda_gurobi", None) or 1.0)

        # ハイパラ（デフォルト安全）
        max_it      = int(getattr(self.config, "gd_max_iter", None) or 2000)
        lr_theta    = float(getattr(self.config, "gd_lr_theta", None)
                            or getattr(self.config, "gd_lr", None) or 1e-3)
        lr_G        = float(getattr(self.config, "gd_lr_G", None)
                            or getattr(self.config, "gd_lr", None) or 1e-2)
        betas       = getattr(self.config, "gd_betas", None) or (0.9, 0.999)
        weight_decay= float(getattr(self.config, "gd_weight_decay", None) or 0.0)
        grad_clip   = float(getattr(self.config, "gd_grad_clip", None) or 0.0)
        tol         = float(getattr(self.config, "gd_tol", None) or 1e-6)
        print_every = int(getattr(self.config, "gd_print_every", None) or 100)
        seed        = int(getattr(self.config, "gd_seed", None) or 0)
        rho_g_l2    = float(getattr(self.config, "lambda_g_l2", None) or 0.0)  # 任意

        torch.manual_seed(seed); np.random.seed(seed)

        # ソース（_inter があれば優先）
        S_list   = getattr(self, "anchors_inter", None) or self.anchors
        Xtr_list = getattr(self, "Xs_train_inter", None) or self.Xs_train
        Xte_list = getattr(self, "Xs_test_inter",  None) or self.Xs_test
        ytr_list = self.ys_train
        yte_list = self.ys_test

        assert len(S_list)==m and len(Xtr_list)==m and len(Xte_list)==m, "機関数 m とリスト長が不一致です。"

        dims = [S.shape[1] for S in S_list]  # d_k
        dtot = int(sum(dims))
        cum  = np.cumsum([0]+dims)
        rdim = S_list[0].shape[0]

        # -------------------- 1) Tensor 化 & B (= blkdiag(S^T S)) --------------------
        device = torch.device("cpu")
        dtype  = torch.double

        Xtr = [torch.tensor(X, dtype=dtype, device=device) for X in Xtr_list]
        Xte = [torch.tensor(X, dtype=dtype, device=device) for X in Xte_list]
        ytr = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in ytr_list]
        yte = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in yte_list]
        S   = [torch.tensor(Si, dtype=dtype, device=device) for Si in S_list]

        # B の各ブロック S_k^T S_k（数値安定のため εI を加える）
        eps_spd = 1e-10
        STS = []
        for k in range(m):
            M = S[k].T @ S[k]
            M = M + eps_spd * torch.eye(M.shape[0], dtype=dtype, device=device)
            STS.append(M)

        # -------------------- 2) モデル & 変数 --------------------
        num_classes = int(max(int(y.max()) for y in ytr_list) + 1)
        hidden = getattr(self.config, "nn_hidden", None) or [64, 32]
        layers = []; in_dim = p_hat
        for h in hidden:
            layers += [nn.Linear(in_dim, h, bias=True), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes, bias=True)]
        model = nn.Sequential(*layers).to(device).double()

        # 初期 G（ウォームスタートがあれば使用）
        G_init = getattr(self, "G_init", None)
        if G_init is None or G_init.shape!=(dtot, p_hat):
            G_np = 0.01 * np.random.randn(dtot, p_hat)
        else:
            G_np = np.array(G_init, copy=True)
            print(1111111111111111111111111111111111111111111)

        # --- B-直交正規化ヘルパ ---
        def b_dot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """<u,v>_B = u^T B v を計算（u,v は dtot）"""
            s = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(m):
                uk = u[cum[k]:cum[k+1]]
                vk = v[cum[k]:cum[k+1]]
                s  = s + (uk @ (STS[k] @ vk))
            return s

        def b_norm(u: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(torch.clamp(b_dot(u,u), min=1e-18))

        def b_orthonormalize_blockwise_(G: torch.Tensor) -> None:
            """
            各ブロック k について:
            C_k = Gk^T (S_k^T S_k) Gk を Cholesky（or EVD）で逆平方根し、
            Gk <- Gk C_k^{-1/2} により (Gk^T B_k Gk) = I を満たす。
            """
            with torch.no_grad():
                for k in range(m):
                    Gk = G[cum[k]:cum[k+1], :]                 # d_k × p̂
                    # C_k = Gk^T B_k Gk
                    Ck = Gk.T @ (STS[k] @ Gk)                  # p̂ × p̂
                    Ck = 0.5 * (Ck + Ck.T)                     # 対称化
                    eps = 1e-12
                    Iph = torch.eye(Ck.shape[0], dtype=Ck.dtype, device=Ck.device)
                    try:
                        R = torch.linalg.cholesky(Ck + eps*Iph)    # Ck = R^T R
                        Rinv = torch.linalg.inv(R)
                        Gk_new = Gk @ Rinv                         # Gk R^{-1} ⇒ (Gk^T B_k Gk) = I
                    except RuntimeError:
                        # 数値不安定なら固有値分解で逆平方根
                        w, V = torch.linalg.eigh(Ck)
                        w = torch.clamp(w, min=1e-12)
                        Ck_inv_sqrt = V @ torch.diag(1.0/torch.sqrt(w)) @ V.T
                        Gk_new = Gk @ Ck_inv_sqrt
                    # 書き戻し
                    G[cum[k]:cum[k+1], :].copy_(Gk_new)

        # 初期 G を B-直交化
        G = torch.tensor(G_np, dtype=dtype, device=device, requires_grad=True)
        b_orthonormalize_blockwise_(G)  # ここで G^T B G ≈ I

        # 最適化器（θ と G の学習率を分ける）
        opt = torch.optim.Adam(
            [{"params": model.parameters(), "lr": lr_theta},
            {"params": [G], "lr": lr_G}],
            betas=betas, weight_decay=weight_decay
        )
        ce = nn.CrossEntropyLoss(reduction="sum")

        # 整合化項
        def reg_term(G: torch.Tensor) -> torch.Tensor:
            tot = torch.tensor(0.0, dtype=dtype, device=device)
            for j in range(p_hat):
                gj = G[:, j]
                term1 = torch.tensor(0.0, dtype=dtype, device=device)
                sumS  = torch.zeros(rdim, dtype=dtype, device=device)
                for k in range(m):
                    gjk = gj[cum[k]:cum[k+1]]
                    term1 = term1 + (gjk @ (STS[k] @ gjk))
                    sumS  = sumS + (S[k] @ gjk)
                tot = tot + (2.0 * m) * term1 - 2.0 * (sumS @ sumS)
            return tot

        last_obj = None

        # -------------------- 3) 学習ループ（投影付き最適化） --------------------
        for it in range(1, max_it+1):
            opt.zero_grad()

            # 分類損
            Jpred = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]                 # d_k × p̂
                logits_k = model(Xtr[k] @ Gk)              # n_k × C
                Jpred = Jpred + ce(logits_k, ytr[k])

            # 整合化 + 任意 Ridge
            Jreg   = reg_term(G)
            Jridge = rho_g_l2 * (G**2).sum()
            obj    = Jpred + lam * Jreg + Jridge

            obj.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [G], grad_clip)
            opt.step()

            # --- B-直交制約を厳密に満たす（リトラクション） ---
            b_orthonormalize_blockwise_(G)   # ⇒ ここで常に G^T B G = I に戻す

            # ログ＆収束
            if print_every and (it % print_every == 0 or it == 1 or it == max_it):
                val = float(obj.detach().cpu().numpy())
                jp  = float(Jpred.detach().cpu().numpy())
                jr  = float(Jreg.detach().cpu().numpy())
                jw  = float(Jridge.detach().cpu().numpy())
                # 直交誤差の確認
                with torch.no_grad():
                    # gram = G^T B G の Frobenius 誤差
                    gram = torch.zeros((p_hat, p_hat), dtype=dtype, device=device)
                    # gram[i,j] = <g_i, g_j>_B
                    for i in range(p_hat):
                        for j2 in range(p_hat):
                            gram[i, j2] = b_dot(G[:, i], G[:, j2])
                    ortho_err = float(torch.sum((gram - torch.eye(p_hat, dtype=dtype, device=device))**2).cpu().numpy())
                if last_obj is None:
                    print(f"iter {it:5d} | J={val:.6g} | Jpred={jp:.6g} | Jreg={jr:.6g} | Jridge={jw:.6g} | "
                        f"||G^TBG-I||_F^2={ortho_err:.3e}")
                else:
                    rel = abs(last_obj - val) / (1.0 + abs(last_obj))
                    print(f"iter {it:5d} | J={val:.6g} | relΔ={rel:.3e} | Jpred={jp:.6g} | Jreg={jr:.6g} | "
                        f"Jridge={jw:.6g} | ||G^TBG-I||_F^2={ortho_err:.3e}")
                    if rel < tol:
                        print("Converged by tolerance.")
                        break
                last_obj = val

        # -------------------- 4) 埋め込み計算＆保存 --------------------
        G_opt = G.detach().cpu().numpy()
        Xs_train_integrate, Xs_test_integrate = [], []
        for k in range(m):
            Gk = G_opt[cum[k]:cum[k+1], :]
            Xs_train_integrate.append((self.Xs_train_inter[k] if getattr(self, "Xs_train_inter", None) is not None
                                    else self.Xs_train[k]) @ Gk)
            Xs_test_integrate.append((self.Xs_test_inter[k] if getattr(self, "Xs_test_inter", None) is not None
                                    else self.Xs_test[k]) @ Gk)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)

        # 後段の初期値として保持
        self.G_init = G_opt

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

    def make_integrate_gen_eig_fitting_objective_(self) -> None:
        """
        NN(θ) + 統合行列 G を同時最適化（Adam）。
        目的:  λ*J_pred(θ,V) + tr(V^T Â_s V) + μ ||OffDiag(V^T B̃_s V)||_F^2
        制約:  diag(V^T B̃_s V) = 1  （列ごと B-ノルム=1 を射影で課す）
        orth_ver=True の場合、B-ノルム/B-直交の代わりにユークリッドノルム/直交を扱う
        """
        import numpy as np
        import torch
        import torch.nn as nn

        print("***** 統合表現 (NN + Bノルム制約 + B直交オフ対角罰則) *****")

        # ---------- config ----------
        m      = int(self.config.num_institution)
        p_hat  = int(self.config.dim_integrate)
        
        # パラメータでバージョンを切り替え
        orth_ver = bool(getattr(self.config, "orth_ver", None) or False)

        if orth_ver:
            print("***** 統合表現 (NN + ユークリッドノルム制約 + 直交罰則) *****")
        else:
            print("***** 統合表現 (NN + Bノルム制約 + B直交罰則) *****")

        lam_pred   = float(getattr(self.config, "lambda_pred", None) or 0.0)   # 予測損の重み λ
        mu_off     = float(getattr(self.config, "lambda_b_offdiag", None) or 10000.0)  # B直交(オフ対角)の重み μ

        max_it      = int(getattr(self.config, "gd_max_iter", None) or 2000)
        lr_theta    = float(getattr(self.config, "gd_lr_theta", None) or getattr(self.config, "gd_lr", None) or 1e-3)
        lr_G        = float(getattr(self.config, "gd_lr_G", None) or getattr(self.config, "gd_lr", None) or 1e-2)
        betas       = getattr(self.config, "gd_betas", None) or (0.9, 0.999)
        weight_decay= float(getattr(self.config, "gd_weight_decay", None) or 0.0)
        grad_clip   = float(getattr(self.config, "gd_grad_clip", None) or 0.0)
        tol         = float(getattr(self.config, "gd_tol", None) or 1e-6)
        print_every = int(getattr(self.config, "gd_print_every", None) or 100)
        seed        = int(getattr(self.config, "gd_seed", None) or 0)

        torch.manual_seed(seed); np.random.seed(seed)

        # ---------- data (inter を優先) ----------
        S_list   = getattr(self, "anchors_inter", None) or self.anchors
        Xtr_list = getattr(self, "Xs_train_inter", None) or self.Xs_train
        Xte_list = getattr(self, "Xs_test_inter",  None) or self.Xs_test
        ytr_list = self.ys_train
        yte_list = self.ys_test

        assert len(S_list)==m and len(Xtr_list)==m and len(Xte_list)==m, "m とリスト長の不一致。"
        dims = [S.shape[1] for S in S_list]; dtot = int(sum(dims)); cum = np.cumsum([0]+dims)
        rdim = S_list[0].shape[0]
        STS_np = [S.T @ S for S in S_list]                        # B_k

        num_classes = int(max(int(y.max()) for y in ytr_list) + 1)

        # ---------- tensors ----------
        device = torch.device("cpu")
        dtype  = torch.double
        Xtr = [torch.tensor(X, dtype=dtype, device=device) for X in Xtr_list]
        Xte = [torch.tensor(X, dtype=dtype, device=device) for X in Xte_list]
        ytr = [torch.tensor(y.astype(int), dtype=torch.long, device=device) for y in ytr_list]
        S   = [torch.tensor(Si, dtype=dtype, device=device) for Si in S_list]
        STS = [torch.tensor(M,  dtype=dtype, device=device) for M in STS_np]

        # ---------- NN model ----------
        hidden = getattr(self.config, "nn_hidden", None) or [64, 32]
        layers, in_dim = [], p_hat
        for h in hidden:
            layers += [nn.Linear(in_dim, h, bias=True), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes, bias=True)]
        model = nn.Sequential(*layers).to(device).double()

        # ---------- init G & projection (Bノルム=1 列ごと) ----------
        G_init = getattr(self, "G_init", None)
        if G_init is None or G_init.shape != (dtot, p_hat):
            G_np = 0.01 * np.random.randn(dtot, p_hat)
        else:
            G_np = np.array(G_init, copy=True)

        def project_columns(G_np: np.ndarray) -> np.ndarray:
            for j in range(p_hat):
                vj = G_np[:, j].copy()
                norm2 = 0.0
                for k in range(m):
                    vjk = vj[cum[k]:cum[k+1]]
                    if orth_ver:
                        norm2 += float(vjk.T @ vjk)
                    else:
                        norm2 += float(vjk.T @ STS_np[k] @ vjk)
                if norm2 > 1e-9:
                    G_np[:, j] = vj / np.sqrt(norm2)
            return G_np

        G_np = project_columns(G_np)
        G = torch.tensor(G_np, dtype=dtype, device=device, requires_grad=True)

        # ---------- helpers ----------
        ce = nn.CrossEntropyLoss(reduction="sum")

        def reg_term(G: torch.Tensor) -> torch.Tensor:
            # tr(V^T Â_s V) を明示的和で実装（既存と同じ）
            tot = torch.tensor(0.0, dtype=dtype, device=device)
            for j in range(p_hat):
                gj   = G[:, j]
                term1 = torch.tensor(0.0, dtype=dtype, device=device)
                sumS  = torch.zeros(rdim, dtype=dtype, device=device)
                for k in range(m):
                    gjk = gj[cum[k]:cum[k+1]]
                    term1 = term1 + (gjk @ (STS[k] @ gjk))   # g^T B_k g
                    sumS  = sumS  + (S[k] @ gjk)             # S_k g
                tot = tot + (2.0*m)*term1 - 2.0*(sumS @ sumS)
            return tot

        def B_times_G(G: torch.Tensor) -> torch.Tensor:
            # BG = blkdiag(B_k) @ G
            blocks = []
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                blocks.append(STS[k] @ Gk)
            return torch.vstack(blocks)

        def offdiag_penalty(G: torch.Tensor) -> torch.Tensor:
            if mu_off == 0.0:
                return torch.tensor(0.0, dtype=dtype, device=device)
            
            if orth_ver:
                M = G.T @ G
            else:
                BG = B_times_G(G)
                M  = G.T @ BG
            
            off = M - torch.diag(torch.diag(M))
            return mu_off * torch.sum(off**2)

        # ---------- optimizer ----------
        opt = torch.optim.Adam(
            [{"params": model.parameters(), "lr": lr_theta, "weight_decay": weight_decay},
            {"params": [G],               "lr": lr_G,     "weight_decay": 0.0}],
            betas=betas
        )

        # ---------- training loop ----------
        last_obj = None
        for it in range(1, max_it + 1):
            opt.zero_grad()

            # J_pred : Σ_k CE( NN(X_k G_k), y_k )
            Jpred = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(m):
                Gk = G[cum[k]:cum[k+1], :]
                logits = model(Xtr[k] @ Gk)
                Jpred += ce(logits, ytr[k])

            Jreg  = reg_term(G)
            Joff  = offdiag_penalty(G)

            obj = lam_pred * Jpred + Jreg + Joff
            obj.backward()

            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [G], max_norm=grad_clip)

            opt.step()

            # 制約: 列ごとに B-ノルム=1 へ射影
            with torch.no_grad():
                G_np = project_columns(G.detach().cpu().numpy())
                G.data[:] = torch.tensor(G_np, dtype=dtype, device=device)

            if print_every and (it % print_every == 0 or it == 1 or it == max_it):
                val = float(obj.detach().cpu().numpy())
                if last_obj is None:
                    print(f"iter {it:5d} | J={val:.6g} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(Jreg):.6g} | Joff={float(Joff):.6g}")
                else:
                    rel = abs(last_obj - val) / (1.0 + abs(last_obj))
                    print(f"iter {it:5d} | J={val:.6g} | relΔ={rel:.3e} | λJpred={float(lam_pred*Jpred):.6g} | Jreg={float(Jreg):.6g} | Joff={float(Joff):.6g}")
                    if rel < tol:
                        print("Converged by tolerance."); break
                last_obj = val

        # ---------- build embeddings ----------
        G_opt = G.detach().cpu().numpy()
        Xs_train_integrate, Xs_test_integrate = [], []
        for k in range(m):
            Gk = G_opt[cum[k]:cum[k+1], :]
            Xs_train_integrate.append((self.Xs_train_inter if getattr(self, "Xs_train_inter", None) is not None else self.Xs_train)[k] @ Gk)
            Xs_test_integrate.append((self.Xs_test_inter  if getattr(self, "Xs_test_inter",  None) is not None else self.Xs_test)[k]   @ Gk)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ  = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ  = np.hstack(self.ys_test)
        self.G_init = G_opt

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

    from numpy.linalg import cholesky, inv

    def cca_projection_matrix(S, Y, p_hat, ridge=0.0, rho_x=1e-6, rho_y=1e-6):
        # 中心化
        S = S - S.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)

        Sxx = S.T @ S + rho_x * np.eye(S.shape[1])
        Syy = Y.T @ Y + rho_y * np.eye(Y.shape[1])
        Sxy = S.T @ Y

        Lx = cholesky(Sxx); Ly = cholesky(Syy)
        C  = np.linalg.solve(Lx, Sxy) @ np.linalg.solve(Ly, np.eye(Ly.shape[0])).T
        # C = Lx^{-1} Sxy Ly^{-T}

        # SVD（対称ではないので普通の SVD でOK）
        U, sing, Vt = np.linalg.svd(C, full_matrices=False)
        A = np.linalg.solve(Lx.T, U[:, :p_hat])       # S側重み p x p_hat
        Uscore = S @ A                                # n x p_hat

        G = Uscore.T @ Uscore + ridge * np.eye(p_hat) # p_hat x p_hat
        P = Uscore @ np.linalg.solve(G, Uscore.T)     # n x n  （ハット行列）
        return P, A, Uscore

    def kcca_projection_matrix(Ks, Ky, p_hat, ridge=0.0, kx=1e-3, ky=1e-3):
        # 二重中心化
        n = Ks.shape[0]
        H = np.eye(n) - np.ones((n,n))/n
        Ks = H @ Ks @ H
        Ky = H @ Ky @ H

        # ホワイトニング
        # （数値安定のため固有分解で -1/2 を作ってもOK）
        from scipy.linalg import fractional_matrix_power
        Ks_mh = fractional_matrix_power(Ks + kx*np.eye(n), -0.5)
        Ky_mh = fractional_matrix_power(Ky + ky*np.eye(n), -0.5)

        C = Ks_mh @ Ks @ Ky @ Ky_mh
        U, sing, Vt = np.linalg.svd(C, full_matrices=False)
        A = Ks_mh @ U[:, :p_hat]               # n x p_hat（S側の双対係数）
        Uscore = Ks @ A                        # n x p_hat

        G = Uscore.T @ Uscore + ridge * np.eye(p_hat)
        P = Uscore @ np.linalg.solve(G, Uscore.T)   # n x n
        return P, A, Uscore

    def visualize_anchors(self, save_dir: Optional[str] = None) -> None:
        """
        アンカーデータの変換フローを訓練/テストの2部構成で可視化する。
        上半分(Train): 1.元, 2.中間, 3.射影, 4.統合Z
        下半分(Test):  1.元, 2.中間, 3.射影
        """
        save_dir = save_dir or self.config.output_path / "visualizations"
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        from sklearn.decomposition import PCA

        # --- 必要なデータの存在チェック ---
        train_attrs = ['anchor', 'anchors_inter', 'Z', 'anchors_integ']
        test_attrs = ['anchor_test', 'anchors_test_inter', 'anchors_test_integ']
        
        has_train_data = all(hasattr(self, attr) and getattr(self, attr) is not None and len(getattr(self, attr, [])) > 0 for attr in train_attrs)
        has_test_data = all(hasattr(self, attr) and getattr(self, attr) is not None and len(getattr(self, attr, [])) > 0 for attr in test_attrs)

        if not has_train_data and not has_test_data:
            self.logger.warning("可視化に必要な訓練データもテストデータも存在しません。")
            return

        num_institutions = len(self.anchors_inter) if has_train_data else len(self.anchors_test_inter)
        if num_institutions == 0: return

        # --- ラベルの準備 ---
        self.assign_anchor_labels()
        anchor_labels_train = self.anchor_y if hasattr(self, 'anchor_y') else np.zeros(self.anchor.shape[0] if has_train_data else 0)
        anchor_labels_test = self.anchor_y_test if hasattr(self, 'anchor_y_test') else np.zeros(self.anchor_test.shape[0] if has_test_data else 0)
        legend_status = "full" if np.unique(anchor_labels_train).size > 1 else False

        # --- プロットの準備 (Train+Testで2倍の行数) ---
        fig, axes = plt.subplots(num_institutions * 2, 4, figsize=(24, 6 * num_institutions * 2), squeeze=False)
        fig.suptitle("Anchor Data Transformation Flow (Top: Train, Bottom: Test)", fontsize=16, y=1.0)

        # --- PCAとスケール計算のためのデータ準備 ---
        Z_train_plot = self.Z.T if has_train_data and self.Z.shape[0] == self.config.dim_integrate else (self.Z if has_train_data else None)

        col1_data = ([self.anchor] if has_train_data else []) + ([self.anchor_test] if has_test_data else [])
        col2_data = (self.anchors_inter if has_train_data else []) + (self.anchors_test_inter if has_test_data else [])
        col3_data = (self.anchors_integ if has_train_data else []) + (self.anchors_test_integ if has_test_data else [])
        col4_data = [Z_train_plot] if has_train_data else []

        def get_2d_data_and_limits(data_list):
            if not data_list: return [], ((0,1), (0,1))
            data_for_pca = [d for d in data_list if d.shape[1] > 2]
            if not data_for_pca:
                data_2d = data_list
            else:
                pca = PCA(n_components=2).fit(np.vstack(data_for_pca))
                data_2d = [pca.transform(d) if d.shape[1] > 2 else d for d in data_list]
            
            all_data_2d = np.vstack(data_2d)
            x_min, x_max = all_data_2d[:, 0].min(), all_data_2d[:, 0].max()
            y_min, y_max = all_data_2d[:, 1].min(), all_data_2d[:, 1].max()
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            limits = ((x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad))
            return data_2d, limits

        col1_2d, (xlim1, ylim1) = get_2d_data_and_limits(col1_data)
        col2_2d, (xlim2, ylim2) = get_2d_data_and_limits(col2_data)
        col3_2d, (xlim3, ylim3) = get_2d_data_and_limits(col3_data)
        col4_2d, (xlim4, ylim4) = get_2d_data_and_limits(col4_data)

        # --- プロットループ ---
        for i in range(num_institutions):
            # --- TRAIN DATA (Top Half) ---
            if has_train_data:
                train_row = i
                sns.scatterplot(x=col1_2d[0][:, 0], y=col1_2d[0][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 0], legend=(i==0 and legend_status))
                axes[train_row, 0].set_title(f"1. Original Anchor (Train)" if i == 0 else "")
                axes[train_row, 0].set_xlim(xlim1); axes[train_row, 0].set_ylim(ylim1); axes[train_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(x=col2_2d[i][:, 0], y=col2_2d[i][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 1], legend=False)
                axes[train_row, 1].set_title(f"2. Intermediate (Train)" if i == 0 else "")
                axes[train_row, 1].set_xlim(xlim2); axes[train_row, 1].set_ylim(ylim2)

                sns.scatterplot(x=col3_2d[i][:, 0], y=col3_2d[i][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 2], legend=False)
                axes[train_row, 2].set_title(f"3. Projection S_hat (Train)" if i == 0 else "")
                axes[train_row, 2].set_xlim(xlim3); axes[train_row, 2].set_ylim(ylim3)

                sns.scatterplot(x=col4_2d[0][:, 0], y=col4_2d[0][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 3], legend=False)
                axes[train_row, 3].set_title(f"4. Integrated Z (Train)" if i == 0 else "")
                axes[train_row, 3].set_xlim(xlim4); axes[train_row, 3].set_ylim(ylim4)

            # --- TEST DATA (Bottom Half) ---
            if has_test_data:
                test_row = i + num_institutions
                train_offset = 1 if has_train_data else 0
                
                anchor_test_2d = col1_2d[train_offset]
                sns.scatterplot(x=anchor_test_2d[:, 0], y=anchor_test_2d[:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 0], legend=(i==0 and legend_status))
                axes[test_row, 0].set_title(f"1. Original Anchor (Test)" if i == 0 else "")
                axes[test_row, 0].set_xlim(xlim1); axes[test_row, 0].set_ylim(ylim1); axes[test_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(x=col2_2d[train_offset * num_institutions + i][:, 0], y=col2_2d[train_offset * num_institutions + i][:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 1], legend=False)
                axes[test_row, 1].set_title(f"2. Intermediate (Test)" if i == 0 else "")
                axes[test_row, 1].set_xlim(xlim2); axes[test_row, 1].set_ylim(ylim2)

                sns.scatterplot(x=col3_2d[train_offset * num_institutions + i][:, 0], y=col3_2d[train_offset * num_institutions + i][:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 2], legend=False)
                axes[test_row, 2].set_title(f"3. Projection S_hat (Test)" if i == 0 else "")
                axes[test_row, 2].set_xlim(xlim3); axes[test_row, 2].set_ylim(ylim3)
                
                # 4列目は空欄にする
                axes[test_row, 3].set_visible(False)

        # レイアウト調整と保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f"anchor_visualization_{self.config.plot_name}"
            plt.savefig(save_path)
            self.logger.info(f"✅ アンカーデータの可視化を保存しました: {save_path}")
    
    
    def visualize_representations(self, save_dir: Optional[str] = None) -> None:
        """
        元データ、中間表現、統合表現（機関ごとと全体）を2次元散布図で可視化する関数。
        訓練データとテストデータをそれぞれ別の図で出力する。
        """
        self.assign_anchor_labels()
        self.visualize_anchors() 
        
        save_dir = save_dir or self.config.output_path / "visualizations"
        if not self.Xs_train or not self.Xs_train_inter or self.X_train_integ.size == 0:
            print("可視化する表現が生成されていません。run()メソッドを実行してください。")
            return

        # 必要なライブラリのインポート
        import matplotlib.pyplot as plt
        import seaborn as sns

        num_institutions = self.config.num_institution

        # 統合表現を機関ごとに再分割
        train_sizes = [len(y) for y in self.ys_train]
        test_sizes = [len(y) for y in self.ys_test]
        train_indices = np.cumsum([0] + train_sizes)
        test_indices = np.cumsum([0] + test_sizes)

        Xs_train_integ_split = [self.X_train_integ[train_indices[i]:train_indices[i+1]] for i in range(num_institutions)]
        Xs_test_integ_split = [self.X_test_integ[test_indices[i]:test_indices[i+1]] for i in range(num_institutions)]

        # 統合表現プロットの軸スケールを統一するための範囲計算
        # Train
        x_min_train, x_max_train = self.X_train_integ[:, 0].min(), self.X_train_integ[:, 0].max()
        y_min_train, y_max_train = self.X_train_integ[:, 1].min(), self.X_train_integ[:, 1].max()
        x_pad_train = (x_max_train - x_min_train) * 0.05
        y_pad_train = (y_max_train - y_min_train) * 0.05
        xlim_train = (x_min_train - x_pad_train, x_max_train + x_pad_train)
        ylim_train = (y_min_train - y_pad_train, y_max_train + y_pad_train)

        # Test
        x_min_test, x_max_test = self.X_test_integ[:, 0].min(), self.X_test_integ[:, 0].max()
        y_min_test, y_max_test = self.X_test_integ[:, 1].min(), self.X_test_integ[:, 1].max()
        x_pad_test = (x_max_test - x_min_test) * 0.05
        y_pad_test = (y_max_test - y_min_test) * 0.05
        xlim_test = (x_min_test - x_pad_test, x_max_test + x_pad_test)
        ylim_test = (y_min_test - y_pad_test, y_max_test + y_pad_test)


        # --- 訓練データの可視化 ---
        fig_train, axes_train = plt.subplots(num_institutions, 4, figsize=(24, 5 * num_institutions), squeeze=False)
        fig_train.suptitle("Representations (Train Data)", fontsize=16)

        for i in range(num_institutions):
            # 1. 元データ (Train)
            sns.scatterplot(
                x=self.Xs_train[i][:, 0], y=self.Xs_train[i][:, 1], hue=self.ys_train[i],
                palette="viridis", ax=axes_train[i, 0], legend="full"
            )
            axes_train[i, 0].set_title(f"Institution {i+1} - Original Data")
            axes_train[i, 0].set_xlabel("Dimension 1")
            axes_train[i, 0].set_ylabel("Dimension 2")

            # 2. 中間表現 (Train)
            sns.scatterplot(
                x=self.Xs_train_inter[i][:, 0], y=self.Xs_train_inter[i][:, 1], hue=self.ys_train[i],
                palette="viridis", ax=axes_train[i, 1], legend="full"
            )
            axes_train[i, 1].set_title(f"Institution {i+1} - Intermediate Expression")
            axes_train[i, 1].set_xlabel("Dimension 1")
            axes_train[i, 1].set_ylabel("Dimension 2")

            # 3. 統合表現 (Train) - 機関ごと
            sns.scatterplot(
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=self.ys_train[i],
                palette="viridis", ax=axes_train[i, 2], legend="full"
            )
            axes_train[i, 2].set_title(f"Institution {i+1} - Integrated Expression")
            axes_train[i, 2].set_xlabel("Dimension 1")
            axes_train[i, 2].set_ylabel("Dimension 2")
            axes_train[i, 2].set_xlim(xlim_train)
            axes_train[i, 2].set_ylim(ylim_train)

            # 4. 統合表現 (Train) - 全機関（強調表示付き）
            other_institutions_indices = [j for j in range(num_institutions) if j != i]
            if other_institutions_indices:
                X_other = np.vstack([Xs_train_integ_split[j] for j in other_institutions_indices])
                y_other = np.hstack([self.ys_train[j] for j in other_institutions_indices])
                sns.scatterplot(
                    x=X_other[:, 0], y=X_other[:, 1], hue=y_other,
                    palette="viridis", ax=axes_train[i, 3], legend=False, alpha=1.0
                )
            sns.scatterplot(
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=self.ys_train[i],
                palette="viridis", ax=axes_train[i, 3], legend="full", alpha=1.0
            )
            axes_train[i, 3].set_title(f"All Institutions (Institution {i+1} Highlighted)")
            axes_train[i, 3].set_xlabel("Dimension 1")
            axes_train[i, 3].set_ylabel("Dimension 2")
            axes_train[i, 3].set_xlim(xlim_train)
            axes_train[i, 3].set_ylim(ylim_train)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / self.config.plot_name)
            
        
        """
        # --- テストデータの可視化 ---
        fig_test, axes_test = plt.subplots(num_institutions, 4, figsize=(24, 5 * num_institutions), squeeze=False)
        fig_test.suptitle("Representations (Test Data)", fontsize=16)

        for i in range(num_institutions):
            # 1. 元データ (Test)
            sns.scatterplot(
                x=self.Xs_test[i][:, 0], y=self.Xs_test[i][:, 1], hue=self.ys_test[i],
                palette="viridis", ax=axes_test[i, 0], legend="full"
            )
            axes_test[i, 0].set_title(f"Institution {i+1} - Original Data")
            axes_test[i, 0].set_xlabel("Dimension 1")
            axes_test[i, 0].set_ylabel("Dimension 2")

            # 2. 中間表現 (Test)
            sns.scatterplot(
                x=self.Xs_test_inter[i][:, 0], y=self.Xs_test_inter[i][:, 1], hue=self.ys_test[i],
                palette="viridis", ax=axes_test[i, 1], legend="full"
            )
            axes_test[i, 1].set_title(f"Institution {i+1} - Intermediate Expression")
            axes_test[i, 1].set_xlabel("Dimension 1")
            axes_test[i, 1].set_ylabel("Dimension 2")

            # 3. 統合表現 (Test) - 機関ごと
            sns.scatterplot(
                x=Xs_test_integ_split[i][:, 0], y=Xs_test_integ_split[i][:, 1], hue=self.ys_test[i],
                palette="viridis", ax=axes_test[i, 2], legend="full"
            )
            axes_test[i, 2].set_title(f"Institution {i+1} - Integrated Expression")
            axes_test[i, 2].set_xlabel("Dimension 1")
            axes_test[i, 2].set_ylabel("Dimension 2")
            axes_test[i, 2].set_xlim(xlim_test)
            axes_test[i, 2].set_ylim(ylim_test)

            # 4. 統合表現 (Test) - 全機関（強調表示付き）
            other_institutions_indices = [j for j in range(num_institutions) if j != i]
            if other_institutions_indices:
                X_other = np.vstack([Xs_test_integ_split[j] for j in other_institutions_indices])
                y_other = np.hstack([self.ys_test[j] for j in other_institutions_indices])
                sns.scatterplot(
                    x=X_other[:, 0], y=X_other[:, 1], hue=y_other,
                    palette="viridis", ax=axes_test[i, 3], legend=False, alpha=1.0
                )
            sns.scatterplot(
                x=Xs_test_integ_split[i][:, 0], y=Xs_test_integ_split[i][:, 1], hue=self.ys_test[i],
                palette="viridis", ax=axes_test[i, 3], legend="full", alpha=1.0
            )
            axes_test[i, 3].set_title(f"All Institutions (Institution {i+1} Highlighted)")
            axes_test[i, 3].set_xlabel("Dimension 1")
            axes_test[i, 3].set_ylabel("Dimension 2")
            axes_test[i, 3].set_xlim(xlim_test)
            axes_test[i, 3].set_ylim(ylim_test)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if save_dir:
            plt.savefig(Path(save_dir) / f"test_{self.config.G_type}_{self.config.nl_gamma}.png")
        """

    def save_representations_to_csv(self, save_dir: Optional[str] = None) -> None:
        """
        中間表現と統合表現をCSVファイルに保存する関数。
        """
        save_dir = Path(save_dir or self.config.output_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if not self.Xs_train_inter or self.X_train_integ.size == 0:
            self.logger.warning("保存する表現が生成されていません。run()メソッドを実行してください。")
            return

        num_institutions = self.config.num_institution

        # --- 中間表現の保存 ---
        intermediate_dfs = []
        for i in range(num_institutions):
            # Train
            df_train_inter = pd.DataFrame(self.Xs_train_inter[i], columns=[f'dim_{j+1}' for j in range(self.Xs_train_inter[i].shape[1])])
            df_train_inter['y'] = self.ys_train[i]
            df_train_inter['data_type'] = 'train'
            df_train_inter['institution'] = i
            intermediate_dfs.append(df_train_inter)

            # Test
            df_test_inter = pd.DataFrame(self.Xs_test_inter[i], columns=[f'dim_{j+1}' for j in range(self.Xs_test_inter[i].shape[1])])
            df_test_inter['y'] = self.ys_test[i]
            df_test_inter['data_type'] = 'test'
            df_test_inter['institution'] = i
            intermediate_dfs.append(df_test_inter)

        df_intermediate_all = pd.concat(intermediate_dfs, ignore_index=True)
        intermediate_save_path = save_dir / "intermediate_representations.csv"
        df_intermediate_all.to_csv(intermediate_save_path, index=False)
        self.logger.info(f"✅ 中間表現をCSVに保存しました: {intermediate_save_path}")


        # --- 統合表現の保存 ---
        # 統合表現を機関ごとに再分割
        train_sizes = [len(y) for y in self.ys_train]
        test_sizes = [len(y) for y in self.ys_test]
        train_indices = np.cumsum([0] + train_sizes)
        test_indices = np.cumsum([0] + test_sizes)

        Xs_train_integ_split = [self.X_train_integ[train_indices[i]:train_indices[i+1]] for i in range(num_institutions)]
        Xs_test_integ_split = [self.X_test_integ[test_indices[i]:test_indices[i+1]] for i in range(num_institutions)]

        integrated_dfs = []
        for i in range(num_institutions):
            # Train
            df_train_integ = pd.DataFrame(Xs_train_integ_split[i], columns=[f'dim_{j+1}' for j in range(Xs_train_integ_split[i].shape[1])])
            df_train_integ['y'] = self.ys_train[i]
            df_train_integ['data_type'] = 'train'
            df_train_integ['institution'] = i
            integrated_dfs.append(df_train_integ)

            # Test
            df_test_integ = pd.DataFrame(Xs_test_integ_split[i], columns=[f'dim_{j+1}' for j in range(Xs_test_integ_split[i].shape[1])])
            df_test_integ['y'] = self.ys_test[i]
            df_test_integ['data_type'] = 'test'
            df_test_integ['institution'] = i
            integrated_dfs.append(df_test_integ)

        df_integrated_all = pd.concat(integrated_dfs, ignore_index=True)
        integrated_save_path = save_dir / "integrated_representations.csv"
        df_integrated_all.to_csv(integrated_save_path, index=False)
        self.logger.info(f"✅ 統合表現をCSVに保存しました: {integrated_save_path}")

    def integrate_metrics(self, which: str = "test") -> dict:
        """
        anchors_[test_]integ の機関間ペアごとに:
          D_{ij} = A_i - A_j（行: サンプル, 列: 次元）
          行ごとの L2 距離 ||D_{ij}[n,:]||_2 を合計（sum, mean, maxも記録）
        結果を self.config.integ_metrics に保存して返す。
        
        Args:
            which: "test" -> self.anchors_test_integ を対象
                   "train"-> self.anchors_integ を対象
        Returns:
            dict: {"pairs": [...], "summary": {...}}
        """
        import numpy as np
        from itertools import combinations

        anchors_list = self.anchors_test_integ if which == "test" else self.anchors_integ

        if not anchors_list or len(anchors_list) < 2:
            self.logger.warning("integrate_metrics: 対象のアンカー統合表現が不足しています。")
            metrics = {"pairs": [], "summary": {}}
            self.config.integ_metrics = 100000
            return metrics

        results = []
        for i, j in combinations(range(len(anchors_list)), 2):
            Ai = anchors_list[i]
            Aj = anchors_list[j]

            if Ai is None or Aj is None or Ai.size == 0 or Aj.size == 0:
                self.logger.warning(f"integrate_metrics: 空の配列をスキップ (i={i}, j={j})")
                continue

            # 行数が異なる場合は小さい方に合わせる
            n = min(Ai.shape[0], Aj.shape[0])
            if (Ai.shape[0] != Aj.shape[0]) or (Ai.shape[1] != Aj.shape[1]):
                self.logger.warning(
                    f"integrate_metrics: 形状不一致 i={i}{Ai.shape}, j={j}{Aj.shape} -> "
                    f"先頭 {n} 行・共通次元に合わせて比較します。"
                )
            dmin = min(Ai.shape[1], Aj.shape[1])
            Di = Ai[:n, :dmin] - Aj[:n, :dmin]  # 行対応の差分
            row_dists = np.linalg.norm(Di, axis=1)  # 各サンプルの距離
            res = {
                "i": i,
                "j": j,
                "sum": float(row_dists.sum()),
                "mean": float(row_dists.mean()),
                "max": float(row_dists.max()),
                "n_rows_used": int(n),
                "dim_used": int(dmin),
            }
            results.append(res)

        if not results:
            metrics = {"pairs": [], "summary": {}}
            self.config.integ_metrics = 100000
            return metrics

        sums = np.array([r["sum"] for r in results], dtype=float)
        summary = {
            "pair_count": int(len(results)),
            "sum_mean": float(sums.mean()),
            "sum_min": float(sums.min()),
            "sum_max": float(sums.max()),
        }

        metrics = {"pairs": results, "summary": summary}
        self.config.integ_metrics = float(sums.mean())  # ← ここに保存
        self.config.integ_metrics = round(self.config.integ_metrics, 1)
        # 簡易出力
        print(f"[integrate_metrics/{which}] ペア数={summary['pair_count']}, "
              f"sum_mean={summary['sum_mean']:.6g}, "
              f"min={summary['sum_min']:.6g}, max={summary['sum_max']:.6g}")
        self.logger.info(f"[integrate_metrics/{which}] {summary}")

        return metrics