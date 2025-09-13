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
        self.assign_anchor_labels(k=5)
        
        
        
        # アンカーデータの生成
        self.anchor_test = self.produce_anchor(
            num_row=self.config.num_anchor_data, num_col=self.Xs_train[0].shape[1], seed=self.config.seed+1
        )
        print("num_row", self.config.num_anchor_data, "num_col", self.Xs_train[0].shape[1])
        print("Xs_train[0].shape", self.Xs_train[0].shape, "Xs_test[0].shape", self.Xs_test[0].shape)
        # 中間表現の生成
        self.make_intermediate_expression()
        #self.make_intermediate_expression(USE_KERNEL=True)

        # 統合表現の生成
        if self.config.G_type == "Imakura":
            self.make_integrate_expression()
        elif self.config.G_type  == "targetvec":
            self.make_integrate_expression_targetvec()
        elif self.config.G_type  == "GEP":
            self.make_integrate_expression_gen_eig(use_eigen_weighting=False)
        elif self.config.G_type  == "GEP_weighted":
            self.make_integrate_expression_gen_eig(use_eigen_weighting=True)
        elif self.config.G_type == "ODC": # この分岐を追加
            self.make_integrate_expression_odc()
        elif self.config.G_type  == "nonlinear":
            self.make_integrate_nonlinear_expression()
        elif self.config.G_type  == "nonlinear_tuning":
            self.make_integrate_nonlinear_expression_tuning()
        elif self.config.G_type == "nonlinear_linear":
            self.make_integrate_nonlinear_linear()
        else:
            print(f"Unknown G_type: {self.config.G_type}")

        self.logger.info(f"{self.config.dim_integrate}:次元")
        self.logger.info(f"{self.config.num_institution_user} 機関人数")
        self.logger.info(f"{self.config.num_institution} 機関数")


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

   # @staticmethod
    #def produce_anchor(num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
    #    """
    #    アンカーデータを生成する関数
    #    """
    #    np.random.seed(seed=seed)
    #    anchor = np.random.randn(num_row, num_col)
    #    return anchor
    
    def produce_anchor(self, num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
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
        if self.config.F_type == "kernel_pca_svd_mixed": #
            mixed = True
        elif self.config.F_type == "kernel_pca_unfixed_mixed":
            unfixed_mixed = True
            # kernel_pca_unfixed_gamma
        for X_train, X_test in zip(tqdm(self.Xs_train), self.Xs_test):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用
            if mixed:
                self.config.f_seed_2 += 1
                if self.config.f_seed_2 % 2 == 0:
                    self.config.F_type = "svd"
                    print("svd")
                else:
                    self.config.F_type = "kernel_pca_self_tuning"
                    print("kernel")
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

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])

        self.logger.info(f"中間表現（訓練データ）の数と次元数: {self.Xs_train_inter[0].shape}")

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
        eigvals, eigvecs = eigh(A_s_tilde, B_s_tilde)                 # SciPy の一般化固有分解
        order   = np.argsort(eigvals)                                 # 昇順
        lambdas = eigvals[order][:p_hat]                              # ★ λ_1 … λ_p̂
        V_sel   = eigvecs[:, order[:p_hat]]
        cum_dims = np.cumsum([0] + [S.shape[1] for S in self.anchors_inter])

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
        # 4.  λ に基づくウェイト計算（オプション）★
        # --------------------------------------------------
        # if use_w:
        #     lam_min, lam_max = lambdas[0], lambdas[-1]
        #     if np.isclose(lam_max, lam_min):
        #         weights = np.ones_like(lambdas)
        #     else:
        #         weights = np.exp(-(lambdas - lam_min) / (lam_max - lam_min))
        #     # 射影行列側に重みを掛けておく（後段の行列積で自動適用）
        #     V_sel = V_sel * weights[np.newaxis, :]
        # else:
        #     weights = np.ones_like(lambdas)   # dummy（あとで保存だけする）

        # --------------------------------------------------
        # 5. 機関ごとの G^(k) 抽出と射影
        # --------------------------------------------------
        # ベクトル vj の分散を計算し、機関ごとに平均を取る
        Xs_train_integrate, Xs_test_integrate = [], []

        for k, (d_k, X_tr_k, X_te_k) in enumerate(
                zip(np.diff(cum_dims), self.Xs_train_inter, self.Xs_test_inter)):

            Gk = V_sel[cum_dims[k]:cum_dims[k + 1], :]               # d_k × p̂
            Xs_train_integrate.append(X_tr_k @ Gk)
            Xs_test_integrate.append( X_te_k @ Gk)

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
