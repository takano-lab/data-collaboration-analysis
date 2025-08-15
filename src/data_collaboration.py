from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh, inv, norm, pinv
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

# ------ 修正後 ---------------------
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
        
    def assign_anchor_labels(self, k=5):
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

    @staticmethod
    def produce_anchor(num_row: int, num_col: int, seed: int = 0) -> np.ndarray:
        """
        アンカーデータを生成する関数
        """
        np.random.seed(seed=seed)
        anchor = np.random.randn(num_row, num_col)
        return anchor

    def make_intermediate_expression(self) -> None:
        print("********************中間表現の生成********************")
        """
        中間表現を生成する関数
        """
        print(self.config)
        print("self.config.dim_intermediate", self.config.dim_intermediate)
        print()
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
                else:
                    self.config.F_type = "kernel_pca_self_tuning"
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

        for X_train_inter, X_test_inter, anchor_inter in zip(
            tqdm(self.Xs_train_inter), self.Xs_test_inter, self.anchors_inter
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

            # そのままで実験 ##########################################
            # X_train_integrate = X_train_inter.T
            # X_test_integrate = X_test_inter.T

            # 統合表現をリストに格納
            Xs_train_integrate.append(X_train_integrate.T)
            Xs_test_integrate.append(X_test_integrate.T)

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

        # --------------------------------------------------
        # 2. 固有値問題  C_s̃ z = λ z  を解く（昇順）
        # --------------------------------------------------
        #eigvals, eigvecs = eigh(C_tildeS)                 # 昇順で返る
        p_hat = self.config.dim_integrate
        #Z = eigvecs[:, :p_hat]                            # r×p̂  —— 目標行列 Z
        
        objective_direction_ratio = getattr(self.config, "objective_direction_ratio", 0)
        if objective_direction_ratio > 0:
            idx, Z, eigvals, eigvecs = self.select_eigvecs_multilogit_hybrid(C_tildeS, self.anchor_y, p_hat=p_hat, objective_direction_ratio=objective_direction_ratio)
        else:
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

        # 必要なら self.Gs = Gs などで保存しておくと解析に便利

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
        for anchor_k, X_tr_k, X_te_k in zip(
            self.anchors_inter, self.Xs_train_inter, self.Xs_test_inter
        ):
            # 3. M_k = A_k^T @ A_1 を計算
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

        # 7. スタックして最終データを保持
        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

    def select_eigvecs_multilogit_hybrid(
        self,
        M: np.ndarray,
        y: np.ndarray,
        p_hat: int,
        objective_direction_ratio: float = 0.0,
        C: float = 1.0,
        random_state: int = 0,
    ):
        """
        ハイブリッド選択：
        1) 前半 floor(p_hat/2) 本は『固有値の小さい順』で先取り
        2) 残りは『ロジスティック回帰のスコア（+ 固有値ペナルティ）』で前向き選択

        戻り値は (selected_idx, Z, eigvals, eigvecs)
        """
        # --- 固有分解 ---
        eigvals, eigvecs = np.linalg.eigh(M)  # eigvecs[:, i] が固有ベクトル u_i
        n = M.shape[0]

        # --- One-hot & 初期確率 ---
        try:
            enc = OneHotEncoder(sparse_output=False, dtype=float)
        except TypeError:  # scikit-learn 古い版対策
            enc = OneHotEncoder(sparse=False, dtype=float)
        Y = enc.fit_transform(y.reshape(-1, 1))  # (n, K)
        K = Y.shape[1]
        class_priors = Y.mean(axis=0)            # (K,)
        P = np.tile(class_priors, (n, K))        # (n, K)

        # --- 1) 固有値で前半を先取り ---
        m1 = int(p_hat*(1-objective_direction_ratio))
        
        eig_order = np.argsort(eigvals)          # 小さい→大きい
        selected_idx = list(eig_order[:m1])
        remaining = [i for i in range(n) if i not in selected_idx]

        # 先取り分で一度フィット（m1>0 のとき）
        if m1 > 0:
            Z_sel = eigvecs[:, selected_idx]
            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                C=C, max_iter=1000, random_state=random_state
            )
            clf.fit(Z_sel, y)
            P = clf.predict_proba(Z_sel)  # (n, K)
        else:
            Z_sel = None  # まだ何もない

        # --- 2) 残りをスコアで前向き選択 ---
        m2 = p_hat - m1
        for _ in range(m2):
            R = Y - P
            W_diag_list = [P[:, c] * (1.0 - P[:, c]) + 1e-12 for c in range(K)]

            best_j, best_score = None, -np.inf
            for j in remaining:
                u = eigvecs[:, j]
                # ロジスティックのスコア検定（OvR の和） - gamma * λ_j
                num = 0.0
                for c in range(K):
                    r_c = R[:, c]
                    W_c = W_diag_list[c]
                    uTr = float(u @ r_c)
                    uWu = float(u @ (W_c * u))
                    num += (uTr ** 2) / uWu
                score = num #- gamma * float(eigvals[j])

                # --- SVM基準にしたい場合はここを差し替え ---
                # score = cv_linear_svm_score(np.column_stack([Z_sel, u]) if Z_sel is not None else u.reshape(-1,1), y)
                #        - gamma * normalized_eigval[j]
                # --------------------------------------------

                if score > best_score:
                    best_score = score
                    best_j = j

            # 採用＆再フィット
            selected_idx.append(best_j)
            remaining.remove(best_j)
            Z_sel = eigvecs[:, selected_idx]

            clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                C=C, max_iter=1000, random_state=random_state
            )
            clf.fit(Z_sel, y)
            P = clf.predict_proba(Z_sel)

        Z = eigvecs[:, selected_idx]
        return selected_idx, Z, eigvals, eigvecs
    
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

        Ks, Ps, gammas = [], [], []
        I_r = np.eye(r)
        
        if self.config.gamma_type == "auto":
            for S̃ in self.anchors_inter:             # S̃ : r×d̃_k
                γ = 1.0 / S̃.shape[1]                # γ = 1/d̃_k
                gammas.append(γ)

        elif self.config.gamma_type == "X_tuning":
            for X_train_inter in self.Xs_train_inter:
                # gamma を計算
                gamma = self_tuning_gamma(X_train_inter, standardize=False, k=3, summary='median')
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
            mu_max = max(eigvalsh(K).max(), 1e-12)            # スペクトル半径
            K = K / mu_max                                # ||K||_2 = 1
            
            Ks.append(K)
            Ps.append(K @ inv(K + lam * I_r))     # 射影

        # --- 2. 固有値問題 → Z (r×p̂ , ‖Z‖_F=1) ---
        M = sum((P - I_r).T @ (P - I_r) for P in Ps)

        # ❶ ほんのわずかな非対称を切り落とす
        M = (M + M.T) * 0.5
        
        objective_direction_ratio = getattr(self.config, "objective_direction_ratio", 0)
        if objective_direction_ratio > 0:
            idx, Z, eigvals, eigvecs = self.select_eigvecs_multilogit_hybrid(M, self.anchor_y, p_hat=p̂, objective_direction_ratio=objective_direction_ratio)
        else:
            # ❷ 実対称用の固有値分解を使う
            eigvals, eigvecs = np.linalg.eigh(M)
            # ❸ 念のため負の丸め誤差を 0 に
            eigvals[eigvals < 0] = 0.0
            Z = eigvecs[:, eigvals.argsort()[:p̂]]
            
        # 列ごとに ||z_j||_2 = 1 へ
        for j in range(Z.shape[1]):
            nz = np.linalg.norm(Z[:, j])
            if nz > 0:
                Z[:, j] /= nz
        # --- 3. 各機関の係数 B^(k) とデータ射影 ---
        Xs_train_intg, Xs_test_intg = [], []
        for K, S̃, γ, X_tr, X_te in zip(Ks, self.anchors_inter, gammas,
                                        self.Xs_train_inter, self.Xs_test_inter):
            
            Bk  = inv(K + lam * I_r) @ Z          # r×p̂

            K_tr = rbf_kernel(X_tr, S̃, gamma=γ)  # n_k×r
            K_te = rbf_kernel(X_te, S̃, gamma=γ)  # t_k×r

            Xs_train_intg.append(K_tr @ Bk)       # n_k×p̂
            Xs_test_intg.append(K_te @ Bk)        # t_k×p̂

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


    def visualize_representations(self, save_dir: Optional[str] = None) -> None:
        """
        元データ、中間表現、統合表現（機関ごとと全体）を2次元散布図で可視化する関数。
        訓練データとテストデータをそれぞれ別の図で出力する。
        """
        save_dir = save_dir or self.config.output_path
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
            plt.savefig(Path(save_dir) / f"_{self.config.dataset}_{self.config.nl_lambda}__{self.config.True_F_type}_{self.config.G_type}_{self.config.gamma_type}.png")
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

