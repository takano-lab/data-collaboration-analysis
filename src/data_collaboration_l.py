from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from config.config import Config
from src.utils import reduce_dimensions

logger = TypeVar("logger")
import csv
from pathlib import Path

from config.timing import timed
from src.utils import reduce_dimensions

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
        #self.ys_train_inter: list[np.ndarray] = []
        #self.ys_test_inter: list[np.ndarray] = []

        # 統合表現
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
            self.make_integrate_nonliner_expression()
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
        for X_train, X_test in zip(tqdm(self.Xs_train), self.Xs_test):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用

            X_train_svd, X_test_svd, anchor_svd = reduce_dimensions(
               X_train=X_train,
               X_test=X_test,
               n_components=self.config.dim_intermediate,
               anchor=self.anchor,
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
        eigvals, eigvecs = eigh(C_tildeS)                 # 昇順で返る
        p_hat = self.config.dim_integrate
        Z = eigvecs[:, :p_hat]                            # r×p̂  —— 目標行列 Z

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


    # ------------------------------------------------------------------
    # 〈非線形統合〉　射影行列 P^(k) で Z を最適化する ２段階アルゴリズム
    # ------------------------------------------------------------------
    def make_integrate_nonliner_expression(self) -> None:
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

        if hasattr(self.config, "nl_lambda"):
            lam = self.config.nl_lambda
        else:
            lam = 1e-2
        #gammas = [11, 15.5, 1000]
        #k = 1
        # --- 1. Gram 行列と射影行列 ---
        for S̃ in self.anchors_inter:             # S̃ : r×d̃_k
            #γ = gammas[k-1]
            #k += 1
            if True:
                # ==============================================================
                # ★ RBF → 薄板スプライン (2 次導関数ペナルティ) 版に書き換え ★
                # ==============================================================

                # -- (1) 薄板スプライン・カーネル ------------------------------------------
                def tps_kernel(X, Y):
                    """Thin‑plate spline φ(||x−y||)  (次数 m=2)."""
                    d = X.shape[1]
                    r = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)

                    if d == 1:                       # 立方スプライン
                        return 0.5 * r**3
                    elif d == 2:
                        eps = 1e-12                  # log(0) 回避
                        return r**2 * np.log(r + eps)
                    else:                            # d > 2
                        r[r == 0] = 1                # 0^0 回避
                        return r**(3 - d)

                # -- (2) 行列を構築 ---------------------------------------------------------
                r, d = S̃.shape                           # アンカー数 r, 元特徴次元 d
                K   = tps_kernel(S̃, S̃)                  # (r×r)
                P   = np.hstack((np.ones((r, 1)), S̃))    # (r×(d+1))  多項式基底 [1, x]

                # -- (3) A⁻¹,  (PᵀA⁻¹P)⁻¹ を安全に求める -----------------------------------
                A_inv = np.linalg.inv(K + lam * np.eye(r))      # λ>0 を想定
                S_mat = P.T @ A_inv @ P                         # (d+1)×(d+1)

                try:
                    M = np.linalg.inv(S_mat)                    # フルランクなら
                except np.linalg.LinAlgError:
                    M = np.linalg.pinv(S_mat)                   # 低ランクなら疑似逆

                # -- (4) g(S̃) = P_λ z  に対応する「射影」行列 P_λ ---------------------------
                P_lambda = (
                    K @ A_inv
                    + (P - K @ A_inv @ P)       # = (I - K A⁻¹) P
                    @ M
                    @ (P.T @ A_inv)
                )
                # ------------- 既存リストに格納（従来の Ks / Ps と同じインターフェース）
                Ks.append(K)
                Ps.append(P_lambda)
                γ = 1.0 / S̃.shape[1]                # γ = 1/d̃_k
                gammas.append(γ)
            else:
                if hasattr(self.config, "nl_gamma"):
                    γ = self.config.nl_gamma
                else:
                    γ = 1.0 / S̃.shape[1]                # γ = 1/d̃_k
                gammas.append(γ)
                K = rbf_kernel(S̃, S̃, gamma=γ)       # r×r
                Ks.append(K)
                Ps.append(K @ inv(K + lam * I_r))     # 射影

        # --- 2. 固有値問題 → Z (r×p̂ , ‖Z‖_F=1) ---
        M = sum((P - I_r).T @ (P - I_r) for P in Ps)

        # ❶ ほんのわずかな非対称を切り落とす
        M = (M + M.T) * 0.5

        # ❷ 実対称用の固有値分解を使う
        eigvals, eigvecs = np.linalg.eigh(M)

        # ❸ 念のため負の丸め誤差を 0 に
        eigvals[eigvals < 0] = 0.0

        Z = eigvecs[:, eigvals.argsort()[:p̂]]
        Z /= norm(Z, 'fro')

        # --- 3. 各機関の係数 B^(k) とデータ射影 ---
        Xs_train_intg, Xs_test_intg = [], []
        for K, S̃, γ, X_tr, X_te in zip(Ks, self.anchors_inter, gammas,
                                        self.Xs_train_inter, self.Xs_test_inter):
            if True:
                S_tilde = S̃
                # ------------------------------------------------------------
                # 立方 / 薄板スプライン用のカーネル
                def tps_kernel(X, Y):
                    d = X.shape[1]
                    r = np.linalg.norm(X[:, None] - Y[None, :], axis=-1)
                    if d == 1:
                        return 0.5 * r**3
                    elif d == 2:
                        eps = 1e-12
                        return r**2 * np.log(r + eps)
                    else:
                        return r**(3 - d)               # d > 2
                # ------------------------------------------------------------

                K   = tps_kernel(S_tilde, S_tilde)        # (r, r)
                P   = np.hstack((np.ones((r, 1)), S_tilde))  # (r, d+1)

                A_inv = np.linalg.inv(K + lam * np.eye(r))   # r×r, λ>0 前提
                S_mat = P.T @ A_inv @ P                      # (d+1)×(d+1)

                # ---- 低ランクなら疑似逆行列 ----------------------------------------------
                try:
                    S_inv = np.linalg.inv(S_mat)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S_mat)

                beta  = S_inv @ (P.T @ A_inv @ Z)            # (d+1)×p̂
                alpha = A_inv @ (Z - P @ beta)               # r×p̂
                # ---------------------------------------------------------------------------
                def features(X):
                    return tps_kernel(X, S_tilde) @ alpha + \
                        np.hstack((np.ones((len(X), 1)), X)) @ beta
                Xs_train_intg.append(features(X_tr))
                Xs_test_intg .append(features(X_te))


            else:
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
            plt.savefig(Path(save_dir) / f"train_{self.config.G_type}_{self.config.nl_lambda}.png")
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

