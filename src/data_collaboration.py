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
from config.timing import timed


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
               F_type=self.config.F_type
            )
            
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
            
            # そのままで実験 ##########################################
            # X_train_integrate = X_train_inter.T
            # X_test_integrate = X_test_inter.T

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
        #use_w   = getattr(self.config, "use_eigen_weighting", False)   # ★
        use_w = use_eigen_weighting
        

        # --------------------------------------------------
        # 1. W̃_s  と  B̃_s  を構築
        # --------------------------------------------------
        W_s_tilde = np.hstack(self.anchors_inter)                     # r × Σd_k
        blocks    = [S.T @ S for S in self.anchors_inter]             # 各 d_k × d_k
        B_s_tilde = reduce(lambda a, b: block_diag(a, b), blocks)

        # --------------------------------------------------
        # 2. Ã_s = 2m B̃_s - 2 WᵀW
        # --------------------------------------------------
        A_s_tilde = 2 * m * B_s_tilde - 2 * (W_s_tilde.T @ W_s_tilde)

        # --------------------------------------------------
        # 3. 一般化固有値問題  A v = λ B v
        # --------------------------------------------------
        eigvals, eigvecs = eigh(A_s_tilde, B_s_tilde)                 # SciPy の一般化固有分解
        order   = np.argsort(eigvals)                                 # 昇順
        lambdas = eigvals[order][:p_hat]                              # ★ λ_1 … λ_p̂
        V_sel   = eigvecs[:, order[:p_hat]]                           # Σd_k × p̂

        # --------------------------------------------------
        # 4.  λ に基づくウェイト計算（オプション）★
        # --------------------------------------------------
        if use_w:
            lam_min, lam_max = lambdas[0], lambdas[-1]
            if np.isclose(lam_max, lam_min):
                weights = np.ones_like(lambdas)
                print(11111111111111111111111111111111)
            else:
                weights = np.exp(-(lambdas - lam_min) / (lam_max - lam_min))
                print(2222222222222222222222222222222)
            # 射影行列側に重みを掛けておく（後段の行列積で自動適用）
            V_sel = V_sel * weights[np.newaxis, :]
        else:
            weights = np.ones_like(lambdas)   # dummy（あとで保存だけする）

        # --------------------------------------------------
        # 5. 機関ごとの G^(k) 抽出と射影
        # --------------------------------------------------
        cum_dims = np.cumsum([0] + [S.shape[1] for S in self.anchors_inter])
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
        self.lambda_selected  = lambdas
        self.weights_selected = weights
