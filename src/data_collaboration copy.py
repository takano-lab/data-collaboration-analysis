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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
            self.make_integrate_expression_imakura()
        elif self.config.G_type  == "targetvec":
            self.make_integrate_expression_targetvec()
        elif self.config.G_type  == "GEP":
            self.make_integrate_expression_gen_eig(use_eigen_weighting=False)
        elif self.config.G_type  == "GEP_weighted":
            self.make_integrate_expression_gen_eig(use_eigen_weighting=True)
        elif self.config.G_type == "ODC": # この分岐を追加
            self.make_integrate_expression_odc()
        elif self.config.G_type  == "nonlinear":
            self.assign_anchor_labels(k=5)
            self.build_laplacians_from_anchor_labels()
            self.make_integrate_nonlinear_expression()
        elif self.config.G_type == "nonlinear_linear":
            self.make_integrate_nonlinear_linear()
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

        elif self.config.anchor_method == "smote":
            """
            SMOTE 風に self.Xs_train と self.Xs_test の元データから公開データ Xpub を合成して返す。
            - 既定パラメータ（k_neighbors=5, lambda ~ U[0,1]）
            - クラス比は元データに概ね比例させ、最終的に num_row 件を返す
            - 指示通り、Xpub にオリジナルの元データ行は含めない（合成データのみ）
            注意: self.Xs_train/self.ys_train などは run() の先頭の分割後に存在する前提
            """
            rng = np.random.default_rng(seed)

            if not self.Xs_train or not self.ys_train:
                raise RuntimeError("SMOTE anchor 生成には先に train_test_split が必要です。")

            # 元データ（train+test）を結合
            X_train_all = np.vstack(self.Xs_train) if len(self.Xs_train) > 1 else self.Xs_train[0]
            y_train_all = np.hstack(self.ys_train) if len(self.ys_train) > 1 else self.ys_train[0]
            X_test_all  = np.vstack(self.Xs_test)  if len(self.Xs_test)  > 1 else self.Xs_test[0]
            y_test_all  = np.hstack(self.ys_test)  if len(self.ys_test)  > 1 else self.ys_test[0]

            X0 = np.vstack([X_train_all, X_test_all])
            y0 = np.hstack([y_train_all, y_test_all])

            # 列数を num_col に合わせる（過不足対策）
            if X0.shape[1] < num_col:
                num_col = X0.shape[1]
            X0 = X0[:, :num_col]

            classes, counts = np.unique(y0, return_counts=True)
            N_total = int(len(y0))
            if N_total == 0:
                # フォールバック: ガウスに戻す
                return np.random.default_rng(seed).normal(size=(num_row, num_col))

            # クラス別に目標生成数を割当（合計 num_row になるよう調整）
            # まず比例配分で丸め、最後のクラスで残差を吸収
            target_counts = []
            allocated = 0
            for i, c in enumerate(classes):
                if i < len(classes) - 1:
                    n_c = int(round(num_row * (counts[i] / N_total)))
                    target_counts.append(n_c)
                    allocated += n_c
                else:
                    target_counts.append(num_row - allocated)

            # クラスごとに近傍を準備
            synthetic_list = []
            for c, n_gen in zip(classes, target_counts):
                if n_gen <= 0:
                    continue
                mask = (y0 == c)
                Xc = X0[mask]
                Nc = Xc.shape[0]
                if Nc == 0:
                    continue

                # 近傍数（自分を含めて n_neighbors 個出すので +1）
                k = min(6, Nc)  # 実近傍5を目安（自分を含め6を取り、あとで除外）

                # Nc==1 の例外処理: 近傍が取れないので微小ノイズで複製
                if Nc == 1:
                    base = np.tile(Xc[0], (n_gen, 1))
                    # スケールに応じた微小ノイズ
                    std = np.std(X0, axis=0, ddof=0)
                    noise = rng.normal(scale=np.where(std>0, 1e-3*std, 1e-3), size=base.shape)
                    synthetic_list.append(base + noise)
                    continue

                nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
                nbrs.fit(Xc)
                # 近傍インデックス（自分を含む） shape: (Nc, k)
                indices = nbrs.kneighbors(Xc, return_distance=False)

                # 合成サンプルを作成
                gen_idx = rng.integers(low=0, high=Nc, size=n_gen)
                # 各元サンプルに対して、近傍から1つ選ぶ（自身 indices[i,0] を避ける）
                nn_choices = []
                for i0 in gen_idx:
                    neigh = indices[i0]
                    if len(neigh) <= 1:
                        # 念のため
                        j = i0
                    else:
                        # 先頭を除いた中からランダム
                        j = rng.choice(neigh[1:])
                    nn_choices.append(j if isinstance(j, (int, np.integer)) else int(j))
                nn_choices = np.array(nn_choices, dtype=int)

                lam = rng.random(size=n_gen)  # U[0,1]
                Xi = Xc[gen_idx]
                Xj = Xc[nn_choices]
                Xsyn = Xi + lam[:, None] * (Xj - Xi)
                synthetic_list.append(Xsyn)

            Xpub_syn = np.vstack(synthetic_list) if synthetic_list else np.empty((0, num_col))

            # 合成数の過不足調整
            if Xpub_syn.shape[0] > num_row:
                idx = rng.choice(Xpub_syn.shape[0], size=num_row, replace=False)
                Xpub_syn = Xpub_syn[idx]
            elif Xpub_syn.shape[0] < num_row:
                # 不足分はまず追加SMOTEで埋める（Gaussianは最終手段）
                deficit = num_row - Xpub_syn.shape[0]

                # 追加SMOTEの関数（Nc>=2のクラスから作る）
                def make_more_smote(need_more: int) -> np.ndarray:
                    chunks = []
                    if need_more <= 0:
                        return np.empty((0, num_col))
                    # 比例配分で各クラスに割当
                    share = np.maximum(1, np.round(need_more * (counts / counts.sum())).astype(int))
                    # 残差調整
                    over = int(share.sum() - need_more)
                    if over > 0:
                        for i in range(len(share)-1, -1, -1):
                            take = min(over, max(0, share[i]-1))
                            share[i] -= take
                            over -= take
                            if over == 0:
                                break
                    elif over < 0:
                        for i in range(len(share)):
                            add = min(-over, 1)
                            share[i] += add
                            over += add
                            if over == 0:
                                break

                    for c, s in zip(classes, share):
                        if s <= 0:
                            continue
                        Xc = X0[y0 == c]
                        Nc = Xc.shape[0]
                        if Nc >= 2:
                            k = min(6, Nc)
                            nbrs2 = NearestNeighbors(n_neighbors=k, algorithm="auto")
                            nbrs2.fit(Xc)
                            inds2 = nbrs2.kneighbors(Xc, return_distance=False)
                            gi = rng.integers(0, Nc, size=int(s))
                            gj = []
                            for i0 in gi:
                                neigh = inds2[i0]
                                if len(neigh) <= 1:
                                    j0 = i0
                                else:
                                    j0 = rng.choice(neigh[1:])
                                gj.append(int(j0))
                            gj = np.array(gj, dtype=int)
                            lam2 = rng.random(size=int(s))
                            Xi2 = Xc[gi]
                            Xj2 = Xc[gj]
                            chunks.append(Xi2 + lam2[:, None] * (Xj2 - Xi2))
                    return np.vstack(chunks) if chunks else np.empty((0, num_col))

                extra = make_more_smote(deficit)
                Xpub_syn = np.vstack([Xpub_syn, extra])
                deficit = num_row - Xpub_syn.shape[0]

                if deficit > 0:
                    # どうしても不足する分のみ、微小Gaussianで補完（最大5%まで）
                    gauss_cap = max(1, int(np.floor(0.05 * num_row)))
                    gauss_fill = min(deficit, gauss_cap)
                    if gauss_fill > 0:
                        std = np.std(X0, axis=0, ddof=0)
                        noise = rng.normal(scale=np.where(std>0, 1e-3*std, 1e-3), size=(gauss_fill, num_col))
                        base = X0[rng.choice(N_total, size=gauss_fill, replace=True)][:, :num_col]
                        Xpub_syn = np.vstack([Xpub_syn, base + noise])
                        deficit = num_row - Xpub_syn.shape[0]

                if deficit > 0 and Xpub_syn.shape[0] > 0:
                    # それでも不足する分は、既存の合成サンプルを再サンプル（複製）
                    # （オリジナルを含めないという条件を維持）
                    idx_rep = rng.choice(Xpub_syn.shape[0], size=deficit, replace=True)
                    Xpub_syn = np.vstack([Xpub_syn, Xpub_syn[idx_rep]])

            # 追跡用に保存（合成のみ）。オリジナルを入れないのが条件
            try:
                self.Xpub = Xpub_syn.copy()
                self.Xpub_y = None  # 必要なら y0 に比例したラベルを別途付ける
            except Exception:
                pass

            return Xpub_syn

    def make_intermediate_expression(self) -> None:
        print("********************中間表現の生成********************")
        """
        中間表現を生成する関数
        """
        print(self.config)
        print("self.config.dim_intermediate", self.config.dim_intermediate)
        print()
        # シードを初期化（各機関で進める）
        self.config.f_seed = 0
        self.config.f_seed_2 = 0

        # True_F_type の解釈:
        # - 文字列: その方式を使用
        # - リスト/タプル: 機関ごとにローテーションして使用
        # - 未設定: 現在の F_type を固定使用
        tf = getattr(self.config, "True_F_type", None)
        if isinstance(tf, (list, tuple)) and len(tf) > 0:
            ftype_sequence = list(tf)
        elif isinstance(tf, str) and len(tf) > 0:
            # 従来の mixed キーワードに相当する簡易プリセットにも対応
            if tf == "kernel_pca_svd_mixed":
                ftype_sequence = ["kernel_pca_self_tuning", "svd"]
            elif tf == "ae_dm_mixed":
                ftype_sequence = ["ae", "dm"]
            elif tf == "ae_svd_mixed":
                ftype_sequence = ["ae", "svd"]
            elif tf == "ae_dm_svd_mixed":
                ftype_sequence = ["ae", "dm", "svd"]
            else:
                ftype_sequence = [tf]
        else:
            ftype_sequence = [self.config.F_type]

        for idx, (X_train, X_test) in enumerate(zip(tqdm(self.Xs_train), self.Xs_test)):
            # 各機関の F_type を選択（ローテーション）
            self.config.F_type = ftype_sequence[idx % len(ftype_sequence)]

            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、reduce_dimensions を適用
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
        
    def _apply_models(self, Z: np.ndarray, models: list) -> None:
        """共通の適用ロジック: 各モデルを使って X_train/test と anchor を統合空間に写像する。"""
        Xs_train_integrate, Xs_test_integrate = [], []
        self.anchors_integ, self.anchors_test_integ = [], []

        for model, X_tr, X_te, S_tr, S_te in zip(models, self.Xs_train_inter, self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter):
            if model.kind == 'linear' and model.G is not None:
                G = model.G  # d x p
                Xs_train_integrate.append(X_tr @ G)
                Xs_test_integrate.append(X_te @ G)
                self.anchors_integ.append(S_tr @ G)
                self.anchors_test_integ.append(S_te @ G)
            elif model.kind == 'rbf' and model.B is not None:
                # K(x,S) @ B
                K_tr = rbf_kernel(X_tr, model.S_train, gamma=model.gamma)
                K_te = rbf_kernel(X_te, model.S_train, gamma=model.gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_tr = K_tr / model.mu_max
                    K_te = K_te / model.mu_max
                Xs_train_integrate.append(K_tr @ model.B)
                Xs_test_integrate.append(K_te @ model.B)
                # anchors
                K_S = rbf_kernel(S_tr, model.S_train, gamma=model.gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_S = K_S / model.mu_max
                self.anchors_integ.append(K_S @ model.B)
                K_S_te = rbf_kernel(S_te, model.S_train, gamma=model.gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_S_te = K_S_te / model.mu_max
                self.anchors_test_integ.append(K_S_te @ model.B)
            elif model.kind == 'rbf+linear' and model.alpha is not None and model.beta is not None:
                # f(x) = K(x,S) alpha + x beta
                S_base = model.S_train if model.S_train is not None else S_tr
                gamma = model.gamma if model.gamma is not None else 1.0
                K_tr = rbf_kernel(X_tr, S_base, gamma=gamma)
                K_te = rbf_kernel(X_te, S_base, gamma=gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_tr = K_tr / model.mu_max
                    K_te = K_te / model.mu_max
                Xs_train_integrate.append(K_tr @ model.alpha + X_tr @ model.beta)
                Xs_test_integrate.append(K_te @ model.alpha + X_te @ model.beta)
                # anchors
                K_S = rbf_kernel(S_tr, S_base, gamma=gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_S = K_S / model.mu_max
                self.anchors_integ.append(K_S @ model.alpha + S_tr @ model.beta)
                K_S_te = rbf_kernel(S_te, S_base, gamma=gamma)
                if model.mu_max and model.mu_max != 1.0:
                    K_S_te = K_S_te / model.mu_max
                self.anchors_test_integ.append(K_S_te @ model.alpha + S_te @ model.beta)

        self.X_train_integ = np.vstack(Xs_train_integrate)
        self.X_test_integ = np.vstack(Xs_test_integrate)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)
        self.Z = Z
        self.logger.info(f"統合表現（訓練データ）の数と次元数: {self.X_train_integ.shape}")

    def make_integrate_expression_imakura(self) -> None:
        """旧 Imakura 実装を Integrator に置き換えたもの。"""
        print("********************統合表現の生成 (Imakura) ********************")
        from src.integration import Integrator
        Z, models, meta = Integrator.imakura(self.anchors_inter, self.config.dim_integrate)
        # g_abs_sum（参考値）
        self.config.g_abs_sum = float(np.sum([np.sum(np.abs(m.G)) for m in models if m.G is not None]))
        self._apply_models(Z, models)

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

        Ks, Ps, gammas, max_mus  = [], [], [], []
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

        print(gammas)

        if hasattr(self.config, "nl_lambda"):
            lam = self.config.nl_lambda
        else:
            lam = 1e-2

        # --- 1. Gram 行列と射影行列 ---
        for i, S̃ in enumerate(self.anchors_inter):             # S̃ : r×d̃_k
            K = rbf_kernel(S̃, S̃, gamma=gammas[i])       # r×r
            # (a) カーネル行列（先に作って正規化）
            if self.config.K_normalization:
                mu_max = max(eigvalsh(K).max(), 1e-12)            # スペクトル半径
                max_mus.append(mu_max)
                K = K / mu_max                                # ||K||_2 = 1
            
            Ks.append(K)
            Ps.append(K @ inv(K + lam * I_r))     # 射影
        
        M = sum((P - I_r).T @ (P - I_r) for P in Ps)
        ## M 正規化 ラプラシアンなしならしなくてよい
        trace_M = np.trace(M)
        if trace_M > 1e-9:
            M /= trace_M
        
        # --- 2. 固有値問題 → Z (r×p̂ , ‖Z‖_F=1) --- 近接ラプラシアンの重みも加える
        Q = M + lw_alpha * self.L_within - lb_beta * self.L_between

        # ❶ ほんのわずかな非対称を切り落とす
        Q = (Q + Q.T) * 0.5
        
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
        
        # --- 3. 各機関の係数 B^(k) とデータ射影 ---
        Xs_train_intg, Xs_test_intg = [], []
        # zipに self.anchors_test_inter を追加
        for i, (K, S̃_train, S̃_test, γ, X_tr, X_te) in enumerate(zip(
            Ks, self.anchors_inter, self.anchors_test_inter, gammas,
            self.Xs_train_inter, self.Xs_test_inter
        )):
            # 学習データから係数 Bk を計算
            #mu_max = max(eigvalsh(K).max(), 1e-12)            # スペクトル半径
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
                K_tr = K_tr / max_mus[i]
                    
                #s = np.linalg.svd(K_te, compute_uv=False)
                #mu_max = s.max()
                K_te = K_te / max_mus[i]
                # ||K||_2 = 1
                #mu_max = max(eigvalsh(K_anchor_test).max(), 1e-12)            # スペクトル半径
                K_anchor_test = K_anchor_test / max_mus[i]                             # ||K||_2 = 1
            
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
        from itertools import combinations

        import numpy as np

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