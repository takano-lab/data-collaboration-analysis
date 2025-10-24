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
from src.integration import (
    build_gep_projectors,
    build_imakura_projectors,
    build_nonlinear_projectors,
    build_odc_projectors,
    build_targetvec_projectors,
)
from src.utils import reduce_dimensions, self_tuning_gamma

logger = TypeVar("logger")
import csv
from pathlib import Path

from config.timing import timed


class DataCollaborationAnalysis:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Config,
        logger: logger,
        Xs_train: list[np.ndarray] | None = None,
        Xs_test: list[np.ndarray] | None = None,
        ys_train: list[np.ndarray] | None = None,
        ys_test: list[np.ndarray] | None = None,
    ) -> None:
        self.config: Config = config
        self.logger = logger

        # 本当はできるだけattributeを持たせない方が良い
        # 元データ
        self.train_df: pd.DataFrame = train_df
        self.test_df: pd.DataFrame = test_df
        self.anchor: np.ndarray = np.array([])
        self.anchor_y: np.ndarray = np.array([])
        self.anchor_test: np.ndarray = np.array([])

        # 機関ごとの分割データ (外部で構築済みのものを優先使用)
        self.Xs_train: list[np.ndarray] = Xs_train or []
        self.Xs_test: list[np.ndarray] = Xs_test or []
        self.ys_train: list[np.ndarray] = ys_train or []
        self.ys_test: list[np.ndarray] = ys_test or []

        # 中間表現
        self.anchors_inter: list[np.ndarray] = []
        self.anchors_test_inter: list[np.ndarray] = []
        self.Xs_train_inter: list[np.ndarray] = []
        self.Xs_test_inter: list[np.ndarray] = []

        # 統合表現
        self.anchors_integ: list[np.ndarray] = []
        self.anchors_test_integ: list[np.ndarray] = []
        self.X_train_integ: np.ndarray = np.array([])
        self.X_test_integ: np.ndarray = np.array([])
        self.y_train_integ: np.ndarray = np.array([])
        self.y_test_integ: np.ndarray = np.array([])
        # Z_integ: r x m_integ で統一したターゲット（あれば設定）
        self.Z_integ: Optional[np.ndarray] = None

    # ------------------------------
    # 共通ヘルパ: インテグレータ（射影関数）
    # ------------------------------
    # 既存のクラス内ヘルパは integration.py に移設

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
        # データの分割（既に渡されていない場合のみ内部で旧分割を実行: 後方互換）
        if not self.Xs_train or not self.Xs_test:
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
            self.make_integrate_expression_gen_eig()
        elif self.config.G_type == "ODC": # この分岐を追加
            self.make_integrate_expression_odc()
        elif self.config.G_type  == "nonlinear":
            #self.assign_anchor_labels(k=5)
            #self.build_laplacians_from_anchor_labels()
            self.make_integrate_nonlinear_expression()
        else:
            print(f"Unknown G_type: {self.config.G_type}")
        
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
            X_train_all = np.vstack([self.Xs_train[i][:3] for i in range (self.config.num_institution)]) if len(self.Xs_train) > 1 else self.Xs_train[0]
            y_train_all = np.hstack([self.ys_train[i][:3] for i in range (self.config.num_institution)]) if len(self.ys_train) > 1 else self.ys_train[0]
            X_test_all  = np.vstack(self.Xs_test)  if len(self.Xs_test)  > 1 else self.Xs_test[0]
            y_test_all  = np.hstack(self.ys_test)  if len(self.ys_test)  > 1 else self.ys_test[0]
            #print(X_train_all.mean())
            print(len(y_train_all))
            #print(len(y_test_all))
            #print(len(self.ys_test))
            #print(len(self.ys_train))
            X0 = np.vstack([X_test_all])
            y0 = np.hstack([y_test_all])

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

            # --- 次元削減 ---
            current_seed = self.config.f_seed  # シフト判定用に保持
            X_train_svd, X_test_svd, anchor_svd, anchor_test_svd = reduce_dimensions(
                X_train=X_train,
                X_test=X_test,
                n_components=self.config.dim_intermediate,
                anchor=self.anchor,
                anchor_test=self.anchor_test,
                F_type=self.config.F_type,
                seed=current_seed,
                config=self.config,
            )
            self.config.f_seed += 1

            # # --- 偏移（第一/第二成分方向シフト） ---
            # # 量: config.inter_shift があればそれを使用 (None / 0 / 未設定 は 0 とみなす)
            # raw_shift = getattr(self.config, "inter_shift", 5.0)
            # # 偶数 → 第1成分 (index 0), 奇数 → 第2成分 (index 1; 次元不足なら 0)
            # axis_idx = 0 if (current_seed % 2 == 0) else 1
            # if X_train_svd.shape[1] <= axis_idx:
            #     axis_idx = 0  # 次元不足フォールバック
            # # シフトベクトル作成

            # shift_vec = np.zeros(X_train_svd.shape[1], dtype=float)
            # shift_vec[axis_idx] = 10.0
            # print(shift_vec, 444444444444444)
            # # 全データ (train/test/anchor/anchor_test) を同じだけ平行移動
            # X_train_svd = X_train_svd + shift_vec
            # X_test_svd = X_test_svd + shift_vec
            # anchor_svd = anchor_svd + shift_vec
            # anchor_test_svd = anchor_test_svd + shift_vec

            # --- 格納 ---
            inter_norm = getattr(self.config, "inter_normalization", False)
            if not inter_norm:
                self.Xs_train_inter.append(X_train_svd)
                self.Xs_test_inter.append(X_test_svd)
                self.anchors_inter.append(anchor_svd)
                self.anchors_test_inter.append(anchor_test_svd)
            
            else:
                #標準化 # qsar だと欠損になる
                
                # SVDを適用したデータをリストに格納
                scaler = StandardScaler()
                
                # アンカーデータの標準化
                anchor_svd = scaler.fit_transform(anchor_svd)
                self.anchors_inter.append(anchor_svd)

                # 訓練データの標準化
                X_train_svd = scaler.transform(X_train_svd)
                self.Xs_train_inter.append(X_train_svd)

                # テストデータの標準化
                X_test_svd = scaler.transform(X_test_svd)
                self.Xs_test_inter.append(X_test_svd)

                # テスト用アンカーデータの標準化
                anchor_test_svd = scaler.transform(anchor_test_svd)
                self.anchors_test_inter.append(anchor_test_svd)

        print("中間表現の次元数: ", self.Xs_train_inter[0].shape[1])

        self.logger.info(f"中間表現（訓練データ）の数と次元数: {self.Xs_train_inter[0].shape}")
    
    # 統合関数の共通適用ヘルパ
    def _apply_integrator_per_institution(self, integrator_builder):
        """integrator_builder を用いて各機関の (X_train, X_test, anchor, anchor_test) に適用する統一ループ。
        integrator_builder は callable で、引数は機関別の必要行列（例: Z_integ と anchor_inter_k）を受け取り、
        projector（X -> ...）と係数行列（ログ用）を返す想定。
        戻り値: (Xs_train_integrate, Xs_test_integrate)
        副作用: self.anchors_integ, self.anchors_test_integ を追加更新
        """
        Xs_train_integrate, Xs_test_integrate = [], []
        for X_train_inter, X_test_inter, anchor_inter, anchor_test_inter in zip(
            tqdm(self.Xs_train_inter), self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter
        ):
            projector, _ = integrator_builder(anchor_inter)
            X_train_integrate = projector(X_train_inter)
            X_test_integrate = projector(X_test_inter)
            anchor_integ = projector(anchor_inter)
            anchor_test_integ = projector(anchor_test_inter)

            Xs_train_integrate.append(X_train_integrate)
            Xs_test_integrate.append(X_test_integrate)
            self.anchors_integ.append(anchor_integ)
            self.anchors_test_integ.append(anchor_test_integ)

        return Xs_train_integrate, Xs_test_integrate

    # 新しい共通ヘルパ: 生成済みプロジェクタ群を適用して属性をセット
    def _apply_projectors_and_set(self, projs: list):
        Xs_train_integ: list[np.ndarray] = []
        Xs_test_integ: list[np.ndarray] = []
        for proj, X_tr, X_te, anc_tr, anc_te in zip(
            projs, self.Xs_train_inter, self.Xs_test_inter, self.anchors_inter, self.anchors_test_inter
        ):
            Xs_train_integ.append(proj(X_tr))
            Xs_test_integ.append(proj(X_te))
            self.anchors_integ.append(proj(anc_tr))
            self.anchors_test_integ.append(proj(anc_te))

        # スタック & y も連結
        self.X_train_integ = np.vstack(Xs_train_integ)
        self.X_test_integ = np.vstack(Xs_test_integ)
        self.y_train_integ = np.hstack(self.ys_train)
        self.y_test_integ = np.hstack(self.ys_test)
        return Xs_train_integ, Xs_test_integ
        
    def make_integrate_expression(self) -> None:
        print("********************統合表現の生成********************")
        """
        統合表現を生成する関数
        """
        # integration.py で projector 群を構築
        projs, Z_integ, g_abs_sum = build_imakura_projectors(self.anchors_inter, self.config.dim_integrate)
        # Z_integ を設定（self.Z_integ に統一）
        self.Z_integ = Z_integ

        # projector を適用して属性にセット
        self._apply_projectors_and_set(projs)

        # メトリクス（従来と同様に出力）
        self.config.g_abs_sum = g_abs_sum
        print(f"擬似逆行列の絶対値の総和: {self.config.g_abs_sum}")
        print("統合表現の次元数: ", self.X_train_integ.shape[1])

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
        c = self.config.num_institution  # 機関数（c に統一）
        r = self.config.num_anchor_data
        I_r = np.eye(r)
        
        # --------------------------------------------------
        # 2. 固有値問題  C_s_tilde z = λ z  を解く（Z_integ を得る）
        # --------------------------------------------------
        m_inter = self.config.dim_integrate

        # --------------------------------------------------
        # 3. 各機関ごとに  g^(k) = (anchor_inter_k)^† Z_integ   を計算
        #    → 係数行列 G^(k)（d_I × m_integ）
        # --------------------------------------------------
        # integration.py のビルダーで projector を構築
        projs, Z_integ = build_targetvec_projectors(self.anchors_inter, m_inter)
        # projector を適用して属性にセット
        self._apply_projectors_and_set(projs)

        # 互換のため保持
        self.Z_integ = Z_integ

    # ============================================================
    # 〈統合関数の最適化〉§3 一般化固有値問題 (8) ベース
    #   A_s_tilde v = λ B_s_tilde v ,  vᵀ B_s_tilde v = 1
    # ============================================================
    def make_integrate_expression_gen_eig(self) -> None:
        """
        川上・高野 (2024) §3   一般化固有値問題による統合関数
        + オプションで λ に基づくウェイト付け   (exp(-(λ_j-λ1)/(λ_max-λ1)))
        """
        print("********************統合表現の生成 (一般化固有値型) ********************")

        # 各種設定
        m_inter = self.config.dim_integrate
        lambda_gen = getattr(self.config, 'lambda_gen_eigen', 0)
        use_eigen_weighting = bool(getattr(self.config, "use_eigen_weighting", False))
        print("lambda_gen", lambda_gen)
        orth_ver = bool(getattr(self.config, "orth_ver", None) or False)

        # projector 構築とメトリクス取得
        projs, metrics = build_gep_projectors(
            self.anchors_inter, m_inter, lambda_gen=lambda_gen, orth_ver=orth_ver
        )

        # 形状のプリントは従来通り（再計算せず形状のみ）
        r = self.anchors_inter[0].shape[0]
        sum_d = sum(S.shape[1] for S in self.anchors_inter)
        print("W_s_tilde.shape", (r, sum_d), "B_s_tilde.shape", (sum_d, sum_d))
        print("lambda_gen", lambda_gen)
        lambdas = metrics["lambdas"]
        print(lambdas)

        # 設定へ反映（従来キー名を維持）
        self.config.jreg_gep = f"{metrics['jreg_gep']:.6g}"
        print(f"Jreg (GEP) = {self.config.jreg_gep}")
        self.config.g_norm_val_gep = f"{metrics['g_norm_val_gep']:.6g}"
        print(f"norm (GEP) = {self.config.g_norm_val_gep}")
        self.config.sum_objective_function = f"{float(np.sum(lambdas)):.4g}"
        print(f"λ の総和 (sum_objective_function): {self.config.sum_objective_function}")
        self.config.g_abs_sum = f"{metrics['g_abs_sum']:.4g}"
        print(f"V_selの絶対値の総和: {self.config.g_abs_sum}")
        self.config.g_mean_var = f"{metrics['g_mean_var']:.4g}"
        print(f"機関ごとのベクトル分散の平均: {self.config.g_mean_var}")
        self.config.g_condition_number = (
            f"{metrics['g_condition_number']:.4g}" if np.isfinite(metrics['g_condition_number']) else "inf"
        )
        print(f"条件数: {self.config.g_condition_number}")

        # projector を適用して属性にセット
        self._apply_projectors_and_set(projs)

        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

        if use_eigen_weighting:
            self.config.eigenvalues = lambdas

    def make_integrate_expression_odc(self) -> None:
        """
        Orthogonal Procrustes Problem (OPP) に基づく統合表現を生成する。
        G_k = U_k V_k^T  where  anchor_k^T @ anchor_1 = U_k Σ_k V_k^T
        """
        print("********************統合表現の生成 (Orthogonal Procrustes) ********************")

        if not self.anchors_inter:
            self.logger.error("アンカーの中間表現が生成されていません。")
            return

        # 2. projector 群を構築して適用
        projs, anchor_1_Z = build_odc_projectors(self.anchors_inter)
        self._apply_projectors_and_set(projs)

        # 互換のため保持
        self.Z_integ = anchor_1_Z
    
        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

    

    def _one_hot(self, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        # classes の順に one-hot を作る（列順が常に一定）
        return (y.reshape(-1, 1) == classes.reshape(1, -1)).astype(float)

    # ------------------------------------------------------------------
    # 〈非線形統合〉　射影行列 P^(k) で Z を最適化する ２段階アルゴリズム
    # ------------------------------------------------------------------
    def make_integrate_nonlinear_expression(self) -> None:
        """
        非線形（カーネル）版：アンカー同士の射影行列で共通ターゲット Z_integ を導き，
    各機関データを同じ次元 m_inter へ写像する。
        """
        m_inter  = self.config.dim_integrate
        # integration.py のビルダーで projector を構築
        projs, Z_integ, eigvals, gammas = build_nonlinear_projectors(
            self.anchors_inter,
            self.Xs_train_inter,
            m_inter,
            gamma_type=getattr(self.config, "gamma_type", "auto"),
            gamma_ratio_krr=getattr(self.config, "gamma_ratio_krr", 1.0),
            K_normalization=bool(getattr(self.config, "K_normalization", False)),
            nl_lambda=getattr(self.config, "nl_lambda", 1e-2),
        )

        # 以前と同様に gamma を表示
        print(gammas)

        # projector を適用して属性にセット
        self._apply_projectors_and_set(projs)

        self.logger.info(f"nonlinear integrate: X_train {self.X_train_integ.shape}")
        print("統合表現の次元数:", self.X_train_integ.shape[1])
        self.logger.info(f"統合表現（訓練）: {self.X_train_integ.shape}")
        self.logger.info(f"統合表現（テスト）: {self.X_test_integ.shape}")

        # 固有値の小さい順に m_inter 個選択し総和
        sum_lambdas = float(np.sum(eigvals[:m_inter]))
        self.config.g_abs_sum = f"{sum_lambdas:.4g}"

        self.Z_integ = Z_integ

        print(f"固有値 λ の上位 {m_inter} 個の総和: {self.config.g_abs_sum}")
        #print(f"固有値 λ の目的関数減少 {p̂} 個の総和: {np.sum(eigvals[idx])}")
        
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
          1) 各機関のアンカー統合表現 A_k を列ごとに標準化（ゼロ平均・単位分散）
          2) D_{ij} = A_i - A_j（行: サンプル, 列: 次元）
          3) 行ごとの L2 距離 ||D_{ij}[n,:]||_2 の「平均」と「標準偏差」を記録

        結果の集約（各ペアの平均距離の平均・標準偏差など）を metrics に入れ、
        self.config.integ_metrics には各ペアの平均距離の平均（1桁丸め）を保存する。

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

        def _standardize(X: np.ndarray) -> np.ndarray:
            """列ごとに標準化（ゼロ平均・単位分散）。分散0列はそのまま0で保持。"""
            if X is None or X.size == 0:
                return X
            mu = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0, ddof=0)
            std_safe = np.where(std > 0, std, 1.0)
            Xz = (X - mu) / std_safe
            # 分散0だった列は 0 に戻す（数値の安定性のため）
            zero_var_cols = (std == 0)
            if np.any(zero_var_cols):
                Xz[:, zero_var_cols] = 0.0
            return Xz

        # 各機関ごとに独立に標準化
        anchors_std = [ _standardize(A) for A in anchors_list ]

        results = []
        for i, j in combinations(range(len(anchors_std)), 2):
            Ai = anchors_std[i]
            Aj = anchors_std[j]

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
                "mean": float(row_dists.mean()),
                "std": float(row_dists.std(ddof=0)),
                "n_rows_used": int(n),
                "dim_used": int(dmin),
            }
            results.append(res)

        if not results:
            metrics = {"pairs": [], "summary": {}}
            self.config.integ_metrics = 100000
            return metrics

        pair_means = np.array([r["mean"] for r in results], dtype=float)
        summary = {
            "pair_count": int(len(results)),
            "mean_of_means": float(pair_means.mean()),
            "std_of_means": float(pair_means.std(ddof=0)),
            "min_mean": float(pair_means.min()),
            "max_mean": float(pair_means.max()),
        }

        metrics = {"pairs": results, "summary": summary}
        self.config.integ_metrics = float(pair_means.mean())  # ← ここに保存（平均距離の平均）
        self.config.integ_metrics = round(self.config.integ_metrics, 5)
        # 簡易出力
        print(f"[integrate_metrics/{which}] ペア数={summary['pair_count']}, "
              f"mean_of_means={summary['mean_of_means']:.6g}, std_of_means={summary['std_of_means']:.6g}, "
              f"min_mean={summary['min_mean']:.6g}, max_mean={summary['max_mean']:.6g}")
        self.logger.info(f"[integrate_metrics/{which}] {summary}")

        return metrics