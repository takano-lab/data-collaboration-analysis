from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class DataCollabVisualizer:
    """
    DataCollaborationAnalysis の可視化を担うクラス。
    既存の visualize_anchors / visualize_representations 相当の処理をこちらへ集約。
    """

    def __init__(self, analysis, logger=None) -> None:
        # analysis: DataCollaborationAnalysis のインスタンス
        self.a = analysis
        self.logger = logger or getattr(analysis, "logger", None)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                pass

    def visualize_anchors(self, save_dir: Optional[str] = None) -> None:
        """
        アンカーデータの変換フローを訓練/テストの2部構成で可視化する。
        上半分(Train): 1.元, 2.中間, 3.射影, 4.統合Z
        下半分(Test):  1.元, 2.中間, 3.射影
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        a = self.a
        save_dir = save_dir or a.config.output_path / "visualizations"

        # --- 必要なデータの存在チェック ---
        train_attrs = ['anchor', 'anchors_inter', 'Z', 'anchors_integ']
        test_attrs = ['anchor_test', 'anchors_test_inter', 'anchors_test_integ']

        has_train_data = all(hasattr(a, attr) and getattr(a, attr) is not None and len(getattr(a, attr, [])) > 0 for attr in train_attrs)
        has_test_data = all(hasattr(a, attr) and getattr(a, attr) is not None and len(getattr(a, attr, [])) > 0 for attr in test_attrs)

        if not has_train_data and not has_test_data:
            self._log("可視化に必要な訓練データもテストデータも存在しません。")
            return

        num_institutions = len(a.anchors_inter) if has_train_data else len(a.anchors_test_inter)
        if num_institutions == 0:
            return

        # --- ラベルの準備 ---
        a.assign_anchor_labels()
        anchor_labels_train = a.anchor_y if hasattr(a, 'anchor_y') else np.zeros(a.anchor.shape[0] if has_train_data else 0)
        anchor_labels_test = a.anchor_y_test if hasattr(a, 'anchor_y_test') else np.zeros(a.anchor_test.shape[0] if has_test_data else 0)
        legend_status = "full" if np.unique(anchor_labels_train).size > 1 else False

        # --- プロットの準備 (Train+Testで2倍の行数) ---
        fig, axes = plt.subplots(num_institutions * 2, 4, figsize=(24, 6 * num_institutions * 2), squeeze=False)
        fig.suptitle("Anchor Data Transformation Flow (Top: Train, Bottom: Test)", fontsize=16, y=1.0)

        # --- PCAとスケール計算のためのデータ準備 ---
        Z_train_plot = a.Z.T if has_train_data and a.Z.shape[0] == a.config.dim_integrate else (a.Z if has_train_data else None)

        col1_data = ([a.anchor] if has_train_data else []) + ([a.anchor_test] if has_test_data else [])
        col2_data = (a.anchors_inter if has_train_data else []) + (a.anchors_test_inter if has_test_data else [])
        col3_data = (a.anchors_integ if has_train_data else []) + (a.anchors_test_integ if has_test_data else [])
        col4_data = [Z_train_plot] if has_train_data else []

        def get_2d_data_and_limits(data_list):
            if not data_list:
                return [], ((0, 1), (0, 1))
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
                sns.scatterplot(x=col1_2d[0][:, 0], y=col1_2d[0][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 0], legend=(i == 0 and legend_status))
                axes[train_row, 0].set_title(f"1. Original Anchor (Train)" if i == 0 else "")
                axes[train_row, 0].set_xlim(xlim1)
                axes[train_row, 0].set_ylim(ylim1)
                axes[train_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(x=col2_2d[i][:, 0], y=col2_2d[i][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 1], legend=False)
                axes[train_row, 1].set_title(f"2. Intermediate (Train)" if i == 0 else "")
                axes[train_row, 1].set_xlim(xlim2)
                axes[train_row, 1].set_ylim(ylim2)

                sns.scatterplot(x=col3_2d[i][:, 0], y=col3_2d[i][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 2], legend=False)
                axes[train_row, 2].set_title(f"3. Projection S_hat (Train)" if i == 0 else "")
                axes[train_row, 2].set_xlim(xlim3)
                axes[train_row, 2].set_ylim(ylim3)

                sns.scatterplot(x=col4_2d[0][:, 0], y=col4_2d[0][:, 1], hue=anchor_labels_train, palette="coolwarm", ax=axes[train_row, 3], legend=False)
                axes[train_row, 3].set_title(f"4. Integrated Z (Train)" if i == 0 else "")
                axes[train_row, 3].set_xlim(xlim4)
                axes[train_row, 3].set_ylim(ylim4)

            # --- TEST DATA (Bottom Half) ---
            if has_test_data:
                test_row = i + num_institutions
                train_offset = 1 if has_train_data else 0

                anchor_test_2d = col1_2d[train_offset]
                sns.scatterplot(x=anchor_test_2d[:, 0], y=anchor_test_2d[:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 0], legend=(i == 0 and legend_status))
                axes[test_row, 0].set_title(f"1. Original Anchor (Test)" if i == 0 else "")
                axes[test_row, 0].set_xlim(xlim1)
                axes[test_row, 0].set_ylim(ylim1)
                axes[test_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(x=col2_2d[train_offset * num_institutions + i][:, 0], y=col2_2d[train_offset * num_institutions + i][:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 1], legend=False)
                axes[test_row, 1].set_title(f"2. Intermediate (Test)" if i == 0 else "")
                axes[test_row, 1].set_xlim(xlim2)
                axes[test_row, 1].set_ylim(ylim2)

                sns.scatterplot(x=col3_2d[train_offset * num_institutions + i][:, 0], y=col3_2d[train_offset * num_institutions + i][:, 1], hue=anchor_labels_test, palette="viridis", ax=axes[test_row, 2], legend=False)
                axes[test_row, 2].set_title(f"3. Projection S_hat (Test)" if i == 0 else "")
                axes[test_row, 2].set_xlim(xlim3)
                axes[test_row, 2].set_ylim(ylim3)

                # 4列目は空欄にする
                axes[test_row, 3].set_visible(False)

        # レイアウト調整と保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f"anchor_visualization_{a.config.plot_name}"
            plt.savefig(save_path)
            self._log(f"✅ アンカーデータの可視化を保存しました: {save_path}")

    def visualize_representations(self, save_dir: Optional[str] = None) -> None:
        """
        元データ、中間表現、統合表現（機関ごとと全体）を2次元散布図で可視化する。
        訓練データとテストデータをそれぞれ別の図で出力する。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        a = self.a
        a.assign_anchor_labels()
        self.visualize_anchors(save_dir=save_dir)

        save_dir = save_dir or a.config.output_path / "visualizations"
        if not a.Xs_train or not a.Xs_train_inter or a.X_train_integ.size == 0:
            print("可視化する表現が生成されていません。run()メソッドを実行してください。")
            return

        num_institutions = a.config.num_institution

        # 統合表現を機関ごとに再分割
        train_sizes = [len(y) for y in a.ys_train]
        test_sizes = [len(y) for y in a.ys_test]
        train_indices = np.cumsum([0] + train_sizes)
        test_indices = np.cumsum([0] + test_sizes)

        Xs_train_integ_split = [a.X_train_integ[train_indices[i]:train_indices[i+1]] for i in range(num_institutions)]
        Xs_test_integ_split = [a.X_test_integ[test_indices[i]:test_indices[i+1]] for i in range(num_institutions)]

        # 統合表現プロットの軸スケールを統一するための範囲計算
        # Train
        x_min_train, x_max_train = a.X_train_integ[:, 0].min(), a.X_train_integ[:, 0].max()
        y_min_train, y_max_train = a.X_train_integ[:, 1].min(), a.X_train_integ[:, 1].max()
        x_pad_train = (x_max_train - x_min_train) * 0.05
        y_pad_train = (y_max_train - y_min_train) * 0.05
        xlim_train = (x_min_train - x_pad_train, x_max_train + x_pad_train)
        ylim_train = (y_min_train - y_pad_train, y_max_train + y_pad_train)

        # Test
        x_min_test, x_max_test = a.X_test_integ[:, 0].min(), a.X_test_integ[:, 0].max()
        y_min_test, y_max_test = a.X_test_integ[:, 1].min(), a.X_test_integ[:, 1].max()
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
                x=a.Xs_train[i][:, 0], y=a.Xs_train[i][:, 1], hue=a.ys_train[i],
                palette="viridis", ax=axes_train[i, 0], legend="full"
            )
            axes_train[i, 0].set_title(f"Institution {i+1} - Original Data")
            axes_train[i, 0].set_xlabel("Dimension 1")
            axes_train[i, 0].set_ylabel("Dimension 2")

            # 2. 中間表現 (Train)
            sns.scatterplot(
                x=a.Xs_train_inter[i][:, 0], y=a.Xs_train_inter[i][:, 1], hue=a.ys_train[i],
                palette="viridis", ax=axes_train[i, 1], legend="full"
            )
            axes_train[i, 1].set_title(f"Institution {i+1} - Intermediate Expression")
            axes_train[i, 1].set_xlabel("Dimension 1")
            axes_train[i, 1].set_ylabel("Dimension 2")

            # 3. 統合表現 (Train) - 機関ごと
            sns.scatterplot(
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=a.ys_train[i],
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
                y_other = np.hstack([a.ys_train[j] for j in other_institutions_indices])
                sns.scatterplot(
                    x=X_other[:, 0], y=X_other[:, 1], hue=y_other,
                    palette="viridis", ax=axes_train[i, 3], legend=False, alpha=1.0
                )
            sns.scatterplot(
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=a.ys_train[i],
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
            plt.savefig(Path(save_dir) / a.config.plot_name)
