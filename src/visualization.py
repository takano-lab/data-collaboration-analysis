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

# ...existing code...
    def visualize_anchors(self, save_dir: Optional[str] = None) -> None:
        """
        アンカーデータの変換フローを訓練/テストの2部構成で可視化する。
        上半分(Train): 1.元, 2.中間, 3.射影, 4.統合Z
        下半分(Test):  1.元, 2.中間, 3.射影
        3 と 4 は、元データが3次元なら3Dプロットに切替。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 投影登録用)
        from sklearn.decomposition import PCA

        a = self.a
        save_dir = save_dir or a.config.output_path / "visualizations"

        # --- 必要なデータの存在チェック ---
        train_attrs = ['anchor', 'anchors_inter', 'Z_integ', 'anchors_integ']
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
        fig.suptitle("Anchor Data Transformation Flow (Top: Train, Bottom: Test)", fontsize=16, y=0.995)

        # --- PCAとスケール計算のためのデータ準備 ---
        Z_train_plot = a.Z_integ.T if has_train_data and getattr(a, "Z_integ", None) is not None and a.Z_integ.ndim == 2 and a.Z_integ.shape[0] == a.config.dim_integrate else (a.Z_integ if has_train_data else None)

        col1_data = ([a.anchor] if has_train_data else []) + ([a.anchor_test] if has_test_data else [])
        col2_data = (a.anchors_inter if has_train_data else []) + (a.anchors_test_inter if has_test_data else [])
        col3_data = (a.anchors_integ if has_train_data else []) + (a.anchors_test_integ if has_test_data else [])
        col4_data = [Z_train_plot] if has_train_data and Z_train_plot is not None else []

        def get_2d_data_and_limits(data_list):
            """与えられた行列群を 2 次元に変換し (必要なら PCA)、表示範囲を返す。
            全て既に 2 次元 (shape[1]==2) の場合は PCA を行わずそのまま返す。
            1 次元 (shape[1]==1) の場合は 2 列目を 0 で埋める。
            3 次元以上を含む場合のみ、その高次元データに PCA を適用し 2 次元へ射影。
            """
            if not data_list:
                return [], ((0, 1), (0, 1))
            # すべて 2 次元ならそのまま
            if all(d is not None and getattr(d, 'ndim', 0) == 2 and d.shape[1] == 2 for d in data_list):
                data_2d = data_list
            else:
                prepared = []
                high_dim_sources = []
                order_marks = []
                for d in data_list:
                    if d is None:
                        prepared.append(None); order_marks.append(None); continue
                    if d.shape[1] == 2:
                        prepared.append(d); order_marks.append("keep")
                    elif d.shape[1] == 1:
                        prepared.append(np.hstack([d, np.zeros((d.shape[0], 1))])); order_marks.append("keep")
                    else:  # >2 は後で PCA
                        high_dim_sources.append(d); order_marks.append("proj")
                projected_iter = iter([])
                if high_dim_sources:
                    pca = PCA(n_components=2).fit(np.vstack(high_dim_sources))
                    projected_iter = iter([pca.transform(d) for d in high_dim_sources])
                data_2d = []
                for mark, d in zip(order_marks, data_list):
                    if d is None:
                        data_2d.append(None)
                    elif mark == "keep":
                        if d.shape[1] == 1:
                            data_2d.append(np.hstack([d, np.zeros((d.shape[0], 1))]))
                        else:
                            data_2d.append(d)
                    else:
                        data_2d.append(next(projected_iter))
            # limits
            stacked = np.vstack([d for d in data_2d if d is not None])
            x_min, x_max = stacked[:, 0].min(), stacked[:, 0].max()
            y_min, y_max = stacked[:, 1].min(), stacked[:, 1].max()
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            limits = ((x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad))
            return data_2d, limits

        def get_3d_limits(data_list):
            """3D 配列群から x/y/z の表示範囲を求める。"""
            if not data_list:
                return (0, 1), (0, 1), (0, 1)
            ds = [d for d in data_list if d is not None and getattr(d, 'ndim', 0) == 2 and d.shape[1] == 3]
            if not ds:
                return (0, 1), (0, 1), (0, 1)
            all_data = np.vstack(ds)
            x_min, x_max = float(all_data[:, 0].min()), float(all_data[:, 0].max())
            y_min, y_max = float(all_data[:, 1].min()), float(all_data[:, 1].max())
            z_min, z_max = float(all_data[:, 2].min()), float(all_data[:, 2].max())
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            z_pad = (z_max - z_min) * 0.05 if z_max > z_min else 0.1
            return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad), (z_min - z_pad, z_max + z_pad)

        def to_3d_axes(ax):
            """与えられた 2D Axes を同じ位置の 3D Axes に差し替えて返す。"""
            fig_ = ax.figure
            spec = ax.get_subplotspec()
            ax.remove()
            ax3d_ = fig_.add_subplot(spec, projection='3d')
            return ax3d_

        # 2D 投影と表示範囲
        col1_2d, (xlim1, ylim1) = get_2d_data_and_limits(col1_data)
        col2_2d, (xlim2, ylim2) = get_2d_data_and_limits(col2_data)
        col3_2d, (xlim3, ylim3) = get_2d_data_and_limits(col3_data)
        col4_2d, (xlim4, ylim4) = get_2d_data_and_limits(col4_data)
        test_idx_start = num_institutions if has_train_data else 0
        col3_test_2d_all = col3_2d[test_idx_start:test_idx_start + num_institutions] if has_test_data else []

        # 3D 用の表示範囲（必要な時のみ使用）
        col3_train_limits3d = get_3d_limits(a.anchors_integ) if has_train_data else ((0, 1), (0, 1), (0, 1))
        col3_test_limits3d = get_3d_limits(a.anchors_test_integ) if has_test_data else ((0, 1), (0, 1), (0, 1))
        col4_limits3d = get_3d_limits([Z_train_plot]) if has_train_data and Z_train_plot is not None else ((0, 1), (0, 1), (0, 1))

        # === 4次元以上は PCA(3成分) で 3D に圧縮して描画できるよう準備 ===
        # Train/Test の射影 S_hat 用（anchors_integ）
        col3_train_3d = []
        col3_test_3d = []

        if has_train_data:
            train_more3 = [d for d in a.anchors_integ if d is not None and getattr(d, "ndim", 0) == 2 and d.shape[1] > 3]
            pca3_train = PCA(n_components=3).fit(np.vstack(train_more3)) if train_more3 else None
            for d in a.anchors_integ:
                if d is None or getattr(d, "ndim", 0) != 2:
                    col3_train_3d.append(None)
                elif d.shape[1] == 3:
                    col3_train_3d.append(d)
                elif d.shape[1] > 3 and pca3_train is not None:
                    col3_train_3d.append(pca3_train.transform(d))
                else:
                    col3_train_3d.append(None)  # 2D 以下は 3D 描画しない（2D にフォールバック）
            train_3d_for_limits = [x for x in col3_train_3d if x is not None]
            if train_3d_for_limits:
                xlim3d_train, ylim3d_train, zlim3d_train = get_3d_limits(train_3d_for_limits)
            else:
                xlim3d_train = ylim3d_train = zlim3d_train = (0, 1)

        if has_test_data:
            test_more3 = [d for d in a.anchors_test_integ if d is not None and getattr(d, "ndim", 0) == 2 and d.shape[1] > 3]
            pca3_test = PCA(n_components=3).fit(np.vstack(test_more3)) if test_more3 else None
            for d in a.anchors_test_integ:
                if d is None or getattr(d, "ndim", 0) != 2:
                    col3_test_3d.append(None)
                elif d.shape[1] == 3:
                    col3_test_3d.append(d)
                elif d.shape[1] > 3 and pca3_test is not None:
                    col3_test_3d.append(pca3_test.transform(d))
                else:
                    col3_test_3d.append(None)
            test_3d_for_limits = [x for x in col3_test_3d if x is not None]
            if test_3d_for_limits:
                xlim3d_test, ylim3d_test, zlim3d_test = get_3d_limits(test_3d_for_limits)
            else:
                xlim3d_test = ylim3d_test = zlim3d_test = (0, 1)

        # 統合 Z (Train) 用（Z_train_plot）
        Z_train_3d = None
        if Z_train_plot is not None and getattr(Z_train_plot, "ndim", 0) == 2 and Z_train_plot.shape[0] > 0:
            if Z_train_plot.shape[1] == 3:
                Z_train_3d = Z_train_plot
            elif Z_train_plot.shape[1] > 3:
                pca3_Z = PCA(n_components=3).fit(Z_train_plot)
                Z_train_3d = pca3_Z.transform(Z_train_plot)
        if Z_train_3d is not None:
            xlim3d_Z, ylim3d_Z, zlim3d_Z = get_3d_limits([Z_train_3d])
        else:
            xlim3d_Z = ylim3d_Z = zlim3d_Z = (0, 1)

        # --- プロットループ ---
        for i in range(num_institutions):
            # --- TRAIN DATA (Top Half) ---
            if has_train_data:
                train_row = i
                # 1. Original (Train)
                sns.scatterplot(x=col1_2d[0][:, 0], y=col1_2d[0][:, 1],
                                hue=anchor_labels_train, palette="coolwarm",
                                ax=axes[train_row, 0], legend=(i == 0 and legend_status))
                axes[train_row, 0].set_title("1. Original Anchor (Train)" if i == 0 else "")
                axes[train_row, 0].set_xlim(xlim1); axes[train_row, 0].set_ylim(ylim1)
                axes[train_row, 0].set_ylabel(f"Inst {i+1}")

                # 2. Intermediate (Train)
                sns.scatterplot(x=col2_2d[i][:, 0], y=col2_2d[i][:, 1],
                                hue=anchor_labels_train, palette="coolwarm",
                                ax=axes[train_row, 1], legend=False)
                axes[train_row, 1].set_title("2. Intermediate (Train)" if i == 0 else "")
                axes[train_row, 1].set_xlim(xlim2); axes[train_row, 1].set_ylim(ylim2)

                # 3. Projection S_hat (Train) - 3D 対応（>=4次元は PCA(3)）
                if 'col3_train_3d' in locals() and col3_train_3d and col3_train_3d[i] is not None:
                    ax3d = to_3d_axes(axes[train_row, 2])
                    d3 = col3_train_3d[i]
                    ax3d.scatter(d3[:, 0], d3[:, 1], d3[:, 2],
                                 c=anchor_labels_train, cmap='coolwarm', s=14, depthshade=True)
                    ax3d.set_title("3. Projection S_hat (Train)" if i == 0 else "")
                    if 'xlim3d_train' in locals():
                        ax3d.set_xlim(xlim3d_train); ax3d.set_ylim(ylim3d_train); ax3d.set_zlim(zlim3d_train)
                else:
                    sns.scatterplot(x=col3_2d[i][:, 0], y=col3_2d[i][:, 1],
                                    hue=anchor_labels_train, palette="coolwarm",
                                    ax=axes[train_row, 2], legend=False)
                    axes[train_row, 2].set_title("3. Projection S_hat (Train)" if i == 0 else "")
                    axes[train_row, 2].set_xlim(xlim3); axes[train_row, 2].set_ylim(ylim3)

                # 4. Integrated Z (Train) - 3D 対応（>=4次元は PCA(3)）
                if 'Z_train_3d' in locals() and Z_train_3d is not None:
                    ax3d_z = to_3d_axes(axes[train_row, 3])
                    ax3d_z.scatter(Z_train_3d[:, 0], Z_train_3d[:, 1], Z_train_3d[:, 2],
                                   c=anchor_labels_train, cmap='coolwarm', s=14, depthshade=True)
                    ax3d_z.set_title("4. Integrated Z (Train)" if i == 0 else "")
                    if 'xlim3d_Z' in locals():
                        ax3d_z.set_xlim(xlim3d_Z); ax3d_z.set_ylim(ylim3d_Z); ax3d_z.set_zlim(zlim3d_Z)
                else:
                    if col4_2d:
                        sns.scatterplot(x=col4_2d[0][:, 0], y=col4_2d[0][:, 1],
                                        hue=anchor_labels_train, palette="coolwarm",
                                        ax=axes[train_row, 3], legend=False)
                        axes[train_row, 3].set_title("4. Integrated Z (Train)" if i == 0 else "")
                        axes[train_row, 3].set_xlim(xlim4); axes[train_row, 3].set_ylim(ylim4)
                    else:
                        axes[train_row, 3].set_visible(False)

            # --- TEST DATA (Bottom Half) ---
            if has_test_data:
                test_row = i + num_institutions
                train_offset = 1 if has_train_data else 0

                # 1. Original (Test)
                anchor_test_2d = col1_2d[train_offset]
                sns.scatterplot(x=anchor_test_2d[:, 0], y=anchor_test_2d[:, 1],
                                hue=anchor_labels_test, palette="viridis",
                                ax=axes[test_row, 0], legend=(i == 0 and legend_status))
                axes[test_row, 0].set_title("1. Original Anchor (Test)" if i == 0 else "")
                axes[test_row, 0].set_xlim(xlim1); axes[test_row, 0].set_ylim(ylim1)
                axes[test_row, 0].set_ylabel(f"Inst {i+1}")

                # 2. Intermediate (Test)
                idx2 = train_offset * num_institutions + i
                sns.scatterplot(x=col2_2d[idx2][:, 0], y=col2_2d[idx2][:, 1],
                                hue=anchor_labels_test, palette="viridis",
                                ax=axes[test_row, 1], legend=False)
                axes[test_row, 1].set_title("2. Intermediate (Test)" if i == 0 else "")
                axes[test_row, 1].set_xlim(xlim2); axes[test_row, 1].set_ylim(ylim2)

                # 3. Projection S_hat (Test) - 3D 対応（>=4次元は PCA(3)）
                if 'col3_test_3d' in locals() and col3_test_3d and col3_test_3d[i] is not None:
                    ax3d_t = to_3d_axes(axes[test_row, 2])
                    d3t = col3_test_3d[i]
                    ax3d_t.scatter(d3t[:, 0], d3t[:, 1], d3t[:, 2],
                                   c=anchor_labels_test, cmap='viridis', s=14, depthshade=True)
                    ax3d_t.set_title("3. Projection S_hat (Test)" if i == 0 else "")
                    if 'xlim3d_test' in locals():
                        ax3d_t.set_xlim(xlim3d_test); ax3d_t.set_ylim(ylim3d_test); ax3d_t.set_zlim(zlim3d_test)
                else:
                    sns.scatterplot(x=col3_2d[idx2][:, 0], y=col3_2d[idx2][:, 1],
                                    hue=anchor_labels_test, palette="viridis",
                                    ax=axes[test_row, 2], legend=False)
                    axes[test_row, 2].set_title("3. Projection S_hat (Test)" if i == 0 else "")
                    axes[test_row, 2].set_xlim(xlim3); axes[test_row, 2].set_ylim(ylim3)

                # 4. Projection Composition (Test) - 全機関の射影を重ね合わせる
                valid_test_3d = [d for d in col3_test_3d if d is not None] if 'col3_test_3d' in locals() else []
                if valid_test_3d:
                    ax3d_comp = to_3d_axes(axes[test_row, 3])
                    for j, d3_other in enumerate(col3_test_3d):
                        if d3_other is None:
                            continue
                        alpha_val = 1.0 if j == i else 1
                        ax3d_comp.scatter(
                            d3_other[:, 0], d3_other[:, 1], d3_other[:, 2],
                            c=anchor_labels_test, cmap='viridis', s=14, depthshade=True, alpha=alpha_val
                        )
                    ax3d_comp.set_title('4. Projection Composition (Test)' if i == 0 else '')
                    if 'xlim3d_test' in locals():
                        ax3d_comp.set_xlim(xlim3d_test); ax3d_comp.set_ylim(ylim3d_test); ax3d_comp.set_zlim(zlim3d_test)
                else:
                    axis4 = axes[test_row, 3]
                    has_drawn = False
                    for j, data_other in enumerate(col3_test_2d_all):
                        if data_other is None or getattr(data_other, 'ndim', 0) != 2:
                            continue
                        alpha_val = 1.0 if j == i else 1
                        legend_option = legend_status if (j == i and i == 0 and legend_status) else False
                        sns.scatterplot(
                            x=data_other[:, 0], y=data_other[:, 1],
                            hue=anchor_labels_test, palette='viridis',
                            ax=axis4, legend=legend_option, alpha=alpha_val
                        )
                        has_drawn = True
                    if has_drawn:
                        axis4.set_visible(True)
                        axis4.set_title('4. Projection Composition (Test)' if i == 0 else '')
                        axis4.set_xlim(xlim3); axis4.set_ylim(ylim3)
                    else:
                        axis4.set_visible(False)

        # レイアウト調整と保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f"{a.config.plot_name}_anchor_visualization.png"
            plt.savefig(save_path)
            self._log(f"✅ アンカーデータの可視化を保存しました: {save_path}")
# ...existing code...

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
        if not a.Xs_train or not a.Xs_train_inter or not a.Xs_train_integ:
            self._log("可視化する表現が生成されていません。run()メソッドを実行してください。")
            return

        num_institutions = a.config.num_institution

        # 統合表現を機関ごとに再分割

        Xs_train_integ_split = a.Xs_train_integ
        Xs_test_integ_split = a.Xs_test_integ

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
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=a.ys_train_integ[i],
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
                y_other = np.hstack([a.ys_train_integ[j] for j in other_institutions_indices])
                sns.scatterplot(
                    x=X_other[:, 0], y=X_other[:, 1], hue=y_other,
                    palette="viridis", ax=axes_train[i, 3], legend=False, alpha=1.0
                )
            sns.scatterplot(
                x=Xs_train_integ_split[i][:, 0], y=Xs_train_integ_split[i][:, 1], hue=a.ys_train_integ[i],
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
