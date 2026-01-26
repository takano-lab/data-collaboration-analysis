from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config.config import Config
from src.common import IntegratedArtifacts


class DataCollabVisualizer:
    """
    Helper to visualize anchors and representations produced by the
    three-layer pipeline (dataset → intermediate → integrated).
    """

    def __init__(self, *, config: Config, artifacts: IntegratedArtifacts, logger=None) -> None:
        self.config = config
        self.logger = logger
        self.artifacts = artifacts
        self.intermediate = artifacts.intermediate
        self.dataset = artifacts.intermediate.dataset

    # ------------------------------------------------------------------ #
    def visualize_anchors(self, save_dir: Optional[str] = None) -> None:
        """
        Draw anchor transformation flow (original → intermediate → integrated).
        """
        config = self.config
        if not (
            getattr(config, "visualize_for_anchor", False)
            or getattr(config, "visualize_anchors_3d", False)
        ):
            return
        import matplotlib.pyplot as plt
        import seaborn as sns
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from sklearn.decomposition import PCA

        inter = self.intermediate
        integ = self.artifacts

        save_dir = Path(save_dir or (config.output_path / "visualizations"))

        has_train_data = bool(inter.anchor.size and inter.anchors_inter)
        if not has_train_data:
            self._log("Anchor visualization skipped: no anchor data available.")
            return
        num_institutions = len(inter.anchors_inter)
        if num_institutions == 0:
            return

        anchor_labels_train = inter.anchor_y if inter.anchor_y.size else np.zeros(inter.anchor.shape[0])

        col1_data = inter.anchor
        col2_data = inter.anchors_inter
        col3_data = integ.anchors_integ
        use_graph = bool(getattr(config, "visual_knn_graph", False))

        def ensure_2d(data_list: Sequence[np.ndarray]):
            if not data_list:
                return [], ((0, 1), (0, 1))
            prepared = []
            for arr in data_list:
                if arr is None:
                    prepared.append(None)
                    continue
                if arr.ndim != 2:
                    arr = np.atleast_2d(arr)
                if arr.shape[1] == 1:
                    arr = np.hstack([arr, np.zeros((arr.shape[0], 1))])
                prepared.append(arr[:, :2])
            has_data = any(arr is not None for arr in prepared)
            stacked = np.vstack([arr for arr in prepared if arr is not None]) if has_data else None
            if stacked is None or stacked.size == 0:
                limits = ((0, 1), (0, 1))
            else:
                xmin, xmax = stacked[:, 0].min(), stacked[:, 0].max()
                ymin, ymax = stacked[:, 1].min(), stacked[:, 1].max()
                xpad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
                ypad = (ymax - ymin) * 0.05 if ymax > ymin else 0.1
                limits = ((xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
            return prepared, limits

        col1_2d, (xlim1, ylim1) = ensure_2d([col1_data])
        col2_2d, (xlim2, ylim2) = ensure_2d(col2_data)
        col3_2d, (xlim3, ylim3) = ensure_2d(col3_data)

        fig, axes = plt.subplots(num_institutions, 3, figsize=(18, 5 * num_institutions), squeeze=False)

        def project_3d(data_list: Sequence[np.ndarray]):
            # 3 次元以上のデータを 3D 表示用に準備。
            # 3 次元そのまま / 4 次元以上は PCA で 3 次元に射影。
            valid = [d for d in data_list if d is not None and d.ndim == 2 and d.shape[1] >= 3]
            if not valid:
                return [None for _ in data_list], None
            pca = PCA(n_components=3, svd_solver="full").fit(np.vstack(valid))
            projected = []
            for d in data_list:
                if d is None:
                    projected.append(None)
                elif d.shape[1] == 3:
                    projected.append(d)
                elif d.shape[1] > 3:
                    projected.append(pca.transform(d))
                else:
                    projected.append(None)
            stacked = [d for d in projected if d is not None]
            if stacked:
                arr = np.vstack(stacked)
                limits = tuple(
                    (arr[:, idx].min() - 0.05 * (arr[:, idx].ptp() or 1.0), arr[:, idx].max() + 0.05 * (arr[:, idx].ptp() or 1.0))
                    for idx in range(3)
                )
            else:
                limits = ((0, 1), (0, 1), (0, 1))
            return projected, limits

        enable_3d = bool(getattr(self.config, "visualize_anchors_3d", False))
        if enable_3d:
            col1_train_3d, col1_train_limits = project_3d([inter.anchor])
            col3_train_3d, train3d_limits = project_3d(integ.anchors_integ)

            def to_3d(ax):
                fig_ = ax.figure
                spec = ax.get_subplotspec()
                ax.remove()
                return fig_.add_subplot(spec, projection="3d")
        else:
            col1_train_3d = col3_train_3d = None
            col1_train_limits = train3d_limits = None

        mst_edges = self._compute_mst_edges(np.asarray(col1_data)) if use_graph else []

        def draw_edges_2d(ax, data2d, edges):
            if data2d is None or not edges:
                return
            for u, v in edges:
                if u >= data2d.shape[0] or v >= data2d.shape[0]:
                    continue
                ax.plot([data2d[u, 0], data2d[v, 0]], [data2d[u, 1], data2d[v, 1]], color="gray", alpha=0.6, linewidth=1.0)

        def draw_edges_3d(ax, data3d, edges):
            if data3d is None or not edges:
                return
            for u, v in edges:
                if u >= data3d.shape[0] or v >= data3d.shape[0]:
                    continue
                ax.plot(
                    [data3d[u, 0], data3d[v, 0]],
                    [data3d[u, 1], data3d[v, 1]],
                    [data3d[u, 2], data3d[v, 2]],
                    color="gray", alpha=0.6, linewidth=1.0,
                )

        # フォント（日本語）を試しに検出
        def get_jp_font():
            from matplotlib import font_manager
            preferred = [
                "IPAexGothic", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
                "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic",
                "Yu Gothic", "MS Gothic",
            ]
            for name in preferred:
                try:
                    path = font_manager.findfont(name, fallback_to_default=False)
                    if path:
                        return font_manager.FontProperties(fname=path)
                except Exception:
                    continue
            return None

        jp_font = self._get_jp_font() if callable(getattr(self, "_get_jp_font", None)) else get_jp_font()

        def add_row_label(ax, text: str):
            if getattr(ax, "name", "") == "3d":
                ax.text2D(-0.12, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=20, fontweight="bold", rotation=90, fontproperties=jp_font)
            else:
                ax.text(-0.12, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=20, fontweight="bold", rotation=90, fontproperties=jp_font)

        point_size_2d_default = int(getattr(config, "visualize_point_size_2d", 60) or 60)
        point_size_3d_default = int(getattr(config, "visualize_point_size_3d", 20) or 20)
        anchor_point_size_2d = int(getattr(config, "visualize_point_size_anchor_2d", point_size_2d_default) or point_size_2d_default)
        anchor_point_size_3d = int(getattr(config, "visualize_point_size_anchor_3d", point_size_3d_default) or point_size_3d_default)

        col_titles = ["アンカーデータ", "中間表現", "統合表現"]
        for ci, title in enumerate(col_titles):
            xpos = (ci + 0.5) / 3.0
            fig.text(xpos, 0.98, title, ha="center", va="top", fontsize=20, fontweight="bold", fontproperties=jp_font)

        for i in range(num_institutions):
            train_row = i
            ax_orig = axes[train_row, 0]
            if enable_3d and col1_train_3d and col1_train_3d[0] is not None:
                ax_orig = to_3d(ax_orig)
                axes[train_row, 0] = ax_orig
                d3o = col1_train_3d[0]
                ax_orig.scatter(d3o[:, 0], d3o[:, 1], d3o[:, 2], c=anchor_labels_train, cmap="copper", s=anchor_point_size_3d, depthshade=True)
                if col1_train_limits:
                    ax_orig.set_xlim(col1_train_limits[0]); ax_orig.set_ylim(col1_train_limits[1]); ax_orig.set_zlim(col1_train_limits[2])
                if use_graph:
                    draw_edges_3d(ax_orig, d3o, mst_edges)
            else:
                sns.scatterplot(
                    x=col1_2d[0][:, 0], y=col1_2d[0][:, 1],
                    hue=anchor_labels_train, palette="copper",
                    ax=ax_orig, legend=False,
                    s=anchor_point_size_2d,
                )
                ax_orig.set_xlim(xlim1); ax_orig.set_ylim(ylim1)
                if use_graph:
                    draw_edges_2d(ax_orig, col1_2d[0], mst_edges)
            add_row_label(ax_orig, f"機関 {i+1}")

            sns.scatterplot(
                x=col2_2d[i][:, 0], y=col2_2d[i][:, 1],
                hue=anchor_labels_train, palette="copper",
                ax=axes[train_row, 1], legend=False,
                s=anchor_point_size_2d,
            )
            axes[train_row, 1].set_xlim(xlim2); axes[train_row, 1].set_ylim(ylim2)
            if use_graph:
                draw_edges_2d(axes[train_row, 1], col2_2d[i], mst_edges)

            if enable_3d and col3_train_3d and col3_train_3d[i] is not None:
                ax3d = to_3d(axes[train_row, 2])
                d3 = col3_train_3d[i]
                ax3d.scatter(d3[:, 0], d3[:, 1], d3[:, 2], c=anchor_labels_train, cmap="copper", s=anchor_point_size_3d, depthshade=True)
                if train3d_limits:
                    ax3d.set_xlim(train3d_limits[0]); ax3d.set_ylim(train3d_limits[1]); ax3d.set_zlim(train3d_limits[2])
                if use_graph:
                    draw_edges_3d(ax3d, d3, mst_edges)
            else:
                sns.scatterplot(
                    x=col3_2d[i][:, 0], y=col3_2d[i][:, 1],
                    hue=anchor_labels_train, palette="copper",
                    ax=axes[train_row, 2], legend=False,
                    s=anchor_point_size_2d,
                )
                axes[train_row, 2].set_xlim(xlim3); axes[train_row, 2].set_ylim(ylim3)
                if use_graph:
                    draw_edges_2d(axes[train_row, 2], col3_2d[i], mst_edges)

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / self._plot_filename("anchor_visualization"))

    # ------------------------------------------------------------------ #
    def visualize_anchors_for_presenations(self, save_dir: Optional[str] = None) -> None:
        """
        Presentation用のアンカー簡易可視化。
        少数（デフォルト10点）だけ抜き出し、マーカー形状・色を大きく変えて強調表示する。
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        inter = self.intermediate
        integ = self.artifacts
        config = self.config

        if not getattr(config, "visualize_for_presenations", False):
            return

        if inter.anchor.size == 0 or not inter.anchors_inter:
            self._log("Presentation anchors skipped: no anchor data available.")
            return

        num_institutions = len(inter.anchors_inter)
        if num_institutions == 0:
            return

        anchor_labels = inter.anchor_y if inter.anchor_y.size else np.zeros(inter.anchor.shape[0])
        anchor = inter.anchor
        n_pick = min(10, anchor.shape[0])
        idx = np.linspace(0, anchor.shape[0] - 1, n_pick, dtype=int)

        def _subset(arr):
            if arr is None or arr.shape[0] == 0:
                return None
            if arr.shape[0] <= idx.max():
                return None
            return arr[idx]

        col1_sub = _subset(anchor)
        col2_sub = [_subset(a) for a in inter.anchors_inter]
        col3_sub = [_subset(a) for a in integ.anchors_integ]
        use_graph = bool(getattr(config, "visual_knn_graph", False))

        def project_3d_list(arr_list):
            valid = [a for a in arr_list if a is not None and a.ndim == 2 and a.shape[1] >= 3]
            if not valid:
                return [None for _ in arr_list], None
            pca = PCA(n_components=3, svd_solver="full").fit(np.vstack(valid))
            projected = []
            for a in arr_list:
                if a is None:
                    projected.append(None)
                elif a.shape[1] == 3:
                    projected.append(a)
                elif a.shape[1] > 3:
                    projected.append(pca.transform(a))
                else:
                    projected.append(None)
            stacked = [a for a in projected if a is not None]
            if stacked:
                arr = np.vstack(stacked)
                limits = tuple(
                    (arr[:, idx].min() - 0.05 * (arr[:, idx].ptp() or 1.0), arr[:, idx].max() + 0.05 * (arr[:, idx].ptp() or 1.0))
                    for idx in range(3)
                )
            else:
                limits = ((0, 1), (0, 1), (0, 1))
            return projected, limits

        enable_3d = bool(getattr(config, "visualize_anchors_3d", False))
        col1_3d = col2_3d = col3_3d = None
        limits1 = limits2 = limits3 = None
        if enable_3d:
            col1_3d, limits1 = project_3d_list([col1_sub] if col1_sub is not None else [])
            col2_3d, limits2 = project_3d_list(col2_sub)
            col3_3d, limits3 = project_3d_list(col3_sub)
        mst_edges = []
        if use_graph and col1_sub is not None:
            mst_edges = self._compute_mst_edges(np.asarray(col1_sub))

        jp_font = self._get_jp_font()
        markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "H", "+"]
        colors = sns.color_palette("tab10", n_pick)

        def draw_edges_2d(ax, data2d, edges):
            if data2d is None or not edges:
                return
            data2d = np.asarray(data2d)
            for u, v in edges:
                if u >= data2d.shape[0] or v >= data2d.shape[0]:
                    continue
                ax.plot([data2d[u, 0], data2d[v, 0]], [data2d[u, 1], data2d[v, 1]], color="gray", alpha=0.6, linewidth=1.0)

        def draw_edges_3d(ax, data3d, edges):
            if data3d is None or not edges:
                return
            data3d = np.asarray(data3d)
            for u, v in edges:
                if u >= data3d.shape[0] or v >= data3d.shape[0]:
                    continue
                ax.plot(
                    [data3d[u, 0], data3d[v, 0]],
                    [data3d[u, 1], data3d[v, 1]],
                    [data3d[u, 2], data3d[v, 2]],
                    color="gray", alpha=0.6, linewidth=1.0,
                )

        def scatter_with_markers(ax, arr2d, labels):
            if arr2d is None:
                return
            arr2d = np.asarray(arr2d)
            if arr2d.ndim != 2:
                return
            if arr2d.shape[1] < 2:
                return
            arr2d = arr2d[:, :2]
            for j, (x, y) in enumerate(arr2d):
                mk = markers[j % len(markers)]
                ax.scatter(x, y, s=300, marker=mk, color=colors[j % len(colors)], edgecolor="k", linewidth=1.5)
            if use_graph:
                draw_edges_2d(ax, arr2d, mst_edges)

        def scatter_with_markers_3d(ax, arr3d, labels, limits=None):
            if arr3d is None:
                return
            arr3d = np.asarray(arr3d)
            if arr3d.ndim != 2 or arr3d.shape[1] < 3:
                return
            for j, (x, y, z) in enumerate(arr3d):
                mk = markers[j % len(markers)]
                ax.scatter(x, y, z, s=300, marker=mk, color=colors[j % len(colors)], edgecolor="k", linewidth=1.5, depthshade=True)
            if limits:
                ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
            if use_graph:
                draw_edges_3d(ax, arr3d, mst_edges)

        fig, axes = plt.subplots(num_institutions, 3, figsize=(18, 5 * num_institutions), squeeze=False)
        col_titles = ["アンカーデータ", "中間表現", "統合表現"]
        for ci, title in enumerate(col_titles):
            xpos = (ci + 0.5) / 3.0
            fig.text(xpos, 0.98, title, ha="center", va="top", fontsize=20, fontweight="bold", fontproperties=jp_font)

        for i in range(num_institutions):
            # Original anchor (shared)
            ax0 = axes[i, 0]
            if enable_3d and col1_3d and col1_3d[0] is not None:
                ax0 = fig.add_subplot(ax0.get_subplotspec(), projection="3d")
                axes[i, 0] = ax0
                scatter_with_markers_3d(ax0, col1_3d[0], anchor_labels[idx], limits1)
            else:
                scatter_with_markers(ax0, col1_sub, anchor_labels[idx])

            # Intermediate per institution
            ax1 = axes[i, 1]
            if enable_3d and col2_3d and col2_3d[i] is not None:
                ax1 = fig.add_subplot(ax1.get_subplotspec(), projection="3d")
                axes[i, 1] = ax1
                scatter_with_markers_3d(ax1, col2_3d[i], anchor_labels[idx], limits2)
            else:
                scatter_with_markers(ax1, col2_sub[i], anchor_labels[idx])

            # Integrated per institution
            ax2 = axes[i, 2]
            if enable_3d and col3_3d and col3_3d[i] is not None:
                ax2 = fig.add_subplot(ax2.get_subplotspec(), projection="3d")
                axes[i, 2] = ax2
                scatter_with_markers_3d(ax2, col3_3d[i], anchor_labels[idx], limits3)
            else:
                scatter_with_markers(ax2, col3_sub[i], anchor_labels[idx])

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
        save_dir = Path(save_dir or (config.output_path / "visualizations"))
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / self._plot_filename("anchor_presentation"))

    # ------------------------------------------------------------------ #
    def visualize_representations(self, save_dir: Optional[str] = None) -> None:
        """
        Plot original/intermediate/integrated representations for each institution.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        dataset = self.dataset
        inter = self.intermediate
        integ = self.artifacts
        cfg = self.config

        do_train = bool(getattr(cfg, "visualize_for_train", False))
        do_test = bool(getattr(cfg, "visualize_for_test", False))
        do_anchor = bool(
            getattr(cfg, "visualize_for_anchor", False)
            or getattr(cfg, "visualize_for_presenations", False)
            or getattr(cfg, "visualize_anchors_3d", False)
        )

        if do_train and (not dataset.Xs_train or not inter.Xs_train_inter or not integ.Xs_train_integ):
            self._log("Train visualization skipped: insufficient training data.")
            do_train = False

        if do_test and (not dataset.Xs_test or not inter.Xs_test_inter or not integ.Xs_test_integ):
            self._log("Test visualization skipped: insufficient test data.")
            do_test = False

        if not any([do_train, do_test, do_anchor]):
            return

        save_dir = Path(save_dir or (cfg.output_path / "visualizations"))
        save_dir.mkdir(parents=True, exist_ok=True)

        num_institutions = len(dataset.Xs_train) if dataset.Xs_train else 0
        train_concat = self._stack_arrays(integ.Xs_train_integ) if do_train else None
        if do_train and train_concat is None:
            self._log("Train visualization skipped: integrated train embeddings are empty.")
            do_train = False
        xlim_train = ylim_train = None
        if do_train and train_concat is not None:
            xlim_train, ylim_train = self._compute_limits(train_concat)

        test_concat = self._stack_arrays(integ.Xs_test_integ) if do_test else None
        if do_test and test_concat is not None:
            xlim_test, ylim_test = self._compute_limits(test_concat)
        else:
            xlim_test = ylim_test = None

        # Integrated 表現用に全機関で共有する 3D 範囲を計算
        integ_train_3d_limits = None
        integ_test_3d_limits = None

        def project_3d_list(arr_list):
            valid = [a for a in arr_list if a is not None and a.ndim == 2 and a.shape[1] >= 3]
            if not valid:
                return [None for _ in arr_list], None
            pca = PCA(n_components=3, svd_solver="full").fit(np.vstack(valid))
            projected = []
            for a in arr_list:
                if a is None:
                    projected.append(None)
                elif a.shape[1] == 3:
                    projected.append(a)
                elif a.shape[1] > 3:
                    projected.append(pca.transform(a))
                else:
                    projected.append(None)
            stacked = [a for a in projected if a is not None]
            if stacked:
                arr = np.vstack(stacked)
                limits = tuple(
                    (arr[:, idx].min() - 0.05 * (arr[:, idx].ptp() or 1.0), arr[:, idx].max() + 0.05 * (arr[:, idx].ptp() or 1.0))
                    for idx in range(3)
                )
            else:
                limits = ((0, 1), (0, 1), (0, 1))
            return projected, limits

        use_graph = bool(getattr(cfg, "visual_knn_graph", False))

        enable_3d = bool(getattr(cfg, "visualize_anchors_3d", False))

        point_size_2d_default = int(getattr(cfg, "visualize_point_size_2d", 60) or 60)
        point_size_3d_default = int(getattr(cfg, "visualize_point_size_3d", 20) or 20)

        train_point_size_default = int(getattr(cfg, "visualize_point_size_train", point_size_2d_default) or point_size_2d_default)
        train_point_size_2d = int(getattr(cfg, "visualize_point_size_train_2d", train_point_size_default) or train_point_size_default)
        train_point_size_3d = int(getattr(cfg, "visualize_point_size_train_3d", point_size_3d_default) or point_size_3d_default)

        test_point_size_default = int(getattr(cfg, "visualize_point_size_test", point_size_2d_default) or point_size_2d_default)
        test_point_size_2d = int(getattr(cfg, "visualize_point_size_test_2d", test_point_size_default) or test_point_size_default)
        test_point_size_3d = int(getattr(cfg, "visualize_point_size_test_3d", point_size_3d_default) or point_size_3d_default)
        orig_train_3d = inter_train_3d = integ_train_3d = None
        orig_test_3d = inter_test_3d = integ_test_3d = None
        orig_train_limits = inter_train_limits = integ_train_limits = None
        orig_test_limits = inter_test_limits = integ_test_limits = None
        if enable_3d and (do_train or do_test):
            if do_train:
                orig_train_3d, orig_train_limits = project_3d_list(dataset.Xs_train)
                inter_train_3d, inter_train_limits = project_3d_list(inter.Xs_train_inter)
                integ_train_3d, integ_train_limits = project_3d_list(integ.Xs_train_integ)
            if do_test and dataset.Xs_test and inter.Xs_test_inter and integ.Xs_test_integ:
                orig_test_3d, orig_test_limits = project_3d_list(dataset.Xs_test)
                inter_test_3d, inter_test_limits = project_3d_list(inter.Xs_test_inter)
                integ_test_3d, integ_test_limits = project_3d_list(integ.Xs_test_integ)

            # Integrated だけは全機関で共有するスケールに合わせる
            def compute_shared_limits(arr_list_3d):
                valid = [a for a in arr_list_3d if a is not None]
                if not valid:
                    return None
                stacked = np.vstack(valid)
                limits = tuple(
                    (stacked[:, idx].min() - 0.05 * (stacked[:, idx].ptp() or 1.0),
                     stacked[:, idx].max() + 0.05 * (stacked[:, idx].ptp() or 1.0))
                    for idx in range(3)
                )
                return limits

            integ_train_3d_limits = compute_shared_limits(integ_train_3d) if integ_train_3d is not None else None
            if integ_train_3d_limits is None:
                integ_train_3d_limits = integ_train_limits

            integ_test_3d_limits = compute_shared_limits(integ_test_3d) if integ_test_3d is not None else None
            if integ_test_3d_limits is None:
                integ_test_3d_limits = integ_test_limits

            def to_3d(ax):
                fig_ = ax.figure
                spec = ax.get_subplotspec()
                ax.remove()
                return fig_.add_subplot(spec, projection="3d")

        def add_row_label(ax, text: str):
            if getattr(ax, "name", "") == "3d":
                ax.text2D(-0.12, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=20, fontweight="bold", rotation=90, fontproperties=jp_font)
            else:
                ax.text(-0.12, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=20, fontweight="bold", rotation=90, fontproperties=jp_font)

        def get_jp_font():
            from matplotlib import font_manager
            preferred = [
                "IPAexGothic", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
                "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic",
                "Yu Gothic", "MS Gothic",
            ]
            for name in preferred:
                try:
                    path = font_manager.findfont(name, fallback_to_default=False)
                    if path:
                        return font_manager.FontProperties(fname=path)
                except Exception:
                    continue
            return None

        jp_font = get_jp_font()
        def draw_edges_2d(ax, data2d, edges):
            if data2d is None or not edges:
                return
            data2d = np.asarray(data2d)
            for u, v in edges:
                if u >= data2d.shape[0] or v >= data2d.shape[0]:
                    continue
                ax.plot([data2d[u, 0], data2d[v, 0]], [data2d[u, 1], data2d[v, 1]], color="gray", alpha=0.6, linewidth=1.0)

        def draw_edges_3d(ax, data3d, edges):
            if data3d is None or not edges:
                return
            data3d = np.asarray(data3d)
            for u, v in edges:
                if u >= data3d.shape[0] or v >= data3d.shape[0]:
                    continue
                ax.plot(
                    [data3d[u, 0], data3d[v, 0]],
                    [data3d[u, 1], data3d[v, 1]],
                    [data3d[u, 2], data3d[v, 2]],
                    color="gray", alpha=0.6, linewidth=1.0,
                )

        mst_edges_train = []
        mst_edges_test = []
        if use_graph and do_train:
            mst_edges_train = [self._compute_mst_edges(np.asarray(arr)) for arr in dataset.Xs_train]
            if dataset.Xs_test:
                mst_edges_test = [self._compute_mst_edges(np.asarray(arr)) for arr in dataset.Xs_test]
            else:
                mst_edges_test = [[] for _ in range(num_institutions)]
        def draw_edges_2d(ax, data2d, edges):
            if data2d is None or not edges:
                return
            data2d = np.asarray(data2d)
            for u, v in edges:
                if u >= data2d.shape[0] or v >= data2d.shape[0]:
                    continue
                ax.plot([data2d[u, 0], data2d[v, 0]], [data2d[u, 1], data2d[v, 1]], color="gray", alpha=0.6, linewidth=1.0)

        def draw_edges_3d(ax, data3d, edges):
            if data3d is None or not edges:
                return
            data3d = np.asarray(data3d)
            for u, v in edges:
                if u >= data3d.shape[0] or v >= data3d.shape[0]:
                    continue
                ax.plot(
                    [data3d[u, 0], data3d[v, 0]],
                    [data3d[u, 1], data3d[v, 1]],
                    [data3d[u, 2], data3d[v, 2]],
                    color="gray", alpha=0.6, linewidth=1.0,
                )

        if do_train:
            fig_train, axes_train = plt.subplots(num_institutions, 3, figsize=(18, 5 * num_institutions), squeeze=False)
            col_titles = ["機関データ", "中間表現", "統合表現"]
            for ci, title in enumerate(col_titles):
                xpos = (ci + 0.5) / 3.0
                fig_train.text(xpos, 0.98, title, ha="center", va="top", fontsize=20, fontweight="bold", fontproperties=jp_font)

            for idx in range(num_institutions):
                orig = self._ensure_2d(dataset.Xs_train[idx])
                ax_orig = axes_train[idx, 0]
                if enable_3d and orig_train_3d and orig_train_3d[idx] is not None:
                    ax_orig = to_3d(ax_orig)
                    axes_train[idx, 0] = ax_orig
                    d3o = orig_train_3d[idx]
                    ax_orig.scatter(d3o[:, 0], d3o[:, 1], d3o[:, 2], c=dataset.ys_train[idx], cmap="viridis", s=train_point_size_3d, depthshade=True)
                    if orig_train_limits:
                        ax_orig.set_xlim(orig_train_limits[0]); ax_orig.set_ylim(orig_train_limits[1]); ax_orig.set_zlim(orig_train_limits[2])
                    if use_graph:
                        draw_edges_3d(ax_orig, d3o, mst_edges_train[idx] if idx < len(mst_edges_train) else [])
                else:
                    sns.scatterplot(x=orig[:, 0], y=orig[:, 1], hue=dataset.ys_train[idx], palette="viridis", ax=ax_orig, legend=False, s=train_point_size_2d)
                    if use_graph:
                        draw_edges_2d(ax_orig, orig, mst_edges_train[idx] if idx < len(mst_edges_train) else [])
                add_row_label(ax_orig, f"機関 {idx+1}")

                inter_data = self._ensure_2d(inter.Xs_train_inter[idx])
                ax_inter = axes_train[idx, 1]
                if enable_3d and inter_train_3d and inter_train_3d[idx] is not None:
                    ax_inter = to_3d(ax_inter)
                    axes_train[idx, 1] = ax_inter
                    d3i = inter_train_3d[idx]
                    ax_inter.scatter(d3i[:, 0], d3i[:, 1], d3i[:, 2], c=dataset.ys_train[idx], cmap="viridis", s=train_point_size_3d, depthshade=True)
                    if inter_train_limits:
                        ax_inter.set_xlim(inter_train_limits[0]); ax_inter.set_ylim(inter_train_limits[1]); ax_inter.set_zlim(inter_train_limits[2])
                    if use_graph:
                        draw_edges_3d(ax_inter, d3i, mst_edges_train[idx] if idx < len(mst_edges_train) else [])
                else:
                    sns.scatterplot(x=inter_data[:, 0], y=inter_data[:, 1], hue=dataset.ys_train[idx], palette="viridis", ax=ax_inter, legend=False, s=train_point_size_2d)
                    if use_graph:
                        draw_edges_2d(ax_inter, inter_data, mst_edges_train[idx] if idx < len(mst_edges_train) else [])

                integ_data = self._ensure_2d(integ.Xs_train_integ[idx])
                ax_integ = axes_train[idx, 2]
                if enable_3d and integ_train_3d and integ_train_3d[idx] is not None:
                    ax_integ = to_3d(ax_integ)
                    axes_train[idx, 2] = ax_integ
                    d3t = integ_train_3d[idx]
                    ax_integ.scatter(d3t[:, 0], d3t[:, 1], d3t[:, 2], c=integ.ys_train_integ[idx], cmap="viridis", s=train_point_size_3d, depthshade=True)
                    limits_use = integ_train_3d_limits or integ_train_limits
                    if limits_use:
                        ax_integ.set_xlim(limits_use[0]); ax_integ.set_ylim(limits_use[1]); ax_integ.set_zlim(limits_use[2])
                    if use_graph:
                        draw_edges_3d(ax_integ, d3t, mst_edges_train[idx] if idx < len(mst_edges_train) else [])
                else:
                    sns.scatterplot(x=integ_data[:, 0], y=integ_data[:, 1], hue=integ.ys_train_integ[idx], palette="viridis", ax=ax_integ, legend=False, s=train_point_size_2d)
                    if xlim_train is not None and ylim_train is not None:
                        ax_integ.set_xlim(xlim_train); ax_integ.set_ylim(ylim_train)
                    if use_graph:
                        draw_edges_2d(ax_integ, integ_data, mst_edges_train[idx] if idx < len(mst_edges_train) else [])

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / self._plot_filename("train_representations"))

        if do_test and dataset.Xs_test and inter.Xs_test_inter and integ.Xs_test_integ and xlim_test:
            fig_test, axes_test = plt.subplots(num_institutions, 3, figsize=(18, 5 * num_institutions), squeeze=False)
            col_titles = ["機関データ", "中間表現", "統合表現"]
            for ci, title in enumerate(col_titles):
                xpos = (ci + 0.5) / 3.0
                fig_test.text(xpos, 0.98, title, ha="center", va="top", fontsize=20, fontweight="bold", fontproperties=jp_font)

            for idx in range(num_institutions):
                orig = self._ensure_2d(dataset.Xs_test[idx])
                ax_orig_t = axes_test[idx, 0]
                if enable_3d and orig_test_3d and orig_test_3d[idx] is not None:
                    ax_orig_t = to_3d(ax_orig_t)
                    axes_test[idx, 0] = ax_orig_t
                    d3ot = orig_test_3d[idx]
                    ax_orig_t.scatter(d3ot[:, 0], d3ot[:, 1], d3ot[:, 2], c=dataset.ys_test[idx], cmap="copper", s=test_point_size_3d, depthshade=True)
                    if orig_test_limits:
                        ax_orig_t.set_xlim(orig_test_limits[0]); ax_orig_t.set_ylim(orig_test_limits[1]); ax_orig_t.set_zlim(orig_test_limits[2])
                    if use_graph:
                        draw_edges_3d(ax_orig_t, d3ot, mst_edges_test[idx] if idx < len(mst_edges_test) else [])
                else:
                    sns.scatterplot(x=orig[:, 0], y=orig[:, 1], hue=dataset.ys_test[idx], palette="copper", ax=ax_orig_t, legend=False, s=test_point_size_2d)
                    if use_graph:
                        draw_edges_2d(ax_orig_t, orig, mst_edges_test[idx] if idx < len(mst_edges_test) else [])
                add_row_label(ax_orig_t, f"機関 {idx+1}")

                inter_data = self._ensure_2d(inter.Xs_test_inter[idx])
                ax_inter_t = axes_test[idx, 1]
                if enable_3d and inter_test_3d and inter_test_3d[idx] is not None:
                    ax_inter_t = to_3d(ax_inter_t)
                    axes_test[idx, 1] = ax_inter_t
                    d3it = inter_test_3d[idx]
                    ax_inter_t.scatter(d3it[:, 0], d3it[:, 1], d3it[:, 2], c=dataset.ys_test[idx], cmap="copper", s=test_point_size_3d, depthshade=True)
                    if inter_test_limits:
                        ax_inter_t.set_xlim(inter_test_limits[0]); ax_inter_t.set_ylim(inter_test_limits[1]); ax_inter_t.set_zlim(inter_test_limits[2])
                    if use_graph:
                        draw_edges_3d(ax_inter_t, d3it, mst_edges_test[idx] if idx < len(mst_edges_test) else [])
                else:
                    sns.scatterplot(x=inter_data[:, 0], y=inter_data[:, 1], hue=dataset.ys_test[idx], palette="copper", ax=ax_inter_t, legend=False, s=test_point_size_2d)
                    if use_graph:
                        draw_edges_2d(ax_inter_t, inter_data, mst_edges_test[idx] if idx < len(mst_edges_test) else [])

                integ_data = self._ensure_2d(integ.Xs_test_integ[idx])
                ax_integ_t = axes_test[idx, 2]
                if enable_3d and integ_test_3d and integ_test_3d[idx] is not None:
                    ax_integ_t = to_3d(ax_integ_t)
                    axes_test[idx, 2] = ax_integ_t
                    d3tt = integ_test_3d[idx]
                    ax_integ_t.scatter(d3tt[:, 0], d3tt[:, 1], d3tt[:, 2], c=integ.ys_test_integ[idx], cmap="copper", s=test_point_size_3d, depthshade=True)
                    limits_use = integ_test_3d_limits or integ_test_limits
                    if limits_use:
                        ax_integ_t.set_xlim(limits_use[0]); ax_integ_t.set_ylim(limits_use[1]); ax_integ_t.set_zlim(limits_use[2])
                    if use_graph:
                        draw_edges_3d(ax_integ_t, d3tt, mst_edges_test[idx] if idx < len(mst_edges_test) else [])
                else:
                    sns.scatterplot(x=integ_data[:, 0], y=integ_data[:, 1], hue=integ.ys_test_integ[idx], palette="copper", ax=ax_integ_t, legend=False, s=test_point_size_2d)
                    ax_integ_t.set_xlim(xlim_test); ax_integ_t.set_ylim(ylim_test)
                    if use_graph:
                        draw_edges_2d(ax_integ_t, integ_data, mst_edges_test[idx] if idx < len(mst_edges_test) else [])

            plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
            plt.savefig(save_dir / self._plot_filename("test_representations"))

        if do_anchor:
            self.visualize_anchors(save_dir=save_dir)
            if getattr(cfg, "visualize_for_presenations", False):
                self.visualize_anchors_for_presenations(save_dir=save_dir)

    # ------------------------------------------------------------------ #
    def _stack_arrays(self, arrays: Sequence[np.ndarray]) -> np.ndarray | None:
        if not arrays:
            return None
        valid = [np.asarray(arr) for arr in arrays if arr is not None and arr.size]
        if not valid:
            return None
        return np.vstack(valid)

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)
        if arr.shape[1] >= 2:
            return arr[:, :2]
        zeros = np.zeros((arr.shape[0], 1))
        return np.hstack([arr, zeros])

    def _compute_limits(self, data: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        data2d = self._ensure_2d(data)
        xmin, xmax = data2d[:, 0].min(), data2d[:, 0].max()
        ymin, ymax = data2d[:, 1].min(), data2d[:, 1].max()
        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
        ypad = (ymax - ymin) * 0.05 if ymax > ymin else 0.1
        return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)

    def _plot_filename(self, suffix: str) -> str:
        base = getattr(self.config, "plot_name", None)
        if not base:
            base = f"{self.config.name or 'data_collab'}_plot.png"
        stem = Path(str(base)).stem
        return f"{stem}_{suffix}.png"

    def _get_jp_font(self):
        try:
            from matplotlib import font_manager
        except Exception:
            return None
        preferred = [
            "IPAexGothic", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
            "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic",
            "Yu Gothic", "MS Gothic",
        ]
        for name in preferred:
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                if path:
                    return font_manager.FontProperties(fname=path)
            except Exception:
                continue
        return None

    def _compute_mst_edges(self, X: np.ndarray) -> list[tuple[int, int]]:
        if X is None or X.size == 0:
            return []
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            from scipy.spatial.distance import pdist, squareform
        except Exception:
            return []
        if X.shape[0] <= 1:
            return []
        dist = squareform(pdist(X, metric="euclidean"))
        mst = minimum_spanning_tree(dist)
        coo = mst.tocoo()
        return list(zip(coo.row.tolist(), coo.col.tolist()))

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                pass
