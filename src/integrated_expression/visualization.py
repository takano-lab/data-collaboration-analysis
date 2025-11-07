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
        import matplotlib.pyplot as plt
        import seaborn as sns
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from sklearn.decomposition import PCA

        inter = self.intermediate
        integ = self.artifacts
        config = self.config

        save_dir = Path(save_dir or (config.output_path / "visualizations"))

        has_train_data = bool(inter.anchor.size and inter.anchors_inter)
        has_test_data = bool(inter.anchor_test.size and inter.anchors_test_inter)

        if not has_train_data and not has_test_data:
            self._log("Anchor visualization skipped: no anchor data available.")
            return

        if has_train_data:
            num_institutions = len(inter.anchors_inter)
        else:
            num_institutions = len(inter.anchors_test_inter)
        if num_institutions == 0:
            return

        anchor_labels_train = inter.anchor_y if inter.anchor_y.size else np.zeros(inter.anchor.shape[0])
        anchor_labels_test = inter.anchor_y_test if inter.anchor_y_test.size else np.zeros(inter.anchor_test.shape[0])
        legend_status = "full" if anchor_labels_train.size and np.unique(anchor_labels_train).size > 1 else False

        Z_train_plot = integ.Z_integ
        if Z_train_plot is not None and Z_train_plot.ndim == 2 and Z_train_plot.shape[0] == config.dim_integrate:
            Z_train_plot = Z_train_plot.T

        col1_data = ([inter.anchor] if has_train_data else []) + ([inter.anchor_test] if has_test_data else [])
        col2_data = (inter.anchors_inter if has_train_data else []) + (inter.anchors_test_inter if has_test_data else [])
        col3_data = (integ.anchors_integ if has_train_data else []) + (integ.anchors_test_integ if has_test_data else [])
        col4_data = [Z_train_plot] if has_train_data and Z_train_plot is not None else []

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

        col1_2d, (xlim1, ylim1) = ensure_2d(col1_data)
        col2_2d, (xlim2, ylim2) = ensure_2d(col2_data)
        col3_2d, (xlim3, ylim3) = ensure_2d(col3_data)
        col4_2d, (xlim4, ylim4) = ensure_2d(col4_data)

        fig, axes = plt.subplots(num_institutions * 2, 4, figsize=(24, 6 * num_institutions * 2), squeeze=False)
        fig.suptitle("Anchor Data Transformation Flow (Top: Train, Bottom: Test)", fontsize=16, y=0.995)

        def to_3d(ax):
            fig_ = ax.figure
            spec = ax.get_subplotspec()
            ax.remove()
            return fig_.add_subplot(spec, projection="3d")

        def project_3d(data_list: Sequence[np.ndarray]):
            valid = [d for d in data_list if d is not None and d.ndim == 2 and d.shape[1] > 3]
            if not valid:
                return [None for _ in data_list], None
            pca = PCA(n_components=3).fit(np.vstack(valid))
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

        col3_train_3d, train3d_limits = project_3d(integ.anchors_integ if has_train_data else [])
        col3_test_3d, test3d_limits = project_3d(integ.anchors_test_integ if has_test_data else [])
        Z_train_3d, Z_limits = project_3d([Z_train_plot] if Z_train_plot is not None else [])
        train_offset = len(integ.anchors_integ) if has_train_data else 0

        for i in range(num_institutions):
            train_row = i
            if has_train_data:
                sns.scatterplot(
                    x=col1_2d[0][:, 0], y=col1_2d[0][:, 1],
                    hue=anchor_labels_train, palette="coolwarm",
                    ax=axes[train_row, 0], legend=(i == 0 and legend_status),
                )
                axes[train_row, 0].set_title("1. Original Anchor (Train)" if i == 0 else "")
                axes[train_row, 0].set_xlim(xlim1); axes[train_row, 0].set_ylim(ylim1)
                axes[train_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(
                    x=col2_2d[i][:, 0], y=col2_2d[i][:, 1],
                    hue=anchor_labels_train, palette="coolwarm",
                    ax=axes[train_row, 1], legend=False,
                )
                axes[train_row, 1].set_title("2. Intermediate (Train)" if i == 0 else "")
                axes[train_row, 1].set_xlim(xlim2); axes[train_row, 1].set_ylim(ylim2)

                if col3_train_3d and col3_train_3d[i] is not None:
                    ax3d = to_3d(axes[train_row, 2])
                    d3 = col3_train_3d[i]
                    ax3d.scatter(d3[:, 0], d3[:, 1], d3[:, 2], c=anchor_labels_train, cmap="coolwarm", s=14, depthshade=True)
                    if train3d_limits:
                        ax3d.set_xlim(train3d_limits[0]); ax3d.set_ylim(train3d_limits[1]); ax3d.set_zlim(train3d_limits[2])
                    ax3d.set_title("3. Projection S_hat (Train)" if i == 0 else "")
                else:
                    sns.scatterplot(
                        x=col3_2d[i][:, 0], y=col3_2d[i][:, 1],
                        hue=anchor_labels_train, palette="coolwarm",
                        ax=axes[train_row, 2], legend=False,
                    )
                    axes[train_row, 2].set_title("3. Projection S_hat (Train)" if i == 0 else "")
                    axes[train_row, 2].set_xlim(xlim3); axes[train_row, 2].set_ylim(ylim3)

                if Z_train_3d and Z_train_3d[0] is not None:
                    ax3dz = to_3d(axes[train_row, 3])
                    d3z = Z_train_3d[0]
                    ax3dz.scatter(d3z[:, 0], d3z[:, 1], d3z[:, 2], c=anchor_labels_train, cmap="coolwarm", s=14, depthshade=True)
                    if Z_limits:
                        ax3dz.set_xlim(Z_limits[0]); ax3dz.set_ylim(Z_limits[1]); ax3dz.set_zlim(Z_limits[2])
                    ax3dz.set_title("4. Integrated Z (Train)" if i == 0 else "")
                elif col4_2d:
                    sns.scatterplot(
                        x=col4_2d[0][:, 0], y=col4_2d[0][:, 1],
                        hue=anchor_labels_train, palette="coolwarm",
                        ax=axes[train_row, 3], legend=False,
                    )
                    axes[train_row, 3].set_xlim(xlim4); axes[train_row, 3].set_ylim(ylim4)
                    axes[train_row, 3].set_title("4. Integrated Z (Train)" if i == 0 else "")
                else:
                    axes[train_row, 3].set_visible(False)

            if has_test_data:
                test_row = i + num_institutions
                test_idx = (0 if not has_train_data else 1) * num_institutions + i
                sns.scatterplot(
                    x=col1_2d[-1][:, 0], y=col1_2d[-1][:, 1],
                    hue=anchor_labels_test, palette="viridis",
                    ax=axes[test_row, 0], legend=(i == 0 and legend_status),
                )
                axes[test_row, 0].set_title("1. Original Anchor (Test)" if i == 0 else "")
                axes[test_row, 0].set_xlim(xlim1); axes[test_row, 0].set_ylim(ylim1)
                axes[test_row, 0].set_ylabel(f"Inst {i+1}")

                sns.scatterplot(
                    x=col2_2d[test_idx][:, 0], y=col2_2d[test_idx][:, 1],
                    hue=anchor_labels_test, palette="viridis",
                    ax=axes[test_row, 1], legend=False,
                )
                axes[test_row, 1].set_title("2. Intermediate (Test)" if i == 0 else "")
                axes[test_row, 1].set_xlim(xlim2); axes[test_row, 1].set_ylim(ylim2)

                if col3_test_3d and col3_test_3d[i] is not None:
                    ax3dt = to_3d(axes[test_row, 2])
                    d3t = col3_test_3d[i]
                    ax3dt.scatter(d3t[:, 0], d3t[:, 1], d3t[:, 2], c=anchor_labels_test, cmap="viridis", s=14, depthshade=True)
                    if test3d_limits:
                        ax3dt.set_xlim(test3d_limits[0]); ax3dt.set_ylim(test3d_limits[1]); ax3dt.set_zlim(test3d_limits[2])
                    ax3dt.set_title("3. Projection S_hat (Test)" if i == 0 else "")
                else:
                    sns.scatterplot(
                        x=col3_2d[test_idx][:, 0], y=col3_2d[test_idx][:, 1],
                        hue=anchor_labels_test, palette="viridis",
                        ax=axes[test_row, 2], legend=False,
                    )
                    axes[test_row, 2].set_title("3. Projection S_hat (Test)" if i == 0 else "")
                    axes[test_row, 2].set_xlim(xlim3); axes[test_row, 2].set_ylim(ylim3)

                if integ.anchors_test_integ and len(integ.anchors_test_integ) > i:
                    sns.scatterplot(
                        x=col3_2d[test_idx][:, 0], y=col3_2d[test_idx][:, 1],
                        hue=anchor_labels_test, palette="viridis",
                        ax=axes[test_row, 3], legend=False,
                    )
                    axes[test_row, 3].set_xlim(xlim3); axes[test_row, 3].set_ylim(ylim3)
                    axes[test_row, 3].set_title("4. Integrated Z (Test)" if i == 0 else "")
                else:
                    axes[test_row, 3].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / self._plot_filename("anchor_visualization"))

    # ------------------------------------------------------------------ #
    def visualize_representations(self, save_dir: Optional[str] = None) -> None:
        """
        Plot original/intermediate/integrated representations for each institution.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        dataset = self.dataset
        inter = self.intermediate
        integ = self.artifacts

        if not dataset.Xs_train or not inter.Xs_train_inter or not integ.Xs_train_integ:
            self._log("Representation visualization skipped: insufficient training data.")
            return

        save_dir = Path(save_dir or (self.config.output_path / "visualizations"))

        num_institutions = len(dataset.Xs_train)
        train_concat = self._stack_arrays(integ.Xs_train_integ)
        if train_concat is None:
            self._log("Representation visualization skipped: integrated train embeddings are empty.")
            return
        xlim_train, ylim_train = self._compute_limits(train_concat)

        test_concat = self._stack_arrays(integ.Xs_test_integ)
        if test_concat is not None:
            xlim_test, ylim_test = self._compute_limits(test_concat)
        else:
            xlim_test = ylim_test = None

        fig_train, axes_train = plt.subplots(num_institutions, 4, figsize=(24, 5 * num_institutions), squeeze=False)
        fig_train.suptitle("Representations (Train Data)", fontsize=16)

        for idx in range(num_institutions):
            orig = self._ensure_2d(dataset.Xs_train[idx])
            sns.scatterplot(x=orig[:, 0], y=orig[:, 1], hue=dataset.ys_train[idx], palette="viridis", ax=axes_train[idx, 0], legend="full")
            axes_train[idx, 0].set_title(f"Institution {idx+1} - Original Data")

            inter_data = self._ensure_2d(inter.Xs_train_inter[idx])
            sns.scatterplot(x=inter_data[:, 0], y=inter_data[:, 1], hue=dataset.ys_train[idx], palette="viridis", ax=axes_train[idx, 1], legend="full")
            axes_train[idx, 1].set_title(f"Institution {idx+1} - Intermediate Expression")

            integ_data = self._ensure_2d(integ.Xs_train_integ[idx])
            sns.scatterplot(x=integ_data[:, 0], y=integ_data[:, 1], hue=integ.ys_train_integ[idx], palette="viridis", ax=axes_train[idx, 2], legend="full")
            axes_train[idx, 2].set_title(f"Institution {idx+1} - Integrated Expression")
            axes_train[idx, 2].set_xlim(xlim_train); axes_train[idx, 2].set_ylim(ylim_train)

            other_indices = [j for j in range(num_institutions) if j != idx]
            if other_indices:
                X_other = self._stack_arrays([integ.Xs_train_integ[j] for j in other_indices])
                y_other = np.hstack([integ.ys_train_integ[j] for j in other_indices])
                other_plot = self._ensure_2d(X_other)
                sns.scatterplot(x=other_plot[:, 0], y=other_plot[:, 1], hue=y_other, palette="viridis", ax=axes_train[idx, 3], legend=False)
            sns.scatterplot(
                x=integ_data[:, 0], y=integ_data[:, 1], hue=integ.ys_train_integ[idx],
                palette="viridis", ax=axes_train[idx, 3], legend="full",
            )
            axes_train[idx, 3].set_title(f"All Institutions (Inst {idx+1} Highlighted)")
            axes_train[idx, 3].set_xlim(xlim_train); axes_train[idx, 3].set_ylim(ylim_train)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / self._plot_filename("train_representations"))

        if dataset.Xs_test and inter.Xs_test_inter and integ.Xs_test_integ and xlim_test:
            fig_test, axes_test = plt.subplots(num_institutions, 3, figsize=(18, 5 * num_institutions), squeeze=False)
            fig_test.suptitle("Representations (Test Data)", fontsize=16)
            for idx in range(num_institutions):
                orig = self._ensure_2d(dataset.Xs_test[idx])
                sns.scatterplot(x=orig[:, 0], y=orig[:, 1], hue=dataset.ys_test[idx], palette="viridis", ax=axes_test[idx, 0], legend="full")
                axes_test[idx, 0].set_title(f"Institution {idx+1} - Original Test Data")

                inter_data = self._ensure_2d(inter.Xs_test_inter[idx])
                sns.scatterplot(x=inter_data[:, 0], y=inter_data[:, 1], hue=dataset.ys_test[idx], palette="viridis", ax=axes_test[idx, 1], legend="full")
                axes_test[idx, 1].set_title(f"Institution {idx+1} - Intermediate Test")

                integ_data = self._ensure_2d(integ.Xs_test_integ[idx])
                sns.scatterplot(x=integ_data[:, 0], y=integ_data[:, 1], hue=integ.ys_test_integ[idx], palette="viridis", ax=axes_test[idx, 2], legend="full")
                axes_test[idx, 2].set_title(f"Institution {idx+1} - Integrated Test")
                axes_test[idx, 2].set_xlim(xlim_test); axes_test[idx, 2].set_ylim(ylim_test)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(save_dir / self._plot_filename("test_representations"))

        # Optionally chain anchor visualization for convenience.
        self.visualize_anchors(save_dir=save_dir)

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

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                pass
