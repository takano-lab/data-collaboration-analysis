from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .builder import IntegratedRepresentationBuilder


def integrate_metrics(builder: "IntegratedRepresentationBuilder") -> dict:
    """Compute pairwise anchor distances in the integrated space."""

    def _standardize(X: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0:
            return X
        mu = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=0)
        std_safe = np.where(std > 0, std, 1.0)
        Xz = (X - mu) / std_safe
        zero_var_cols = std == 0
        if np.any(zero_var_cols):
            Xz[:, zero_var_cols] = 0.0
        return Xz

    def _standardize_with_params(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        if X is None or X.size == 0:
            return X
        std_safe = np.where(std > 0, std, 1.0)
        Xz = (X - mu) / std_safe
        zero_var_cols = std == 0
        if np.any(zero_var_cols):
            Xz[:, zero_var_cols] = 0.0
        return Xz

    def _compute_metrics(anchors_std_list: list[np.ndarray]) -> dict:
        if not anchors_std_list or len(anchors_std_list) < 2:
            builder._log("integrate_metrics: insufficient institutions.", level="warning")
            return {"pairs": [], "summary": {}}

        results = []
        for i, j in combinations(range(len(anchors_std_list)), 2):
            Ai = anchors_std_list[i]
            Aj = anchors_std_list[j]
            if Ai is None or Aj is None or Ai.size == 0 or Aj.size == 0:
                builder._log(f"integrate_metrics: skip invalid pair (i={i}, j={j})", level="warning")
                continue

            n = min(Ai.shape[0], Aj.shape[0])
            if Ai.shape != Aj.shape:
                builder._log(
                    f"integrate_metrics: shape mismatch i={i}{Ai.shape}, j={j}{Aj.shape} -> using first {n} rows",
                    level="warning",
                )
            dmin = min(Ai.shape[1], Aj.shape[1])
            Di = Ai[:n, :dmin] - Aj[:n, :dmin]
            row_dists = np.linalg.norm(Di, axis=1)
            results.append(
                {
                    "i": i,
                    "j": j,
                    "mean": float(row_dists.mean()),
                    "std": float(row_dists.std(ddof=0)),
                    "n_rows_used": int(n),
                    "dim_used": int(dmin),
                }
            )

        if not results:
            return {"pairs": [], "summary": {}}

        pair_means = np.array([r["mean"] for r in results], dtype=float)
        summary = {
            "pair_count": int(len(results)),
            "mean_of_means": float(pair_means.mean()),
            "std_of_means": float(pair_means.std(ddof=0)),
            "min_mean": float(pair_means.min()),
            "max_mean": float(pair_means.max()),
        }
        return {"pairs": results, "summary": summary}

    # train metrics
    train_list = builder.anchors_integ
    if not train_list or len(train_list) < 2:
        builder._log("integrate_metrics: missing train anchors.", level="warning")
        metrics_train = {"pairs": [], "summary": {}}
        builder.config.integ_metrics_train = 100000
    else:
        mus_stds = []
        anchors_train_std = []
        for Ak in train_list:
            mu = np.nanmean(Ak, axis=0)
            std = np.nanstd(Ak, axis=0, ddof=0)
            anchors_train_std.append(_standardize_with_params(Ak, mu, std))
            mus_stds.append((mu, std))
        metrics_train = _compute_metrics(anchors_train_std)
        if metrics_train.get("summary"):
            val = float(metrics_train["summary"]["mean_of_means"])
            builder.config.integ_metrics_train = round(val, 5)
        else:
            builder.config.integ_metrics_train = 100000

    # test metrics (standardize using train params if available)
    test_list = builder.anchors_test_integ
    if not test_list or len(test_list) < 2:
        builder._log("integrate_metrics: missing test anchors.", level="warning")
        metrics_test = {"pairs": [], "summary": {}}
        builder.config.integ_metrics_test = 100000
    else:
        if "mus_stds" not in locals() or len(mus_stds) != len(test_list):
            anchors_test_std = [_standardize(Ak) for Ak in test_list]
        else:
            anchors_test_std = [
                _standardize_with_params(Ak_test, mu_std[0], mu_std[1])
                for Ak_test, mu_std in zip(test_list, mus_stds)
            ]
        metrics_test = _compute_metrics(anchors_test_std)
        if metrics_test.get("summary"):
            val = float(metrics_test["summary"]["mean_of_means"])
            builder.config.integ_metrics_test = round(val, 5)
        else:
            builder.config.integ_metrics_test = 100000

    if metrics_train.get("summary"):
        s = metrics_train["summary"]
        builder._log(
            f"[integrate_metrics/train] pairs={s['pair_count']}, "
            f"mean_of_means={s['mean_of_means']:.6g}, std_of_means={s['std_of_means']:.6g}, "
            f"min_mean={s['min_mean']:.6g}, max_mean={s['max_mean']:.6g}",
            level="info",
        )
    if metrics_test.get("summary"):
        s = metrics_test["summary"]
        builder._log(
            f"[integrate_metrics/test] pairs={s['pair_count']}, "
            f"mean_of_means={s['mean_of_means']:.6g}, std_of_means={s['std_of_means']:.6g}, "
            f"min_mean={s['min_mean']:.6g}, max_mean={s['max_mean']:.6g}",
            level="info",
        )

    return {"train": metrics_train, "test": metrics_test}


def evaluate_nonlinearity_indices(builder: "IntegratedRepresentationBuilder") -> dict:
    import traceback

    def _lni_from_pair(X: np.ndarray, Z: np.ndarray) -> float:
        if X is None or Z is None or X.size == 0 or Z.size == 0:
            return np.nan
        n = min(X.shape[0], Z.shape[0])
        if n <= 1:
            return np.nan
        Xn = np.asarray(X[:n, :], dtype=float)
        Zn = np.asarray(Z[:n, :], dtype=float)
        ones = np.ones((n, 1), dtype=float)
        X_aug = np.hstack([Xn, ones])
        try:
            W, *_ = np.linalg.lstsq(X_aug, Zn, rcond=None)
            Z_hat = X_aug @ W
        except Exception as ex:
            builder._log(f"[LNI] lstsq failed: {ex} | X_aug={X_aug.shape}, Z={Zn.shape}", level="error")
            traceback.print_exc()
            return np.nan
        diff = Zn - Z_hat
        rss = float(np.linalg.norm(diff, ord="fro") ** 2)
        Zbar = Zn.mean(axis=0, keepdims=True)
        tss = float(np.linalg.norm(Zn - Zbar, ord="fro") ** 2)
        if tss <= 1e-12:
            return 0.0
        lni = rss / tss
        return float(np.clip(lni, 0.0, 1.0)) if np.isfinite(lni) else np.nan

    def _mean_finite(vals: list[float]) -> float:
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(arr.mean())

    def _compute_lni(Z_true: np.ndarray, Z_pred: np.ndarray) -> float:
        diff = Z_true - Z_pred
        rss = float(np.linalg.norm(diff, ord="fro") ** 2)
        Zbar = Z_true.mean(axis=0, keepdims=True)
        tss = float(np.linalg.norm(Z_true - Zbar, ord="fro") ** 2)
        if tss <= 1e-12:
            return 0.0
        lni = rss / tss
        return float(np.clip(lni, 0.0, 1.0)) if np.isfinite(lni) else np.nan

    def _fit_lni_model(X: Optional[np.ndarray], Z: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], float]:
        if X is None or Z is None or X.size == 0 or Z.size == 0:
            return None, np.nan
        n = min(X.shape[0], Z.shape[0])
        if n <= 1:
            return None, np.nan
        Xn = np.asarray(X[:n, :], dtype=float)
        Zn = np.asarray(Z[:n, :], dtype=float)
        ones = np.ones((n, 1), dtype=float)
        X_aug = np.hstack([Xn, ones])
        try:
            W, *_ = np.linalg.lstsq(X_aug, Zn, rcond=None)
            Z_hat = X_aug @ W
        except Exception as ex:
            builder._log(f"[LNI] lstsq failed: {ex} | X_aug={X_aug.shape}, Z={Zn.shape}", level="error")
            traceback.print_exc()
            return None, np.nan
        return W, _compute_lni(Zn, Z_hat)

    def _eval_lni_with_model(
        X: Optional[np.ndarray],
        Z: Optional[np.ndarray],
        W: Optional[np.ndarray],
    ) -> float:
        if W is None or X is None or Z is None or X.size == 0 or Z.size == 0:
            return np.nan
        n = min(X.shape[0], Z.shape[0])
        if n <= 1:
            return np.nan
        Xn = np.asarray(X[:n, :], dtype=float)
        Zn = np.asarray(Z[:n, :], dtype=float)
        ones = np.ones((n, 1), dtype=float)
        X_aug = np.hstack([Xn, ones])
        if X_aug.shape[1] != W.shape[0]:
            return np.nan
        try:
            Z_hat = X_aug @ W
        except Exception as ex:
            builder._log(f"[LNI] evaluation failed: {ex} | X_aug={X_aug.shape}, W={W.shape}", level="error")
            traceback.print_exc()
            return np.nan
        return _compute_lni(Zn, Z_hat)

    def _fit_and_score_pairs(pairs: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]]):
        models = []
        scores = []
        for X, Z in pairs:
            W, lni = _fit_lni_model(X, Z)
            models.append(W)
            scores.append(lni)
        return models, scores

    def _score_pairs_with_models(
        pairs: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        models: list[Optional[np.ndarray]],
    ) -> list[float]:
        results: list[float] = []
        for idx in range(max(len(pairs), len(models))):
            X, Z = (pairs[idx] if idx < len(pairs) else (None, None))
            W = models[idx] if idx < len(models) else None
            results.append(_eval_lni_with_model(X, Z, W))
        return results

    pairs_inter = [(builder.anchor, Ak) for Ak in (builder.anchors_inter or [])]
    pairs_integ = list(zip(builder.anchors_inter or [], builder.anchors_integ or []))
    pairs_inter_test = [(builder.anchor_test, Ak) for Ak in (builder.anchors_test_inter or [])]
    pairs_integ_test = list(zip(builder.anchors_test_inter or [], builder.anchors_test_integ or []))

    models_inter, list_inter = _fit_and_score_pairs(pairs_inter)
    models_integ, list_integ = _fit_and_score_pairs(pairs_integ)
    list_inter_test = _score_pairs_with_models(pairs_inter_test, models_inter)
    list_integ_test = _score_pairs_with_models(pairs_integ_test, models_integ)

    def _fmt_list(vs):
        def _fmt(x):
            return "nan" if (x is None or not np.isfinite(x)) else f"{float(x):.4f}"

        return "[" + ", ".join(_fmt(x) for x in vs) + "]"

    try:
        builder._log(f"[LNI] inter: {_fmt_list(list_inter)}")
        builder._log(f"[LNI] integ: {_fmt_list(list_integ)}")
        builder._log(f"[LNI] inter_test: {_fmt_list(list_inter_test)}")
        builder._log(f"[LNI] integ_test: {_fmt_list(list_integ_test)}")
    except Exception:
        pass

    lni_inter = _mean_finite(list_inter)
    lni_integ = _mean_finite(list_integ)
    lni_inter_test = _mean_finite(list_inter_test)
    lni_integ_test = _mean_finite(list_integ_test)

    try:
        if np.isfinite(lni_inter):
            builder.config.lni_inter = round(lni_inter, 4)
        if np.isfinite(lni_inter_test):
            builder.config.lni_inter_test = round(lni_inter_test, 4)
        if np.isfinite(lni_integ):
            builder.config.lni_integ = round(lni_integ, 4)
        if np.isfinite(lni_integ_test):
            builder.config.lni_integ_test = round(lni_integ_test, 4)
    except Exception:
        pass

    result = {
        "inter": lni_inter,
        "integ": lni_integ,
        "inter_test": lni_inter_test,
        "integ_test": lni_integ_test,
    }
    try:
        builder._log(
            {k: (None if (v is None or not np.isfinite(v)) else round(v, 6)) for k, v in result.items()}
        )
    except Exception as ex:
        builder._log(f"[WARN] logging LNI result failed: {ex}", level="warning")
        traceback.print_exc()
    return result


__all__ = ["integrate_metrics", "evaluate_nonlinearity_indices"]
