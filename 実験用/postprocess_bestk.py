from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_best_k(long_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(long_csv)
    required = {"dataset", "repeat", "n_train", "method", "dr", "k", "clf", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    keys_base = ["dataset", "n_train", "clf"]

    # raw mean and k (D)
    raw = df[df["method"] == "raw"].copy()
    raw_grp = raw.groupby(keys_base, as_index=False).agg(
        acc_raw_mean=("acc", "mean"),
        raw_k=("k", lambda s: int(pd.Series(s).iloc[0]) if len(s) else np.nan),
    )

    # linear: PCA only, pick best k by mean acc
    lin = df[(df["method"] == "linear_dr") & (df["dr"] == "PCA")].copy()
    lin_mean = lin.groupby(keys_base + ["k"], as_index=False)["acc"].mean()
    lin_best_idx = lin_mean.groupby(keys_base)["acc"].idxmax()
    lin_best = lin_mean.loc[lin_best_idx].rename(columns={"k": "pca_best_k", "acc": "acc_pca_best_mean"})

    # nonlinear: KernelPCA/Isomap, best k per method, then choose best method
    nl = df[(df["method"] == "nonlinear_dr") & (df["dr"].isin(["KernelPCA", "Isomap"]))].copy()
    nl_mean = nl.groupby(keys_base + ["dr", "k"], as_index=False)["acc"].mean()

    # best k per method
    def best_per_dr(nl_df: pd.DataFrame, dr_name: str, k_col: str, acc_col: str) -> pd.DataFrame:
        sub = nl_df[nl_df["dr"] == dr_name]
        if sub.empty:
            return pd.DataFrame(columns=keys_base + [k_col, acc_col])
        idx = sub.groupby(keys_base)["acc"].idxmax()
        best = sub.loc[idx].rename(columns={"k": k_col, "acc": acc_col})
        return best[keys_base + [k_col, acc_col]]

    kpca_best = best_per_dr(nl_mean, "KernelPCA", "kpca_best_k", "acc_kpca_best_mean")
    isomap_best = best_per_dr(nl_mean, "Isomap", "isomap_best_k", "acc_isomap_best_mean")

    # merge and choose nl best between kpca and isomap
    merged = raw_grp.merge(lin_best, on=keys_base, how="outer")
    merged = merged.merge(kpca_best, on=keys_base, how="outer")
    merged = merged.merge(isomap_best, on=keys_base, how="outer")

    # decide nl best
    a = merged["acc_kpca_best_mean"].fillna(-np.inf)
    b = merged["acc_isomap_best_mean"].fillna(-np.inf)
    nl_best_is_kpca = a >= b
    merged["nl_best_dr"] = np.where(nl_best_is_kpca, "KernelPCA", "Isomap")
    # use pandas nullable integer dtype
    merged["nl_best_k"] = (
        np.where(nl_best_is_kpca, merged["kpca_best_k"], merged["isomap_best_k"]).astype(float)
    )
    merged["nl_best_k"] = pd.Series(merged["nl_best_k"]).astype(pd.Int64Dtype())
    merged["acc_nl_best_mean"] = np.where(nl_best_is_kpca, a, b)
    # mask where neither exists
    neither = merged["acc_kpca_best_mean"].isna() & merged["acc_isomap_best_mean"].isna()
    merged.loc[neither, ["nl_best_dr", "nl_best_k", "acc_nl_best_mean"]] = np.nan

    # order columns
    cols = (
        keys_base
        + ["raw_k", "acc_raw_mean", "pca_best_k", "acc_pca_best_mean",
           "kpca_best_k", "acc_kpca_best_mean", "isomap_best_k", "acc_isomap_best_mean",
           "nl_best_dr", "nl_best_k", "acc_nl_best_mean"]
    )
    merged = merged.reindex(columns=cols)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Postprocess long CSV to report best k per method")
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to long CSV (output of experiment)")
    ap.add_argument("--out", dest="out_csv", required=False, help="Path to write best-k summary CSV")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = in_path.with_name(in_path.stem + "_bestk_summary.csv")

    summarize_best_k(in_path, out_path)
    print(f"Wrote best-k summary to {out_path}")


if __name__ == "__main__":
    main()
