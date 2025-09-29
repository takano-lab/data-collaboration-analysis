from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_bestk_from_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "n_train", "clf", "method", "dr", "k", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input long CSV missing columns: {sorted(missing)}")
    keys = ["dataset", "n_train", "clf"]

    # raw: take k (D) and mean acc
    raw = df[df["method"] == "raw"].copy()
    raw_k = raw.groupby(keys)["k"].first().rename("raw_k").reset_index()
    raw_mean = raw.groupby(keys)["acc"].mean().rename("acc_raw_mean").reset_index()
    out = raw_k.merge(raw_mean, on=keys, how="outer")

    # PCA best k by mean acc
    lin = df[(df["method"] == "linear_dr") & (df["dr"] == "PCA")].copy()
    if not lin.empty:
        lin_mean = lin.groupby(keys + ["k"], as_index=False)["acc"].mean()
        idx = lin_mean.groupby(keys)["acc"].idxmax()
        pca_best = lin_mean.loc[idx, keys + ["k", "acc"]].rename(columns={"k": "pca_best_k", "acc": "acc_pca_best_mean"})
        out = out.merge(pca_best, on=keys, how="outer")

    # Nonlinear best per method then best overall
    nl = df[(df["method"] == "nonlinear_dr") & (df["dr"].isin(["KernelPCA", "Isomap"]))].copy()
    if not nl.empty:
        nl_mean = nl.groupby(keys + ["dr", "k"], as_index=False)["acc"].mean()
        def best_per_dr(dr_name: str, kcol: str, acccol: str) -> pd.DataFrame:
            sub = nl_mean[nl_mean["dr"] == dr_name]
            if sub.empty:
                return pd.DataFrame(columns=keys + [kcol, acccol])
            idx = sub.groupby(keys)["acc"].idxmax()
            best = sub.loc[idx, keys + ["k", "acc"]].rename(columns={"k": kcol, "acc": acccol})
            return best
        kpca_best = best_per_dr("KernelPCA", "kpca_best_k", "acc_kpca_best_mean")
        isomap_best = best_per_dr("Isomap", "isomap_best_k", "acc_isomap_best_mean")
        out = out.merge(kpca_best, on=keys, how="outer")
        out = out.merge(isomap_best, on=keys, how="outer")
        a = out["acc_kpca_best_mean"].fillna(-np.inf)
        b = out["acc_isomap_best_mean"].fillna(-np.inf)
        is_kpca = a >= b
        out["nl_best_dr"] = np.where(is_kpca, "KernelPCA", "Isomap")
        out["nl_best_k"] = np.where(is_kpca, out["kpca_best_k"], out["isomap_best_k"])  # keep as float if NaN
        out["acc_nl_best_mean_from_bestk"] = np.where(is_kpca, a, b)
        neither = out["acc_kpca_best_mean"].isna() & out["acc_isomap_best_mean"].isna()
        out.loc[neither, ["nl_best_dr", "nl_best_k", "acc_nl_best_mean_from_bestk"]] = np.nan

    return out


def merge_into_aligned_summary(long_csv: Path, aligned_summary_csv: Path, out_csv: Path) -> pd.DataFrame:
    df_long = pd.read_csv(long_csv)
    df_sum = pd.read_csv(aligned_summary_csv)
    keys = ["dataset", "n_train", "clf"]

    bestk = compute_bestk_from_long(df_long)
    merged = df_sum.merge(bestk, on=keys, how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge best-k info into aligned summary CSV")
    ap.add_argument("--long", required=True, help="Path to long CSV (out of experiment)")
    ap.add_argument("--aligned", required=True, help="Path to aligned summary CSV")
    ap.add_argument("--out", required=False, help="Output CSV path; default aligned stem + _with_bestk.csv")
    args = ap.parse_args()

    long_p = Path(args.long)
    aligned_p = Path(args.aligned)
    out_p = Path(args.out) if args.out else aligned_p.with_name(aligned_p.stem + "_with_bestk.csv")
    merge_into_aligned_summary(long_p, aligned_p, out_p)
    print(f"Wrote merged aligned summary with best-k to {out_p}")


if __name__ == "__main__":
    main()
