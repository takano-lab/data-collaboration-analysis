import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.institution_data_pipeline.load_data import LOADERS


def dataset_stats(name: str):
    try:
        df = LOADERS[name]()
    except Exception as e:
        return {"dataset": name, "error": str(e)}

    # Identify feature columns: drop target and common ID-like columns
    cols = list(df.columns)
    id_like = {"MouseID", "subject", "id", "ID"}
    feature_cols = [c for c in cols if c != "target" and c not in id_like]

    # Keep only numeric features for sparsity computation
    non_numeric_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].to_numpy(copy=False)

    n_samples = len(df)
    n_features = X.shape[1]

    # y stats
    if "target" in df.columns:
        y = df["target"]
        labels = sorted(pd.Series(y).unique().tolist(), key=lambda x: str(x))
        n_labels = len(labels)
        vc = y.value_counts().to_dict()
        vc_norm = (y.value_counts(normalize=True).round(6)).to_dict()
    else:
        labels = []
        n_labels = 0
        vc = {}
        vc_norm = {}

    # sparsity (fraction of exact zeros among numeric features)
    zero_frac = float((X == 0).sum() / X.size) if X.size else 0.0

    return {
        "dataset": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_labels": n_labels,
        "labels": labels,
        "non_numeric_features": non_numeric_cols,
        "sparsity_zero_fraction": round(zero_frac, 6),
        "class_counts": vc,
        "class_proportions": vc_norm,
    }


if __name__ == "__main__":
    targets = ["digits", "mice", "har"]
    results = []
    for t in targets:
        results.append(dataset_stats(t))
    # Pretty print
    for r in results:
        print("==", r.get("dataset"), "==")
        if "error" in r:
            print("error:", r["error"])
            continue
        print("n_samples:", r["n_samples"]) 
        print("n_features:", r["n_features"]) 
        print("n_labels:", r["n_labels"]) 
        print("labels:", r["labels"]) 
        print("non_numeric_features:", r["non_numeric_features"]) 
        print("sparsity_zero_fraction:", r["sparsity_zero_fraction"]) 
        print("class_counts:", r["class_counts"]) 
        print("class_proportions:", r["class_proportions"]) 
        print()
