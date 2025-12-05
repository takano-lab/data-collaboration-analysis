from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.institution_data_pipeline.load_data import LOADERS


def summarize_categoricals(df: pd.DataFrame, cols: list[str], target: str | None = "target"):
    for col in cols:
        print(f"-- {col} --")
        if col not in df.columns:
            print("missing column")
            continue
        s = df[col]
        n = len(s)
        nunique = s.nunique(dropna=True)
        nnull = s.isna().sum()
        print(f"nunique: {nunique}  missing: {nnull}")
        vc = s.value_counts(dropna=False)
        vcn = s.value_counts(normalize=True, dropna=False)
        for k in vc.index.tolist():
            print(f"  {k}: {vc[k]} ({vcn[k]:.3f})")
        if target and target in df.columns:
            ct = pd.crosstab(df[col], df[target], normalize="index")
            print("  per-class proportions (row-normalized):")
            # limit width
            with pd.option_context('display.max_columns', 50):
                print(ct.round(3))
        print()


if __name__ == "__main__":
    df = LOADERS["mice"]()
    summarize_categoricals(df, ["Genotype", "Treatment", "Behavior"], target="target")

