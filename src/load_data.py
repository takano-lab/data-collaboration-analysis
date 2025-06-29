# data_loader.py
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import Config

# -------------------------------------------------- #
# テーブル：データ名 → 読み込みロジック             #
# -------------------------------------------------- #

def _load_statlog() -> pd.DataFrame:
    colnames = [f"col{i}" for i in range(20)] + ["target"]
    df = pd.read_csv("input/statlog_german.data", delim_whitespace=True, header=None, names=colnames)
    return df

def _load_adult() -> pd.DataFrame:
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
            "occupation","relationship","race","sex","capital_gain","capital_loss",
            "hours_per_week","native_country","target"]
    df = pd.read_csv("input/adult.data", names=cols, na_values=" ?", skipinitialspace=True)
    return df

def _load_diabetes130() -> pd.DataFrame:
    df = pd.read_csv("input/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")
    df = df.rename(columns={"readmitted": "target"})
    return df

def _load_credit_default() -> pd.DataFrame:
    df = pd.read_excel("input/credit_default.xls", header=1)
    df = df.rename(columns={"default payment next month": "target"})
    return df

def _load_bank_marketing() -> pd.DataFrame:
    df = pd.read_csv("input/bank-additional/bank-additional-full.csv", sep=";")
    df = df.rename(columns={"y": "target"})
    return df

def _load_har() -> pd.DataFrame:
    # HAR データセットのルートパス
    root = Path("input/UCI_HAR_Dataset")  # WindowsでもOKな相対パス
        
    features = pd.read_csv(root / "features.txt", sep="\s+", header=None)[1].tolist()

    # 名前が重複している列を自動リネーム（例：angle(X,gravityMean) → angle(X,gravityMean).1）
    from collections import Counter
    
    cnt = Counter(features)
    dupes = [name for name, c in cnt.items() if c > 1]
    print(dupes)

    def make_unique(names: list[str]) -> list[str]:
        counter = Counter()
        result = []
        for name in names:
            counter[name] += 1
            if counter[name] == 1:
                result.append(name)
            else:
                result.append(f"{name}.{counter[name]-1}")
        return result

    features = make_unique(features)


    # 各ファイルを読み込み
    def load_split(split: str) -> pd.DataFrame:
        X = pd.read_csv(root / split / f"X_{split}.txt", delim_whitespace=True, header=None, names=features)
        y = pd.read_csv(root / split / f"y_{split}.txt", header=None, names=["activity"])
        subj = pd.read_csv(root / split / f"subject_{split}.txt", header=None, names=["subject"])
        return pd.concat([subj, y, X], axis=1)
    
    df = pd.concat([load_split("train"), load_split("test")], axis=0).reset_index(drop=True)
    df = df.rename(columns={"activity": "target"})
    
    return df

LOADERS = {
    "statlog": _load_statlog,
    "adult": _load_adult,
    "diabetes130": _load_diabetes130,
    "credit_default": _load_credit_default,
    "bank_marketing": _load_bank_marketing,
    "har": _load_har,
}

# -------------------------------------------------- #
# メイン関数                                         #
# -------------------------------------------------- #

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config.dataset not in LOADERS:
        raise ValueError(f"unknown dataset: {config.dataset}")

    df = LOADERS[config.dataset]()

    # ── 目的変数と特徴量を分離
    y = df["target"]
    X = df.drop(columns=["target"])

    # ── one-hot encoding（targetは除く）
    X = pd.get_dummies(X, drop_first=True)
    
    # ── カテゴリ変数は除外
    #X = X.select_dtypes(exclude=['object', 'category'])

    # ── 再結合
    df = pd.concat([X, y], axis=1)
    
    feature_num = len(df.columns) - 1  # 特徴量の数（目的変数を除く）
    config.dim_intermediate = feature_num # 中間表現の次元数
    config.dim_integrate = feature_num # 統合表現の次元数
    
    config.num_institution_user = max(config.dim_integrate, 50) # int(len(df) / (config.num_institution * 2))  # 1機関あたりのユーザ数を計算
    config.num_institution = min(100, int(len(df) / (config.num_institution_user * 2)))
    
    # ── train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        random_state=config.seed,
        shuffle=True,
        stratify=None
    )

    # ── 行数制約でカット（デモ用）
    lim = config.num_institution * config.num_institution_user
    train_df = train_df.iloc[:lim].reset_index(drop=True)
    test_df = test_df.iloc[:lim].reset_index(drop=True)

    # ── 保存
    config.output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.output_path / "train.csv", index=False)
    test_df.to_csv(config.output_path / "test.csv", index=False)

    return train_df, test_df

# -------------------------------------------------- #
# 例: 実行                                           #
# -------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config(name="statlog", output_path=Path("./statlog_split"))
    load_data(cfg)
    print("done")