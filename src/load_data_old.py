from __future__ import annotations

from pathlib import Path

import category_encoders as ce
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import Config


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    print(df.head())
    print(len(df))
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=config.seed)

    # data保存
    print(config.output_path)
    # train, testをcsvに書き出し
    train_df = train_df[:config.num_institution * config.num_institution_user]
    test_df = test_df[:config.num_institution * config.num_institution_user]
    train_df.to_csv(config.output_path / "train.csv", index=False)
    test_df.to_csv(config.output_path / "test.csv", index=False)
    
    return train_df, test_df

def load_data_breast_cancer(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]: 
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # データ読み込み
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target  # 目的変数を追加

    print(df.head()) 
    print(len(df)) 

    # データ分割
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=config.seed) 

    # データ保存（必要数だけ切り取って）
    print(1111111111) 
    print(config.output_path) 
    train_df = train_df[:config.num_institution * config.num_institution_user] 
    test_df = test_df[:config.num_institution * config.num_institution_user] 

    train_df.to_csv(config.output_path / "train.csv", index=False) 
    test_df.to_csv(config.output_path / "test.csv", index=False) 

    return train_df, test_df

def load_data_diabetes(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]: 
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # データ読み込み
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target  # 目的変数を追加

    print(df.head()) 
    print(len(df)) 

    # データ分割
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=config.seed) 

    # データ保存（過剰切り詰め防止付き）
    print(1111111111) 
    print(config.output_path) 

    max_train = len(train_df)
    max_test = len(test_df)
    required_rows = config.num_institution * config.num_institution_user

    train_df = train_df[:min(required_rows, max_train)]
    #test_df = test_df[:config.dim_intermediate+1]#min(required_rows, max_test)]
    test_df = test_df[:min(config.num_institution_user, max_test)]
    train_df.to_csv(config.output_path / "train.csv", index=False) 
    test_df.to_csv(config.output_path / "test.csv", index=False) 

    return train_df, test_df

def load_data_har(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    # HAR データセットのルートパス
    root = Path("../input/UCI_HAR_Dataset")  # WindowsでもOKな相対パス
        
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

    # 任意の数だけ取得（行数 = 機関数 × 各機関のユーザ数）
    n = config.num_institution * config.num_institution_user
    df = df.iloc[:2*n]

    # train/test に分割（ラベルは "activity"）
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=config.seed, stratify=df["activity"])

    # 出力確認
    print(f"[INFO] Saving to: {config.output_path}")
    config.output_path.mkdir(parents=True, exist_ok=True)

    # CSV 書き出し
    train_df.to_csv(config.output_path / "train.csv", index=False)
    test_df.to_csv(config.output_path / "test.csv", index=False)

    return train_df, test_df
