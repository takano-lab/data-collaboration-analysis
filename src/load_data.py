# data_loader.py
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces, fetch_openml, load_digits, make_moons, make_swiss_roll
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from tdc.single_pred import ADME, HTS, Tox

from config.config import Config

# -------------------------------------------------- #
# テーブル：データ名 → 読み込みロジック             #
# -------------------------------------------------- #

def _load_qsar() -> pd.DataFrame:
    # カラム名（UCI 公式説明より）
    columns = [
        "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]", "NssssC", "nCb-", "C%",
        "nCp", "nO", "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]",
        "Me", "Mi", "nN-N", "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]",
        "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]", "SdO", "TI2_L",
        "nCrt", "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)", "Psi_i_A", "nN",
        "SM6_B(m)", "nArCOOR", "nX", "target"
    ]

    # データ読み込み（区切り文字は ';'）
    df = pd.read_csv("input/qsar+biodegradation/biodeg.csv", header=None, sep=";")
    df.columns = columns

    # ターゲット変換：RB → 1（ready biodeg）、NRB → 0（not ready）
    df["target"] = df["target"].map({"RB": 1, "NRB": 0})

    return df

def _load_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    # データ読み込み
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target  # 目的変数を追加
    return df

def _load_diabetes() -> pd.DataFrame: 
    from sklearn.datasets import load_diabetes

    # データ読み込み
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target  # 目的変数を追加
    return df


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

def _load_digits_df() -> pd.DataFrame:
    """8×8 手書き数字 (n=1 797) を DataFrame 化。"""
    bunch = load_digits(as_frame=True)
    # `bunch.frame` には data と target が入り済み
    df = bunch.frame.copy()
    df = df.rename(columns={"target": "target"})
    df = df.drop(columns=["org"])
    return df


from sklearn.preprocessing import StandardScaler

def _load_concentric_circles_df() -> pd.DataFrame:
    path = Path("input/Three_Organization_Dataset.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"y": "target"})

    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df

def _load_two_gaussian_distributions_df() -> pd.DataFrame:
    path = Path("input/Two_Gaussian_Distributions.csv")
    df = pd.read_csv(path)

    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df

def _load_3D_gaussian_clusters_df() -> pd.DataFrame:
    path = Path("input/3D_8_Gaussian_Clusters.csv")
    df = pd.read_csv(path)

    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df

def load_tdc_dataset(name: str, **kwargs) -> pd.DataFrame:
    """
    指定した TDC データセットを DataFrame で返す。
    返却 DataFrame の教師列を `target` にリネームして統一。

    Parameters
    ----------
    name : str
        データセット名（大文字小文字は公式表記に合わせる）
        - 'AMES'
        - 'Tox21_SR-ARE'
        - 'HIV'
        - 'CYP3A4_Veith'
        - 'CYP2D6_Veith'
        - 'CYP1A2_Veith'
    **kwargs :
        TDC のデータローダにそのまま渡す追加引数
        （例：split を変えたいときに `path='./data2'` など）

    Returns
    -------
    pd.DataFrame
        SMILES などの特徴列と `target` 列を含む表
    """
    # --- データローダの振り分け --------------------------
    if name == "AMES":
        loader = Tox(name="AMES", **kwargs)                          # :contentReference[oaicite:0]{index=0}
    elif name.startswith("Tox21"):
        # name="Tox21_SR-ARE" のように label を一緒に与える
        _, label = name.split("_", 1)
        loader = Tox(name="Tox21", label_name=label, **kwargs)       # :contentReference[oaicite:1]{index=1}
    elif name == "HIV":
        loader = HTS(name="HIV", **kwargs)                           # :contentReference[oaicite:2]{index=2}
    elif name.endswith("_Veith"):
        loader = ADME(name=name, **kwargs)                           # :contentReference[oaicite:3]{index=3}
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    # --- DataFrame を取得し、教師列を統一 -----------------
    df = loader.get_data()                 # （列例：['Drug', 'Y']）
    df = df.rename(columns={"Y": "target"})

    return df

def _load_mnist_df() -> pd.DataFrame:
    """
    OpenML 経由で MNIST データセットを読み込んで DataFrame で返す。
    ピクセル列 + 'target' ラベル列。
    """
    data = fetch_openml("mnist_784", version=1, as_frame=True)
    df = data.frame
    df = df.rename(columns={"class": "target"})  # ラベル列名を 'target' に統一
    return df

def _load_fashion_mnist_df() -> pd.DataFrame:
    """
    OpenML から Fashion-MNIST を読み込み、pandas.DataFrame で返す。
    'target' 列を含み、ピクセル列は 784 次元。
    """
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=True)
    df = data.frame
    df = df.rename(columns={"class": "target"})  # ラベル列を統一
    return df

def _load_mice_df() -> pd.DataFrame:
    """
    Mice Protein Expression (n=1080, 77 特徴量)  
    - UCI ML Repo: https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression
    CSV をあらかじめ `input/mice_protein_expression.csv` に配置して読み込む想定。
    """
    path = Path(r"input\mice+protein+expression\Data_Cortex_Nuclear.xls")
    df = pd.read_excel(path)
    df = df.rename(columns={"class": "target"})
    
       # 'MouseID' は代入に不要なため一時的に除外
    mouse_ids = df['MouseID']
    df_features = df.drop(columns=['MouseID'])
    
    # 数値データのみを対象にK近傍法を適用
    numeric_cols = df_features.select_dtypes(include=np.number).columns
    
    # KNNImputerのインスタンスを作成 (n_neighbors=5がデフォルト)
    imputer = KNNImputer(n_neighbors=5)
    
    # 代入を実行し、結果をDataFrameに戻す
    imputed_data = imputer.fit_transform(df_features[numeric_cols])
    df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols, index=df_features.index)
    
    # 元のDataFrameに代入された値を反映
    df_features[numeric_cols] = df_imputed
    
    # 除外していた 'MouseID' を元に戻す
    df_final = pd.concat([mouse_ids, df_features], axis=1)

    return df_final

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
    "qsar":_load_qsar,
    "breast_cancer":_load_breast_cancer,
    "diabetes":_load_diabetes,
    "statlog": _load_statlog,
    "adult": _load_adult,
    "diabetes130": _load_diabetes130,
    "credit_default": _load_credit_default,
    "bank_marketing": _load_bank_marketing,
    "har": _load_har,
    "digits": _load_digits_df,
    "concentric_circles": _load_concentric_circles_df,
    "two_gaussian_distributions": _load_two_gaussian_distributions_df,
    "3D_gaussian_clusters": _load_3D_gaussian_clusters_df,
    "mice": _load_mice_df,

    # === TDC datasets ===
    "ames": lambda: load_tdc_dataset("AMES"),
    "tox21_sr_are": lambda: load_tdc_dataset("Tox21_SR-ARE"),
    "hiv": lambda: load_tdc_dataset("HIV"),
    "cyp3a4": lambda: load_tdc_dataset("CYP3A4_Veith"),
    "cyp2d6": lambda: load_tdc_dataset("CYP2D6_Veith"),
    "cyp1a2": lambda: load_tdc_dataset("CYP1A2_Veith"),
    
    "mnist": _load_mnist_df,
    "fashion_mnist": _load_fashion_mnist_df,
}

def drop_rare_labels(df, ycol="target", min_count=2):
    """min_count 未満しか無いクラスは丸ごと捨てる"""
    vc = df[ycol].value_counts()
    ok_labels = vc[vc >= min_count].index
    return df[df[ycol].isin(ok_labels)].copy()

# -------------------------------------------------- #
# メイン関数                                         #
# -------------------------------------------------- #
from sklearn.preprocessing import LabelEncoder


def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config.dataset not in LOADERS:
        raise ValueError(f"unknown dataset: {config.dataset}")

    df = LOADERS[config.dataset]()
    df = drop_rare_labels(df, "target", min_count=2)

    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])

    # ── 目的変数と特徴量を分離
    y = df["target"]
    X = df.drop(columns=["target"])

    # ── one-hot encoding（targetは除く）
    X = pd.get_dummies(X, drop_first=True)

    # ── 特徴量を標準化 # one-hot も標準化している点に注意
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── numpy.ndarray を DataFrame に変換
    X = pd.DataFrame(X, columns=pd.get_dummies(df.drop(columns=["target"]), drop_first=True).columns)

    # ── 再結合
    df = pd.concat([X, y.reset_index(drop=True)], axis=1)



    if config.dataset == 'qsar':
        config.feature_num = 41
        config.dim_intermediate = 37 # 中間表現の次元数
        config.dim_integrate = 37 # 統合表現の次元数
        config.num_institution_user = 25
        config.num_institution = 20
        config.num_anchor_data = 396
        config.metrics = "accuracy"
    
    elif config.dataset == "adult":
        config.feature_num = 51
        config.dim_intermediate = 50 # 中間表現の次元数
        config.dim_integrate = 50 # 統合表現の次元数
        config.num_institution_user = 50
        config.num_institution = 10
        config.num_anchor_data = 693          

    elif config.dataset == "diabetes130":
        config.feature_num = 200
        config.dim_intermediate = 100 # 中間表現の次元数
        config.dim_integrate = 100 # 統合表現の次元数
        config.num_institution_user = 500
        config.num_institution = 10
        config.num_anchor_data = 693  
    
    
    elif config.dataset == 'mice':
        config.feature_num = 77
        config.dim_intermediate = 46 # 中間表現の次元数
        config.dim_integrate = 46 # 統合表現の次元数
        config.num_institution_user = 50
        #config.num_institution = 10
        config.num_anchor_data = 693
        config.metrics = "accuracy"        

    elif config.dataset == 'breast_cancer':
        config.feature_num = 15  # 特徴量の数（目的変数を除く）
        config.dim_intermediate = config.feature_num-1 # 中間表現の次元数
        config.dim_integrate = config.feature_num-1 # 統合表現の次元数
        config.num_institution_user = 16
        config.num_institution = min(100, int(len(df) / (config.num_institution_user * 2)))
        
    elif config.dataset == 'digits':
        #feature_num = 15  # 特徴量の数（目的変数を除く）
        #config.dim_intermediate = 5 # 中間表現の次元数
        #config.dim_integrate = 5 # 統合表現の次元数
        config.feature_num = min(len(df.columns) - 1, 51)
        config.num_institution_user = 30
        config.num_institution = min(100, int(len(df) / (config.num_institution_user * 2)))

    elif config.dataset == 'mnist' or config.dataset == 'fashion_mnist':
        config.feature_num = len(df.columns) - 1  # 特徴量の数（目的変数を除く）
        config.dim_intermediate = 50 # 中間表現の次元数
        config.dim_integrate = 50 # 統合表現の次元数
        config.num_institution = 50
        config.num_institution_user = 50
        
    elif config.dataset == 'concentric_circles':
        config.feature_num = 2
        config.dim_intermediate = 2  # 中間表現の次元数
        config.dim_integrate = 2  # 統合表現の次元数
        config.num_institution = 3
        config.num_institution_user = int(len(df) / (config.num_institution * 2))
    
    elif config.dataset == 'two_gaussian_distributions':
        config.feature_num = 2
        config.dim_intermediate = 2
        config.dim_integrate = 2
        config.num_institution_user = 50
        config.num_institution = 5
    
    elif config.dataset == '3D_gaussian_clusters':
        config.feature_num = 3
        config.dim_intermediate = 2
        config.dim_integrate = 2
        config.num_institution = 3
        config.num_institution_user = int(len(df) / (config.num_institution * 2))       
        
    else:
        #config.feature_num = min(len(df.columns) - 1)#, 50)  # 特徴量の数（目的変数を除く）
        #config.dim_intermediate = config.feature_num-1 # 中間表現の次元数
        #config.dim_integrate = config.feature_num-1 # 統合表現の次元数
        config.feature_num = len(df.columns) - 1
        config.dim_intermediate = config.feature_num - 1
        config.dim_integrate = config.feature_num - 1
        config.num_institution_user = max(config.dim_integrate + 1, 50) # int(len(df) / (config.num_institution * 2))  # 1機関あたりのユーザ数を計算
        config.num_institution = int(len(df) / (config.num_institution_user * 2))
    
    
    # 特徴量だけを取得（target を除外）
    y_name = config.y_name
    feature_columns = [col for col in df.columns if col != y_name]

    # 最大 feature_num 個までに制限
    limited_features = feature_columns[:config.feature_num]

    # 最終的に残す列（順序は：特徴量 + target）
    final_columns = limited_features + [y_name]

    # 制限後のデータフレーム
    df = df[final_columns]
    df = df[:2*config.num_institution_user*config.num_institution]
    
    
    # ── train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        random_state=config.seed,
        shuffle=True,
        stratify=df["target"]
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