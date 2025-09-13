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

def _load_housing() -> pd.DataFrame:
    """
    California Housing データセットを DataFrame で返す。
    目的変数は 'target' 列に格納。
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"MedHouseVal": "target"})  # 目的変数を 'target' に統一
    return df

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
    df = pd.read_csv(r"C:\Users\sueya\Git-Repositories\takano_labo\dca_yanagi\input\qsar+biodegradation\biodeg.csv", header=None, sep=";")
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
    # 'org' 列が存在する場合のみ削除
    if "org" in df.columns:
        df = df.drop(columns=["org"])
    
        # NaN 値を確認し、処理する
    if df.isnull().any().any():
        # NaN を 0 で埋める場合
        df = df.fillna(0)
        # または、NaN を削除する場合
        # df = df.dropna()
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

def _load_concentric_three_circles_df() -> pd.DataFrame:
    path = Path("input/concentric_three_classes_big.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"y": "target"})
    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df = df.sample(frac=1).reset_index(drop=True)
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
    path = Path("input/3D_3_Gaussian_Clusters.csv")
    df = pd.read_csv(path)
    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df = df.sample(frac=1).reset_index(drop=True)
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df

def _load_3D_8_gaussian_clusters_df() -> pd.DataFrame:
    path = Path("input/3D_8_Gaussian_Clusters.csv")
    df = pd.read_csv(path)
    # "target" 列以外を標準化
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != "target"]
    df = df.sample(frac=1).reset_index(drop=True)
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

def _load_wine_quality() -> pd.DataFrame:
    """
    UCI Wine Quality (red/white)。セミコロン区切り。
    例:
      input/winequality/winequality-red.csv
      input/winequality/winequality-white.csv
      input/wine+quality/winequality-red.csv など
    'quality' を 3クラスにビニング（low/mid/high）。不足時は 2クラスへフォールバック。
    """
    folder_candidates = [
        Path("input/winequality"),
        Path("input/wine+quality"),
        Path("input/wine_quality"),
        Path("input/wine"),
    ]
    red = None
    white = None
    for f in folder_candidates:
        if (f / "winequality-red.csv").exists():
            red = f / "winequality-red.csv"
        if (f / "winequality-white.csv").exists():
            white = f / "winequality-white.csv"

    if not red and not white:
        raise FileNotFoundError("Wine Quality の CSV が見つかりません。例: input/winequality/winequality-red.csv")

    frames = []
    if red:
        df_r = pd.read_csv(red, sep=";")
        df_r["wine_type"] = "red"
        frames.append(df_r)
    if white:
        df_w = pd.read_csv(white, sep=";")
        df_w["wine_type"] = "white"
        frames.append(df_w)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # quality を 3クラスへ（多くの研究で使われる簡便な分割）
    # low: <=5, mid: ==6, high: >=7
    q = df["quality"].astype(int)
    bins3 = pd.Series(np.where(q <= 5, "low", np.where(q >= 7, "high", "mid")))
    df = df.drop(columns=["quality"])
    df = df.rename(columns={"quality": "quality_orig"}) if "quality" in df.columns else df
    df["target"] = bins3

    # クラス数チェック → 足りなければ 2値へ（<=5: bad, >=6: good）
    vc = df["target"].value_counts()
    if (vc < 10).any():
        df["target"] = np.where(q >= 6, "good", "bad")

    return df


def _load_glass() -> pd.DataFrame:
    """
    UCI Glass Identification。ヘッダ無し CSV。
    例:
      input/glass+identification/glass.data
      input/glass/glass.data
    最初の列は ID なので落とし、最後の Type を 'target' に。
    """
    candidates = [
        Path("input/glass+identification/glass.data"),
        Path("input/glass/glass.data"),
        Path("input/glass.data"),
        Path("input/glass/glass.csv"),
        Path("input/glass.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Glass データが見つかりません。例: input/glass+identification/glass.data")

    columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "target"]
    df = pd.read_csv(path, header=None, names=columns)
    df = df.drop(columns=["Id"])
    return df


def _load_seeds() -> pd.DataFrame:
    """
    UCI Seeds。空白区切り or CSV。
    例:
      input/seeds/seeds_dataset.txt
      input/seeds/seeds_dataset.csv
      input/seeds_dataset.txt
    最後の列(1..3)を 'target' に。
    """
    candidates = [
        Path("input/seeds/seeds_dataset.txt"),
        Path("input/seeds/seeds_dataset.csv"),
        Path("input/seeds_dataset.txt"),
        Path("input/seeds_dataset.csv"),
        Path("input/seeds/seeds.data"),
        Path("input/seeds/seeds.data.txt"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Seeds データが見つかりません。例: input/seeds/seeds_dataset.txt")

    columns = [
        "area",
        "perimeter",
        "compactness",
        "length_of_kernel",
        "width_of_kernel",
        "asymmetry_coefficient",
        "length_of_kernel_groove",
        "target",
    ]
    if path.suffix.lower() in [".txt", ".data"]:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    else:
        df = pd.read_csv(path, header=None, names=columns)
    return df


def _load_letter_recognition() -> pd.DataFrame:
    """
    UCI Letter Recognition。先頭列が A..Z のラベル。
    例:
      input/letter+recognition/letter-recognition.data
      input/letter-recognition/letter-recognition.data
    """
    candidates = [
        Path("input/letter+recognition/letter-recognition.data"),
        Path("input/letter-recognition/letter-recognition.data"),
        Path("input/letter_recognition/letter-recognition.data"),
        Path("input/letter-recognition.csv"),
        Path("input/letter_recognition.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Letter Recognition データが見つかりません。例: input/letter+recognition/letter-recognition.data")

    columns = [
        "target",
        "x-box", "y-box", "width", "high", "onpix",
        "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
        "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
    ]
    df = pd.read_csv(path, header=None, names=columns)
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
    "digits_v2": _load_digits_df,
    "concentric_circles": _load_concentric_circles_df,
    "concentric_three_circles": _load_concentric_three_circles_df,
    "two_gaussian_distributions": _load_two_gaussian_distributions_df,
    "3D_gaussian_clusters": _load_3D_gaussian_clusters_df,
    "3D_8_gaussian_clusters": _load_3D_8_gaussian_clusters_df,
    "mice": _load_mice_df,
    "housing": _load_housing,
    
    # UCI: 追加
    "wine_quality": _load_wine_quality,
    "glass": _load_glass,
    "seeds": _load_seeds,
    "letter_recognition": _load_letter_recognition,

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
import pandas as pd

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
    import pandas as pd
    X = pd.get_dummies(X, drop_first=True)

    # ── 特徴量を標準化 # one-hot も標準化している点に注意
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── numpy.ndarray を DataFrame に変換
    X = pd.DataFrame(X, columns=pd.get_dummies(df.drop(columns=["target"]), drop_first=True).columns)

    # ── 再結合
    df = pd.concat([X, y.reset_index(drop=True)], axis=1)

    # if config.dataset == 'qsar':
    #     config.feature_num = 41
    #     config.dim_intermediate = 37 # 中間表現の次元数
    #     config.dim_integrate = 37 # 統合表現の次元数
    #     config.num_institution_user = 25 # 100
    #     config.num_institution = 20#int(len(df) / (config.num_institution_user * 2))                                                            #20
    #     #config.num_anchor_data = 396
    #     #config.metrics = "accuracy"
    
    # elif config.dataset == "adult":
    #     config.feature_num = 51
    #     config.dim_intermediate = 50 # 中間表現の次元数
    #     config.dim_integrate = 50 # 統合表現の次元数
    #     config.num_institution_user = 150#50
    #     config.num_institution = 10
    #     #config.num_anchor_data = 693          

    # elif config.dataset == "diabetes130":
    #     config.feature_num = 200
    #     config.dim_intermediate = 100 # 中間表現の次元数
    #     config.dim_integrate = 100 # 統合表現の次元数
    #     config.num_institution_user = 500
    #     config.num_institution = 10
    #     #config.num_anchor_data = 693  
    
    
    # elif config.dataset == 'mice':
    #     config.feature_num = 77
    #     config.dim_intermediate = 46 # 中間表現の次元数
    #     config.dim_integrate = 46 # 統合表現の次元数
    #     config.num_institution_user = 50 # 50
    #     config.num_institution = 5
    #     #config.num_anchor_data = 693
    #     #config.metrics = "accuracy"        

    # elif config.dataset == 'breast_cancer':
    #     config.feature_num = 15  # 特徴量の数（目的変数を除く）
    #     config.dim_intermediate = config.feature_num-1 # 中間表現の次元数
    #     config.dim_integrate = config.feature_num-1 # 統合表現の次元数
    #     config.num_institution_user = 60#16
    #     config.num_institution = min(100, int(len(df) / (config.num_institution_user * 2)))
    #     #config.metrics = "accuracy"
        
    # elif config.dataset == 'digits':
    #     config.feature_num =  len(df.columns) - 1 # 特徴量の数（目的変数を除く）
    #     config.dim_intermediate = 15 # 中間表現の次元数
    #     config.dim_integrate = 15 # 統合表現の次元数
    #     config.feature_num = min(len(df.columns) - 1, 51)
    #     config.num_institution_user = 100#30
    #     config.num_institution = 10#min(100, int(len(df) / (config.num_institution_user * 2)))

    # elif config.dataset == 'mnist' or config.dataset == 'fashion_mnist':
    #     config.feature_num = len(df.columns) - 1  # 特徴量の数（目的変数を除く）
    #     config.dim_intermediate = 10 # 中間表現の次元数
    #     config.dim_integrate = 10 # 統合表現の次元数
    #     config.num_institution_user = 50
    #     config.num_institution = 10
                                                                      
    # elif config.dataset == 'concentric_circles':
    #     config.feature_num = 2
    #     config.dim_intermediate = 2  # 中間表現の次元数
    #     config.dim_integrate = 2  # 統合表現の次元数
    #     config.num_institution = 2
    #     config.num_institution_user = int(len(df) / (config.num_institution * 2))                                                              

    # elif config.dataset == 'concentric_three_circles':
    #     config.feature_num = 2
    #     config.dim_intermediate = 2  # 中間表現の次元数
    #     config.dim_integrate = 2  # 統合表現の次元数
    #     config.num_institution = 2
    #     config.num_institution_user = int(len(df) / (config.num_institution * 2))               
        
    # elif config.dataset == 'two_gaussian_distributions':
    #     config.feature_num = 2
    #     config.dim_intermediate = 2
    #     config.dim_integrate = 2
    #     config.num_institution_user = 50
    #     config.num_institution = 5
    
    # elif config.dataset == '3D_gaussian_clusters':
    #     config.feature_num = 3
    #     config.dim_intermediate = 2
    #     config.dim_integrate = 2
    #     config.num_institution = 2
    #     config.num_institution_user = int(len(df) / (config.num_institution * 2))       
    
    # elif config.dataset == '3D_8_gaussian_clusters':
    #     config.feature_num = 3
    #     config.dim_intermediate = 2
    #     config.dim_integrate = 2
    #     config.num_institution = 2
    #     config.num_institution_user = int(len(df) / (config.num_institution * 2))     
            
    # elif config.dataset == 'digits_':
    #     config.feature_num = len(df.columns) - 1
    #     config.dim_intermediate = 4
    #     config.dim_integrate = 4
    #     config.num_institution = 10
    #     config.num_institution_user = 100    
    
    # elif config.dataset == 'digits_v2':
    #     config.feature_num = len(df.columns) - 1
    #     config.dim_intermediate = 30
    #     config.dim_integrate = 30
    #     config.num_institution = 29
    #     config.num_institution_user = 30

    # elif config.dataset == 'housing':
    #     config.feature_num = len(df.columns) - 1
    #     config.dim_intermediate = config.feature_num - 1
    #     config.dim_integrate = config.feature_num - 1
    #     config.num_institution = 10
    #     config.num_institution_user = 10
    #     config.metrics = "rmse"

    # elif config.dataset == "statlog":
    #     config.feature_num = len(df.columns) - 1
    #     config.dim_intermediate = config.feature_num - 1
    #     config.dim_integrate = config.feature_num - 1
    #     config.num_institution_user = 200# 30#, max(config.dim_integrate + 1, 50) # int(len(df) / (config.num_institution * 2))  # 1機関あたりのユーザ数を計算
    #     config.num_institution = min(int(len(df) / (config.num_institution_user * 2)), 5)
    #     #config.metrics = "auc"
    
    print("config.feature_num", config.feature_num)
    
    # else: のフォールバックを安全に（None/"undefined"/<=0 を未設定扱い）
    import numpy as np
    
    def _is_undefined(v):
        return (
            v is None
            or (isinstance(v, str) and v.strip().lower() in ("undefined", "none", ""))
            or (isinstance(v, (int, float)) and v <= 0)
        )
    if _is_undefined(config.feature_num):
        config.feature_num = len(df.columns) - 1

    if _is_undefined(config.dim_intermediate):
        config.dim_intermediate = config.feature_num - 1
    if _is_undefined(config.dim_integrate):
        config.dim_integrate = config.dim_intermediate
    # num_institution_user が未設定ならグローバルDEFAULTSで与える想定だが、最後の砦として50
    if _is_undefined(config.num_institution_user):
        config.num_institution_user = 50
    if _is_undefined(config.num_institution):
        # クラス分布を考慮して安全に決める
        y = df["target"].to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        # 機関あたりのユーザ数は、最低でもクラス数を満たす
        if _is_undefined(config.num_institution_user) or config.num_institution_user < n_classes:
            config.num_institution_user = max(int(config.num_institution_user or 0), n_classes)

        # 総件数による上限
        max_by_total = len(df) // (2 * config.num_institution_user)
        # クラス頻度による上限（各クラスにつき train/test に1件ずつ必要 → 2*num_institution）
        max_by_class = int(np.min(counts) // 2)

        safe_num_inst = max(1, min(max_by_total, max_by_class))
        config.num_institution = safe_num_inst
    print(f"num_institution_user={config.num_institution_user}, num_institution={config.num_institution}")
    print("config.feature_num", config.feature_num, "config.dim_intermediate", config.dim_intermediate, "config.dim_integrate", config.dim_integrate)
    #config.num_institution_user *= 3
    #config.num_institution //= 3
    #config.num_institution = int(config.num_institution)
    #config.dim_intermediate = 2
    #config.dim_integrate = 2
    
    # 特徴量だけを取得（target を除外）
    y_name = config.y_name
    feature_columns = [col for col in df.columns if col != y_name]

    # 最大 feature_num 個までに制限
    limited_features = feature_columns[:config.feature_num]

    # 最終的に残す列（順序は：特徴量 + target）
    final_columns = limited_features + [y_name]

    # 制限後のデータフレーム
    df = df[final_columns]
    #df = df[:2*config.num_institution_user*config.num_institution]


    if config.dataset == "housing":
        # ── train/test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.5,
            random_state=config.seed,
            shuffle=True,
        )

    else:
        # ── train/test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.5,
            random_state=config.seed,
            shuffle=True,
            stratify=df["target"]
        )
        
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    def split_train_test_by_institution(
        df: pd.DataFrame,
        label_col: str,
        num_institution: int,
        num_institution_user: int,
        random_state: int = 42,
    ):
        """
        各 institution の train/test それぞれが「全クラスを最低1件ずつ」含むように分割する。
        返す DataFrame は行順が institution 単位にまとまっており、
        先頭から num_institution_user 行ごとが 1 機関に対応する（train/test とも）。
        """
        import numpy as np

        rng = np.random.default_rng(random_state)

        y = df[label_col].to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        n_classes = classes.size

        # 必要条件チェック
        n_per_side = num_institution * num_institution_user
        if 2 * n_per_side > len(df):
            raise ValueError(f"rows={len(df)} < needed(total)={2*n_per_side}")
        if num_institution_user < n_classes:
            raise ValueError(f"num_institution_user({num_institution_user}) < n_classes({n_classes}) -> "
                            f"各機関に各クラス最低1件ずつは不可能です。")
        # 各クラスの最低必要数（両側で各機関1件ずつ）= 2 * num_institution
        need_per_class = 2 * num_institution
        lack = {int(c): int(n) for c, n in zip(classes, counts) if n < need_per_class}
        if lack:
            raise ValueError(f"各クラスの件数不足: {lack} "
                            f"(必要: 各クラス {need_per_class} 件以上)")

        # まず「保証割当」: 各クラスから train/test の各機関へ1件ずつ
        N = len(df)
        all_idx = np.arange(N)

        train_bins = [[] for _ in range(num_institution)]
        test_bins  = [[] for _ in range(num_institution)]

        used = set()

        for c in classes:
            idx_c = np.flatnonzero(y == c)
            rng.shuffle(idx_c)
            # 先頭 num_institution を train に1件ずつ
            for i in range(num_institution):
                tr_id = idx_c[i]
                train_bins[i].append(tr_id)
                used.add(int(tr_id))
            # 次の num_institution を test に1件ずつ
            for i in range(num_institution):
                te_id = idx_c[num_institution + i]
                test_bins[i].append(te_id)
                used.add(int(te_id))
            # 残りは未使用プールへ自然に落ちる

        # 残りプール
        remain = np.array([idx for idx in all_idx if idx not in used], dtype=int)
        rng.shuffle(remain)

        # 片側の残り必要数
        remain_train_need = n_per_side - sum(len(b) for b in train_bins)
        remain_test_need  = n_per_side - sum(len(b) for b in test_bins)
        if remain_train_need < 0 or remain_test_need < 0:
            raise RuntimeError("内部計算エラー: 残り必要数が負になりました。")

        if remain.size < (remain_train_need + remain_test_need):
            raise ValueError("残りサンプルが不足しています。パラメータを見直してください。")

        train_pool = remain[:remain_train_need]
        test_pool  = remain[remain_train_need: remain_train_need + remain_test_need]

        # プールを各機関に均等配分して、各 bin を num_institution_user 件に揃える
        def distribute(pool, bins, target_size):
            pool = list(pool)
            p = 0
            for i in range(len(bins)):
                need = target_size - len(bins[i])
                if need <= 0:
                    continue
                take = min(need, len(pool) - p)
                if take > 0:
                    bins[i].extend(pool[p:p+take])
                    p += take
            # まだ満たない bin があればラウンドロビンで埋める
            i = 0
            while any(len(b) < target_size for b in bins) and p < len(pool):
                if len(bins[i]) < target_size:
                    bins[i].append(pool[p]); p += 1
                i = (i + 1) % len(bins)
            if any(len(b) < target_size for b in bins):
                raise ValueError("配分サンプル不足で各 bin を target_size に揃えられません。")
            return bins

        train_bins = distribute(train_pool, train_bins, num_institution_user)
        test_bins  = distribute(test_pool,  test_bins,  num_institution_user)

        # institution 順に結合（各 bin 内はシャッフルしておく）
        for b in train_bins:
            rng.shuffle(b)
        for b in test_bins:
            rng.shuffle(b)

        train_idx = np.concatenate([np.array(b, dtype=int) for b in train_bins])
        test_idx  = np.concatenate([np.array(b, dtype=int) for b in test_bins])

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # 検証（各機関の train/test が全クラスを含むか）
        def check(df_side, name):
            ok = True
            for i in range(num_institution):
                part = df_side.iloc[i*num_institution_user:(i+1)*num_institution_user]
                have = np.unique(part[label_col].to_numpy())
                if len(np.intersect1d(have, classes)) != n_classes:
                    ok = False
                    # 必要ならログ: print(f"[WARN] {name} inst#{i}: classes={sorted(have)}")
            return ok
        assert check(train_df, "train") and check(test_df, "test"), \
            "内部検証エラー: 条件を満たせていません。"

        return train_df, test_df

    # 使い方例
    train_df, test_df = split_train_test_by_institution(df, "target", config.num_institution, config.num_institution_user)

    
    # ── 行数制約でカット（デモ用）
    # lim = config.num_institution * config.num_institution_user
    # train_df = train_df.iloc[:lim].reset_index(drop=True)
    # test_df = test_df.iloc[:lim].reset_index(drop=True)

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