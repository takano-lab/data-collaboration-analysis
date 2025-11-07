# data_loader.py
from __future__ import annotations

import pickle as pkl
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces, fetch_openml, load_digits, make_moons, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
# from tdc.single_pred import ADME, HTS, Tox

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

def _load_coil20_df() -> pd.DataFrame:
    """
    COIL-20 画像データを DataFrame 化して返す。
    - 入力: input/archive/coil-20/<label>/obj<label>__<rotation>.png
    - 出力: 各ピクセル列 + 'target'（= label）。rotation 列は作らない。
    """
    from PIL import Image

    root = Path("input/archive/coil-20")
    if not root.exists():
        raise FileNotFoundError("input/archive/coil-20 が見つかりません。")

    rows: list[tuple[int, np.ndarray]] = []
    first_shape = None

    # ラベルはサブフォルダ名（1..20）を想定
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        try:
            label = int(label_dir.name)
        except Exception:
            # 非数値ディレクトリは無視
            continue
        for p in sorted(label_dir.iterdir()):
            if p.suffix.lower() != ".png":
                continue
            # ファイル名パターンは気にせず rotation は使わない
            img = Image.open(p).convert("L")  # グレースケール
            arr = np.asarray(img, dtype=np.float32)
            if first_shape is None:
                first_shape = arr.shape
            rows.append((label, arr.reshape(-1)))

    if not rows:
        raise FileNotFoundError("COIL-20 の画像が見つかりませんでした。")

    d = rows[0][1].shape[0]
    pix_cols = [f"pix_{i:05d}" for i in range(d)]
    data = np.stack([x for (_, x) in rows], axis=0)
    labels = [lab for (lab, _) in rows]
    df = pd.DataFrame(data, columns=pix_cols)
    df.insert(0, "target", labels)
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

def _load_mnist_df_1248() -> pd.DataFrame:
    """
    MNIST からクラス {1,4,8} のみを抽出して返すローダ。
    後段の LabelEncoder で 0..2 にリマップされる想定。
    """
    df = _load_mnist_df()
    # OpenML の MNIST は target が文字列のことが多いので文字列化して判定
    mask = df["target"].astype(str).isin(["1", "2", "4", "8"])
    df = df.loc[mask].reset_index(drop=True)
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

def _load_cifar10_df() -> pd.DataFrame:
    """
    ローカルの CIFAR-10 (python 版) アーカイブから読み込み、DataFrame を返す。
    - 入力ファイル候補: input/cifar-10-python.tar.gz（推奨）
      フォールバック: C:\\Users\\sueya\\Git-Repositories\\takano_labo\\dca\\input\\cifar-10-python.tar.gz
    - 特徴量: 32x32x3 = 3072 次元（pix_00000..）
    - ラベル列: 'target'（クラス名: airplane, automobile, ...）
    """
    candidates = [
        Path("input/cifar-10-python.tar.gz"),
        Path(r"C:\\Users\\sueya\\Git-Repositories\\takano_labo\\dca\\input\\cifar-10-python.tar.gz"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("CIFAR-10 アーカイブが見つかりません: input/cifar-10-python.tar.gz")

    def _pkl_load(fileobj):
        return pkl.load(fileobj, encoding="latin1")

    with tarfile.open(path, mode="r:gz") as tar:
        # ラベル名を読み込む
        meta_member = next((m for m in tar.getmembers() if m.name.endswith("batches.meta")), None)
        if meta_member is None:
            raise FileNotFoundError("batches.meta がアーカイブ内に見つかりません")
        with tar.extractfile(meta_member) as f:
            meta = _pkl_load(f)
        label_names = meta.get("label_names") or meta.get(b"label_names")
        # デコード（bytes -> str）
        label_names = [ln.decode("utf-8") if isinstance(ln, (bytes, bytearray)) else str(ln) for ln in label_names]

        # データバッチを順に読み込む（train: 1..5, test: test_batch）
        batch_names = [
            "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch",
        ]
        def _find_member(endswith_name: str):
            return next((m for m in tar.getmembers() if m.name.endswith(endswith_name)), None)

        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        for bn in batch_names:
            member = _find_member(bn)
            if member is None:
                # 訓練/テストのどちらかが欠けていても継続
                continue
            with tar.extractfile(member) as f:
                d = _pkl_load(f)
            data = d.get("data") if "data" in d else d.get(b"data")
            labels = d.get("labels") if "labels" in d else d.get(b"labels")
            if data is None or labels is None:
                # CIFAR-10 の標準形式と異なる場合はスキップ
                continue
            X_arr = np.asarray(data, dtype=np.float32)  # (N, 3072)
            y_arr = np.asarray(labels, dtype=int)
            X_list.append(X_arr)
            y_list.append(y_arr)

    if not X_list:
        raise RuntimeError("CIFAR-10 のデータバッチが読み込めませんでした。")

    X = np.vstack(X_list)
    y_idx = np.concatenate(y_list)
    # インデックス -> クラス名
    y = [label_names[i] if 0 <= i < len(label_names) else str(i) for i in y_idx]

    pix_cols = [f"pix_{i:05d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=pix_cols)
    df.insert(0, "target", y)
    return df

def _load_cifar10_800_df() -> pd.DataFrame:
    """
    CIFAR-10 を読み込み、ピクセル(3072次元)を標準化後 PCA で 800 次元に圧縮して返す。
    - 返却: 'target' 列 + pc_1..pc_800 の数値列
    - PCA は randomized SVD を用い、random_state=0 で決定論的に。
    """
    from sklearn.preprocessing import StandardScaler

    base = _load_cifar10_df()
    y = base["target"].copy()
    X = base.drop(columns=["target"]).to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=800, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(Xs)

    cols = [f"pc_{i+1}" for i in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df.insert(0, "target", y.values)
    return df

# =============================
# MedMNIST loaders
# =============================
def _load_medmnist(kind: str) -> pd.DataFrame:
    """
    MedMNIST の各データセットを結合して DataFrame で返す。
    - kind 例: 'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
              'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
              'organamnist_axial', 'organamnist_coronal', 'organamnist_sagittal'
    - 画像はフラット化（pix_00000..）。
    - ラベルは 'target' 列に保存。マルチラベル（例: chestmnist）は multi_010... の文字列に変換。
    """
    try:
        import medmnist
        from medmnist import INFO
    except Exception as e:
        raise RuntimeError("MedMNIST を利用するには 'medmnist' パッケージが必要です。'uv add medmnist' 済みか確認してください。") from e

    key = kind.lower()
    if key not in INFO:
        raise ValueError(f"Unknown MedMNIST kind: {kind}")
    info = INFO[key]
    cls_name = info.get('python_class', None)
    if not cls_name or not hasattr(medmnist, cls_name):
        raise RuntimeError(f"MedMNIST dataset class not found for kind={kind} (python_class={cls_name})")
    DataClass = getattr(medmnist, cls_name)

    # as_rgb: 3ch のとき True に（flattenはどちらでも対応）
    as_rgb = bool(info.get('n_channels', 1) == 3)

    def _label_to_str(lbl: np.ndarray) -> str | int:
        arr = np.asarray(lbl).reshape(-1)
        if arr.size == 1:
            # 単ラベルは int
            try:
                return int(arr[0])
            except Exception:
                return int(float(arr[0]))
        # マルチラベルは multi_0101.. の文字列に
        bits = ''.join(str(int(x)) for x in arr)
        return f"multi_{bits}"

    X_rows: list[np.ndarray] = []
    y_rows: list[object] = []
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, download=True, as_rgb=as_rgb)
        # dataset は __getitem__ で (image, label) を返す
        for i in range(len(ds)):
            img, label = ds[i]
            # 画像を ndarray 化してフラット
            img_np = np.asarray(img)
            X_rows.append(img_np.reshape(-1).astype(float))
            y_rows.append(_label_to_str(label))

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=object)
    pix_cols = [f"pix_{i:05d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=pix_cols)
    df.insert(0, "target", y)
    return df

#def _load_medmnist_1() -> pd.DataFrame:  # pathmnist
#    return _load_medmnist('pathmnist')

#def _load_medmnist_2() -> pd.DataFrame:  # chestmnist (multi-label)
#    return _load_medmnist('chestmnist')

def _load_medmnist_3() -> pd.DataFrame:  # dermamnist
    """DermaMNIST を読み込み、クラスID 3 (Dermatofibroma) を除外して返す。"""
    df = _load_medmnist('dermamnist')
    # target は単一ラベル（int）想定
    if 'target' in df.columns:
        df = df[df['target'] != 3].reset_index(drop=True)
    return df

#def _load_medmnist_4() -> pd.DataFrame:  # octmnist
#    return _load_medmnist('octmnist')

def _load_medmnist_5() -> pd.DataFrame:  # pneumoniamnist
    return _load_medmnist('pneumoniamnist')

def _load_medmnist_6() -> pd.DataFrame:  # retinamnist
    """RetinaMNIST を読み込み、クラスID 4 (Proliferative) を除外して返す。"""
    df = _load_medmnist('retinamnist')
    if 'target' in df.columns:
        df = df[df['target'] != 4].reset_index(drop=True)
    return df

def _load_medmnist_7() -> pd.DataFrame:  # breastmnist
    return _load_medmnist('breastmnist')

#def _load_medmnist_8() -> pd.DataFrame:  # bloodmnist
#    return _load_medmnist('bloodmnist')

def _load_medmnist_9() -> pd.DataFrame:  # tissuemnist
    return _load_medmnist('tissuemnist')

def _load_medmnist_10() -> pd.DataFrame:  # organamnist_axial
    return _load_medmnist('organamnist_axial')

def _load_medmnist_11() -> pd.DataFrame:  # organamnist_coronal
    return _load_medmnist('organamnist_coronal')

def _load_medmnist_12() -> pd.DataFrame:  # organamnist_sagittal
    return _load_medmnist('organamnist_sagittal')

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


def _load_iris() -> pd.DataFrame:
    """
    Iris flower dataset.
    既存のパイプラインに合わせて 'target' 列を保持して返す。
    まずは scikit-learn 組込みを使用。
    """
    from sklearn.datasets import load_iris

    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    # 既に 'target' 列を含む
    return df


def _load_ecoli() -> pd.DataFrame:
    """
    UCI Ecoli dataset。
    - まずローカル候補ファイルを探し、なければ OpenML から取得。
    - 'sequence_name' などの識別子は落とし、'class' を 'target' に。
    """
    # ローカル候補
    candidates = [
        Path("input/ecoli/ecoli.data"),
        Path("input/ecoli.data"),
        Path("input/ecoli/ecoli.csv"),
        Path("input/ecoli.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is not None:
        if path.suffix.lower() == ".data":
            cols = [
                "sequence_name",
                "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class",
            ]
            df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        else:
            df = pd.read_csv(path)
            # 列名が不足していたら補う
            if "class" not in df.columns:
                # 最右列がクラスである想定
                df.columns = [
                    "sequence_name",
                    "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class",
                ][: len(df.columns)]
        # ID列を落とす
        if "sequence_name" in df.columns:
            df = df.drop(columns=["sequence_name"]) 
        # クラス名を統一
        if "class" in df.columns:
            df = df.rename(columns={"class": "target"})
        return df

    # Fallback: OpenML
    data = fetch_openml("ecoli", version=1, as_frame=True)
    df = data.frame.copy()
    # OpenMLではターゲット名が 'class' のことが多い
    if "target" not in df.columns:
        for cand in ["class", "Class", "target", "label", "y"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "target"})
                break
        else:
            # Bunch.target があれば利用
            if getattr(data, "target", None) is not None:
                df["target"] = data.target
            else:
                raise ValueError("Ecoli: target 列が判別できませんでした。")
    # 不要IDがあれば落とす
    for c in ["sequence_name", "id", "ID", "instance"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def _load_vowel() -> pd.DataFrame:
    """
    Vowel recognition dataset。
    - まずローカル候補を探し、無ければ OpenML 'vowel' を利用。
    - ラベル列を 'target' に統一。
    """
    # ローカル候補
    candidates = [
        Path("input/vowel/vowel.data"),
        Path("input/vowel.data"),
        Path("input/vowel/vowel.csv"),
        Path("input/vowel.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is not None:
        # UCI 元データの整形はバリエーションが多いので、柔軟に吸収
        if path.suffix.lower() in [".data", ".txt"]:
            df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None)
        else:
            df = pd.read_csv(path)
        # 列名推定（既にヘッダ付きかもしれない）
        if "target" not in df.columns and not {"class", "Class"} & set(df.columns):
            # 典型例: 1列目 speaker, 2列目 sex, 3列目 item, 次に 10特徴, そしてクラス
            if df.shape[1] >= 13:
                base = ["speaker", "sex", "item"] + [f"f{i}" for i in range(1, 11)] + ["class"]
                df.columns = base[: df.shape[1]]
        # ラベル名統一
        for cand in ["class", "Class", "Vowel", "label", "y"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "target"})
                break
        return df

    # Fallback: OpenML
    data = fetch_openml("vowel", version=1, as_frame=True)
    df = data.frame.copy()
    if "target" not in df.columns:
        for cand in ["Class", "class", "y", "label", "Vowel"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "target"})
                break
        else:
            if getattr(data, "target", None) is not None:
                df["target"] = data.target
            else:
                raise ValueError("Vowel: target 列が判別できませんでした。")
    # 不要ID候補を落とす
    for c in ["id", "ID", "instance", "Train", "Test"]:
        if c in df.columns:
            df = df.drop(columns=[c])
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
    "coil20": _load_coil20_df,
    "cifar10": _load_cifar10_df,
    # cifar10 -> 800 次元 PCA 版
    "cifar10_800": _load_cifar10_800_df,
    "cifer10_800": _load_cifar10_800_df,  # タイポ対策のエイリアス
    # === MedMNIST ===
    #"medmnist_1": _load_medmnist_1,  # pathmnist (9)
    #"medmnist_2": _load_medmnist_2,  # chestmnist (14, multi-label)
    "medmnist_3": _load_medmnist_3,  # dermamnist (7)
    #"medmnist_4": _load_medmnist_4,  # octmnist (4)
    "medmnist_5": _load_medmnist_5,  # pneumoniamnist (2)
    "medmnist_6": _load_medmnist_6,  # retinamnist (5)
    "medmnist_7": _load_medmnist_7,  # breastmnist (2)
    #"medmnist_8": _load_medmnist_8,  # bloodmnist (8)
    "medmnist_9": _load_medmnist_9,  # tissuemnist (8)
    "medmnist_10": _load_medmnist_10,  # organamnist_axial (11)
    "medmnist_11": _load_medmnist_11,  # organamnist_coronal (11)
    "medmnist_12": _load_medmnist_12,  # organamnist_sagittal (11)
    # New: classic ML small datasets
    "iris": _load_iris,
    "ecoli": _load_ecoli,
    "vowel": _load_vowel,
    
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
    # Subset MNIST
    "mnist_1248": _load_mnist_df_1248,
}

def drop_rare_labels(df, ycol="target", min_count=2):
    """min_count 未満しか無いクラスは丸ごと捨てる"""
    vc = df[ycol].value_counts()
    ok_labels = vc[vc >= min_count].index
    return df[df[ycol].isin(ok_labels)].copy()

import pandas as pd

# -------------------------------------------------- #
# メイン関数                                         #
# -------------------------------------------------- #
from sklearn.preprocessing import LabelEncoder


# ==================================================
# 内部ヘルパー（振る舞いを変えずに段階的リファクタ）
# ==================================================
def _one_hot_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """LabelEncoder 適用後の df (target 数値化済) を one-hot & 標準化する。
    元コードの順序: one-hot -> 標準化 -> DataFrame 再構築 を忠実に再現。
    """
    y = df["target"].reset_index(drop=True)
    X_raw = df.drop(columns=["target"])  # target 以外
    X_onehot = pd.get_dummies(X_raw, drop_first=True)
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_onehot)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X_onehot.columns)
    return pd.concat([X_scaled, y], axis=1)


## NOTE: _initialize_config_params は institution_data 側へ責務移行のため削除

def load_data(config: Config) -> pd.DataFrame:
    """データセットを読み込み、one-hot+標準化まで行い単一 DataFrame を返す。
    機関関連のパラメータ補完や列制限は institution_data.prepare_institutional_dataset 内で実行する。"""
    if config.dataset not in LOADERS:
        raise ValueError(f"unknown dataset: {config.dataset}")

    df_raw = LOADERS[config.dataset]()
    df_raw = drop_rare_labels(df_raw, "target", min_count=2)
    le = LabelEncoder()
    df_raw["target"] = le.fit_transform(df_raw["target"])

    df_proc = _one_hot_and_scale(df_raw)
    return df_proc

# -------------------------------------------------- #
# 例: 実行                                           #
# -------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config(name="statlog", output_path=Path("./statlog_split"))
    load_data(cfg)
