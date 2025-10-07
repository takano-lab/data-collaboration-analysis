"""institution_data.py
機関（institution）単位のデータ分割ユーティリティ。

責務:
 1. 特徴量数・機関数・1機関当たりサンプル数など Config に基づき DataFrame を調整
 2. train/test への分割
 3. 指定された機関分布方式 (even / division) で機関ごとに行を並べ替え
 4. 各機関の (X, y) を numpy 配列リスト Xs_train, Xs_test, ys_train, ys_test として返却

用語:
  - num_institution: 機関数 (train/test 共通) ※必要条件を満たさない場合は内部で安全な値へ調整
  - num_institution_user: 1機関あたりのサンプル数 (train/test それぞれ)
  - division 方式: ラベル半々で train/test に分割後、学習側の機関数を
      k * num_labels (k は最大の整数, k * num_labels <= 要求 num_institution) に調整。
      1 機関 1 ラベルのみを保持 (ラベル毎に均等な機関数)。テスト側も同じ構造。
  - even 方式: 各機関が全ラベルを少なくとも 1 件ずつ含む（元 `split_train_test_by_institution_even` を移植）。

注意:
  - 既存コードとの互換性のため、旧 `clip_datasets` / `train_test_split` は
    インターフェースを保ちつつ名称を調整。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

try:
    from tqdm import tqdm
except ImportError:  # フォールバック（tqdm 非導入環境）
    def tqdm(x, **kwargs):
        return x

from config.config import Config

# ================================================== #
# ヘルパー: パラメータ安全化 & 特徴量クリップ
# ================================================== #

def _is_undefined(v) -> bool:
    return (
        v is None
        or (isinstance(v, str) and v.strip().lower() in ("undefined", "none", ""))
        or (isinstance(v, (int, float)) and v <= 0)
    )


############################################
# 新: 3 分割ヘルパー (設定補正 / 特徴量制限 / 基本 train-test 分割)
############################################

def ensure_institution_params(df: pd.DataFrame, config: Config) -> None:
    """config の未設定パラメータを df に基づき補正し、適切な num_institution を設定する。"""
    y_name = getattr(config, "y_name", "target")
    if _is_undefined(config.feature_num):
        config.feature_num = len(df.columns) - 1
    if _is_undefined(config.dim_intermediate):
        config.dim_intermediate = config.feature_num - 1
    if _is_undefined(config.dim_integrate):
        config.dim_integrate = config.dim_intermediate
    if _is_undefined(config.num_institution_user):
        config.num_institution_user = 50

    if _is_undefined(config.num_institution):
        y = df[y_name].to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if _is_undefined(config.num_institution_user) or config.num_institution_user < n_classes:
            config.num_institution_user = max(int(config.num_institution_user or 0), n_classes)
        max_by_total = len(df) // (2 * config.num_institution_user)
        max_by_class = int(np.min(counts) // 2)
        config.num_institution = max(1, min(max_by_total, max_by_class))


def limit_feature_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """feature_num に従って特徴量列を制限し DataFrame を返す。"""
    y_name = getattr(config, "y_name", "target")
    feature_columns = [c for c in df.columns if c != y_name]
    limited_features = feature_columns[: config.feature_num]
    final_columns = limited_features + [y_name]
    return df[final_columns].copy()


def basic_train_test_split(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """単純な train/test 分割 (stratify は housing を除き有効)"""
    y_name = getattr(config, "y_name", "target")
    stratify_arg = df[y_name] if getattr(config, "dataset", None) != "housing" else None
    train_df, test_df = sk_train_test_split(
        df,
        test_size=0.5,
        shuffle=True,
        random_state=getattr(config, "seed", 42),
        stratify=stratify_arg,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ================================================== #
# even 方式 (既存ロジック移植): 各機関が全クラス保持
# ================================================== #
def split_train_test_by_institution_even(
    df: pd.DataFrame,
    config: Config,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """even 方式: df だけ受け取り、内部で
      1) train/test 分割
      2) 各機関に全クラス1件割当
      3) 行順を機関ブロック化
    を行う。config.num_institution は ensure_institution_params で設定済み想定。
    """
    label_col = getattr(config, "y_name", "target")
    rng = np.random.default_rng(random_state)

    # まず train/test split
    train_df, test_df = basic_train_test_split(df, config)
    num_institution = config.num_institution
    num_institution_user = config.num_institution_user

    def _repack(side_df: pd.DataFrame) -> pd.DataFrame:
        y = side_df[label_col].to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if num_institution_user < n_classes:
            raise ValueError("num_institution_user < #classes: even 方式は不可能です")
        need_per_class = num_institution
        lack = {int(c): int(n) for c, n in zip(classes, counts) if n < need_per_class}
        if lack:
            raise ValueError(f"クラス不足 (even): {lack}")
        bins = [[] for _ in range(num_institution)]
        used = set()
        for c in classes:
            idx_c = np.flatnonzero(y == c)
            rng.shuffle(idx_c)
            for i in range(num_institution):
                bins[i].append(idx_c[i])
                used.add(int(idx_c[i]))
        remain = [i for i in range(len(side_df)) if i not in used]
        rng.shuffle(remain)
        target_size = num_institution_user
        p = 0
        for i in range(num_institution):
            need = target_size - len(bins[i])
            take = min(need, len(remain) - p)
            if take > 0:
                bins[i].extend(remain[p:p + take])
                p += take
        for b in bins:
            rng.shuffle(b)
        new_idx = np.concatenate([np.array(b, dtype=int) for b in bins])
        return side_df.iloc[new_idx].reset_index(drop=True)

    train_packed = _repack(train_df)
    test_packed = _repack(test_df)
    return train_packed, test_packed


# ================================================== #
# division 方式: 1 機関 1 ラベル
# ================================================== #
def split_train_test_by_institution_division(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    requested_num_institution: int,
    num_institution_user: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """division 方式: 各機関は単一ラベルのみ保持。

    手順:
      1. train/test は事前 split 済みで入力される想定。
      2. ラベル集合 size = L。
      3. 機関数 = floor(requested_num_institution / L) * L (>= L の倍数が 0 の場合は L にフォールバック)。
      4. 各ラベルに同数の機関 (k) を割当て (k = num_institution // L)。
      5. 各ラベル内サンプルをシャッフルし、機関ごとに num_institution_user 件ずつ切り出し。
         もし不足したら可能な最大数で再調整 (k や num_institution_user を縮退) し再割当。
      6. test 側も同様。

    Returns
    -------
    train_packed, test_packed, final_num_institution
    """
    rng = np.random.default_rng(random_state)

    labels = np.array(sorted(train_df[label_col].unique()))
    L = len(labels)
    if L == 0:
        raise ValueError("ラベルが存在しません")

    # 機関数決定
    if requested_num_institution < L:
        # 倍数が 0 になるケース → フォールバック
        final_num_institution = L
    else:
        k = requested_num_institution // L
        final_num_institution = max(L, k * L)

    k = final_num_institution // L  # ラベル毎の機関数

    def _allocate(side_df: pd.DataFrame) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        for lab in labels:
            sub = side_df[side_df[label_col] == lab].sample(frac=1, random_state=rng.integers(0, 1_000_000))
            needed = k * num_institution_user
            if len(sub) < needed:
                # 縮退: 取れるだけ取り、足りない機関は削除
                possible_k = max(1, len(sub) // max(1, num_institution_user))
                if possible_k == 0:
                    raise ValueError(f"ラベル {lab} のサンプル不足 (len={len(sub)})")
                use_k = possible_k
            else:
                use_k = k
            # slice & 分割
            take = use_k * num_institution_user
            sub = sub.iloc[:take]
            # 機関ごとに append (行順で後段再構築: ラベルの機関が連続)
            for i in range(use_k):
                seg = sub.iloc[i * num_institution_user: (i + 1) * num_institution_user]
                parts.append(seg)
        packed = pd.concat(parts, axis=0).reset_index(drop=True)
        return packed

    train_packed = _allocate(train_df)
    test_packed = _allocate(test_df)
    return train_packed, test_packed, final_num_institution


# ================================================== #
# 配列化 (Xs_train など) ※旧 train_test_split 相当
# ================================================== #
def to_institution_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_institution: int,
    num_institution_user: int,
    y_name: str = "target",
):
    X_train_df = train_df.drop(columns=[y_name])
    y_train_ser = train_df[y_name]
    X_test_df = test_df.drop(columns=[y_name])
    y_test_ser = test_df[y_name]

    Xs_train: List[np.ndarray] = []
    Xs_test: List[np.ndarray] = []
    ys_train: List[np.ndarray] = []
    ys_test: List[np.ndarray] = []

    total_needed = num_institution * num_institution_user
    if len(train_df) < total_needed or len(test_df) < total_needed:
        raise ValueError(
            f"サンプル不足: required per side={total_needed}, train={len(train_df)}, test={len(test_df)}"
        )

    for start in tqdm(range(0, num_institution * num_institution_user, num_institution_user), desc="institution slices"):
        end = start + num_institution_user
        Xs_train.append(X_train_df.iloc[start:end].to_numpy())
        ys_train.append(y_train_ser.iloc[start:end].to_numpy())
        Xs_test.append(X_test_df.iloc[start:end].to_numpy())
        ys_test.append(y_test_ser.iloc[start:end].to_numpy())

    return Xs_train, Xs_test, ys_train, ys_test


# ================================================== #
# メイン高水準 API
# ================================================== #
def prepare_institutional_dataset(df: pd.DataFrame, config: Config):
    """df (前処理済み) から機関配列を準備。

    even: ensure -> limit -> (even 内部で train/test) -> repack
    division: ensure -> limit -> basic train/test -> division repack (機関数調整で config 上書き)
    """
    ensure_institution_params(df, config)
    df_limited = limit_feature_columns(df, config)

    dist = getattr(config, "data_distribution", "even")
    label_col = getattr(config, "y_name", "target")
    dist = "ence"
    if dist == "division":
        # baseline split
        tr_df, te_df = basic_train_test_split(df_limited, config)
        tr_pack, te_pack, eff_inst = split_train_test_by_institution_division(
            tr_df,
            te_df,
            label_col=label_col,
            requested_num_institution=config.num_institution,
            num_institution_user=config.num_institution_user,
            random_state=getattr(config, "seed", 42),
        )
        config.num_institution = eff_inst  # 上書き
        train_packed, test_packed = tr_pack, te_pack
    else:
        train_packed, test_packed = split_train_test_by_institution_even(
            df_limited,
            config,
            random_state=getattr(config, "seed", 42),
        )

    Xs_train, Xs_test, ys_train, ys_test = to_institution_arrays(
        train_packed,
        test_packed,
        num_institution=config.num_institution,
        num_institution_user=config.num_institution_user,
        y_name=label_col,
    )
    return Xs_train, Xs_test, ys_train, ys_test, train_packed, test_packed


__all__ = [
    "ensure_institution_params",
    "limit_feature_columns",
    "basic_train_test_split",
    "split_train_test_by_institution_even",
    "split_train_test_by_institution_division",
    "to_institution_arrays",
    "prepare_institutional_dataset",
]
