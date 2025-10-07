"""institution_data.py
機関単位 (institution) のデータ分割ユーティリティ。

目的:
  - 既存 load_data 内にあった even 分割( train/test 同時割当 ) を外部化 (結果が変わらない)
  - division 方式 (1 機関 = 1 ラベル) も統合
  - 機関配列 (Xs_train, Xs_test, ys_train, ys_test) の生成

注意:
  - even 分割は従来の関数 split_train_test_by_institution_even のロジックをそのまま移植
    (全データを一度に扱い、train/test を同時に確保してから行順を機関ブロック化)
  - division 分割は一旦 stratified train/test split を行ったあとラベル毎に機関を構成
  - 乱数シード挙動も元実装に合わせ、even 方式内部 RNG の seed=42 デフォルトを維持
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

from config.config import Config


# ------------------------- 共通ヘルパ ------------------------- #
def _is_undefined(v) -> bool:
    return (
        v is None
        or (isinstance(v, str) and v.strip().lower() in ("undefined", "none", ""))
        or (isinstance(v, (int, float)) and v <= 0)
    )


def ensure_institution_params(df: pd.DataFrame, config: Config) -> None:
    """未設定パラメータを df に基づき安全に補完 (元 load_data の挙動を踏襲)"""
    if _is_undefined(config.feature_num):
        config.feature_num = len(df.columns) - 1
    if _is_undefined(config.dim_intermediate):
        config.dim_intermediate = config.feature_num - 1
    if _is_undefined(config.dim_integrate):
        config.dim_integrate = config.dim_intermediate
    if _is_undefined(config.num_institution_user):
        config.num_institution_user = 50
    if _is_undefined(config.num_institution):
        y = df["target"].to_numpy()
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if _is_undefined(config.num_institution_user) or config.num_institution_user < n_classes:
            config.num_institution_user = max(int(config.num_institution_user or 0), n_classes)
        max_by_total = len(df) // (2 * config.num_institution_user)
        max_by_class = int(np.min(counts) // 2)
        config.num_institution = max(1, min(max_by_total, max_by_class))


def limit_feature_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    y_name = config.y_name
    feature_cols = [c for c in df.columns if c != y_name]
    limited = feature_cols[: config.feature_num]
    final_cols = limited + [y_name]
    return df[final_cols].copy()


# ------------------------- even (joint) ------------------------- #
def even_joint_split(
    df: pd.DataFrame,
    label_col: str,
    num_institution: int,
    num_institution_user: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """元 load_data の split_train_test_by_institution_even をそのまま移植。
    df 全体から train/test の各機関へクラスを同時割当して行順を構成する。"""
    rng = np.random.default_rng(random_state)
    y = df[label_col].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    n_classes = classes.size
    n_per_side = num_institution * num_institution_user
    if 2 * n_per_side > len(df):
        raise ValueError(f"rows={len(df)} < needed(total)={2*n_per_side}")
    if num_institution_user < n_classes:
        raise ValueError(
            f"num_institution_user({num_institution_user}) < n_classes({n_classes}) -> 不可"
        )
    need_per_class = 2 * num_institution
    lack = {int(c): int(n) for c, n in zip(classes, counts) if n < need_per_class}
    if lack:
        raise ValueError(f"各クラス件数不足: {lack} (必要 {need_per_class})")

    N = len(df)
    all_idx = np.arange(N)
    train_bins = [[] for _ in range(num_institution)]
    test_bins = [[] for _ in range(num_institution)]
    used = set()

    for c in classes:
        idx_c = np.flatnonzero(y == c)
        rng.shuffle(idx_c)
        for i in range(num_institution):  # train 保証
            tr_id = idx_c[i]
            train_bins[i].append(tr_id)
            used.add(int(tr_id))
        for i in range(num_institution):  # test 保証
            te_id = idx_c[num_institution + i]
            test_bins[i].append(te_id)
            used.add(int(te_id))

    remain = np.array([i for i in all_idx if i not in used], dtype=int)
    rng.shuffle(remain)
    remain_train_need = n_per_side - sum(len(b) for b in train_bins)
    remain_test_need = n_per_side - sum(len(b) for b in test_bins)
    if remain.size < (remain_train_need + remain_test_need):
        raise ValueError("残りサンプル不足")
    train_pool = remain[:remain_train_need]
    test_pool = remain[remain_train_need : remain_train_need + remain_test_need]

    def distribute(pool, bins, target_size):
        pool = list(pool)
        p = 0
        for i in range(len(bins)):
            need = target_size - len(bins[i])
            if need <= 0:
                continue
            take = min(need, len(pool) - p)
            if take > 0:
                bins[i].extend(pool[p : p + take])
                p += take
        i = 0
        while any(len(b) < target_size for b in bins) and p < len(pool):
            if len(bins[i]) < target_size:
                bins[i].append(pool[p])
                p += 1
            i = (i + 1) % len(bins)
        if any(len(b) < target_size for b in bins):
            raise ValueError("target_size 充足失敗")
        return bins

    train_bins = distribute(train_pool, train_bins, num_institution_user)
    test_bins = distribute(test_pool, test_bins, num_institution_user)
    for b in train_bins:
        rng.shuffle(b)
    for b in test_bins:
        rng.shuffle(b)
    train_idx = np.concatenate([np.array(b, dtype=int) for b in train_bins])
    test_idx = np.concatenate([np.array(b, dtype=int) for b in test_bins])
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 内部検証
    def _check(side_df):
        for i in range(num_institution):
            part = side_df.iloc[i * num_institution_user : (i + 1) * num_institution_user]
            have = np.unique(part[label_col].to_numpy())
            if len(np.intersect1d(have, classes)) != n_classes:
                return False
        return True
    assert _check(train_df) and _check(test_df), "even 分割検証失敗"
    return train_df, test_df


# ------------------------- division ------------------------- #
def division_split(
    df: pd.DataFrame,
    config: Config,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """division: 1 機関 = 1 ラベル。train/test は 50:50 stratified 後に各ラベル内で機関割当。"""
    label_col = config.y_name
    stratify_arg = df[label_col] if getattr(config, "dataset", None) != "housing" else None
    train_df, test_df = sk_train_test_split(
        df,
        test_size=0.5,
        shuffle=True,
        random_state=getattr(config, "seed", random_state),
        stratify=stratify_arg,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    labels = np.array(sorted(train_df[label_col].unique()))
    L = len(labels)
    requested = config.num_institution
    if requested < L:
        final_inst = L
    else:
        k = requested // L
        final_inst = max(L, k * L)
    k = final_inst // L
    rng = np.random.default_rng(random_state)

    def _alloc(side_df: pd.DataFrame) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        for lab in labels:
            sub = side_df[side_df[label_col] == lab].sample(
                frac=1, random_state=rng.integers(0, 1_000_000)
            )
            need = k * config.num_institution_user
            if len(sub) < need:
                possible_k = max(1, len(sub) // max(1, config.num_institution_user))
                use_k = possible_k
            else:
                use_k = k
            take = use_k * config.num_institution_user
            sub = sub.iloc[:take]
            for i in range(use_k):
                seg = sub.iloc[
                    i * config.num_institution_user : (i + 1) * config.num_institution_user
                ]
                parts.append(seg)
        return pd.concat(parts, axis=0).reset_index(drop=True)

    train_packed = _alloc(train_df)
    test_packed = _alloc(test_df)
    config.num_institution = final_inst  # 調整結果を反映
    return train_packed, test_packed


# ------------------------- 配列化 ------------------------- #
def to_institution_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    y_name = config.y_name
    X_train_df = train_df.drop(columns=[y_name])
    y_train_ser = train_df[y_name]
    X_test_df = test_df.drop(columns=[y_name])
    y_test_ser = test_df[y_name]
    n_inst = config.num_institution
    per_inst = config.num_institution_user
    total_needed = n_inst * per_inst
    if len(train_df) < total_needed or len(test_df) < total_needed:
        raise ValueError(
            f"サンプル不足: need per side={total_needed}, train={len(train_df)}, test={len(test_df)}"
        )
    Xs_train: List[np.ndarray] = []
    Xs_test: List[np.ndarray] = []
    ys_train: List[np.ndarray] = []
    ys_test: List[np.ndarray] = []
    for start in range(0, n_inst * per_inst, per_inst):
        end = start + per_inst
        Xs_train.append(X_train_df.iloc[start:end].to_numpy())
        ys_train.append(y_train_ser.iloc[start:end].to_numpy())
        Xs_test.append(X_test_df.iloc[start:end].to_numpy())
        ys_test.append(y_test_ser.iloc[start:end].to_numpy())
    return Xs_train, Xs_test, ys_train, ys_test


# ------------------------- 高水準 API ------------------------- #
def prepare_institutional_dataset(
    df: pd.DataFrame, config: Config
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """前処理済み df から機関配列を構築 (even / division)"""
    ensure_institution_params(df, config)
    df_lim = limit_feature_columns(df, config)
    dist = getattr(config, "data_distribution", None)
    if dist == "division":
        train_df, test_df = division_split(df_lim, config)
    else:  # even (既定)
        train_df, test_df = even_joint_split(
            df_lim,
            label_col=config.y_name,
            num_institution=config.num_institution,
            num_institution_user=config.num_institution_user,
            random_state=42,  # 元実装互換
        )
    Xs_train, Xs_test, ys_train, ys_test = to_institution_arrays(train_df, test_df, config)
    return Xs_train, Xs_test, ys_train, ys_test, train_df, test_df


__all__ = [
    "ensure_institution_params",
    "limit_feature_columns",
    "even_joint_split",
    "division_split",
    "to_institution_arrays",
    "prepare_institutional_dataset",
]
