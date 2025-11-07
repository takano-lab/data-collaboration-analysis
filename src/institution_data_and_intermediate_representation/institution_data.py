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
    
    # --- inter_integ_dim_ratio 適用（未設定/None/空/False は 1 とみなす・一度だけ）---
    ratio_raw = getattr(config, "inter_integ_dim_ratio", 1)
    if ratio_raw in (None, "", False):
        ratio = 1.0
    else:
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError):
            ratio = 1.0
    if ratio != 1.0:
        orig = config.dim_integrate
        new_dim = int(round(orig * ratio))
        config.dim_integrate = new_dim

    if _is_undefined(getattr(config, "labeling_ratio", None)):
        config.labeling_ratio = 0.5
    if _is_undefined(getattr(config, "bias_ratio", None)):
        config.bias_ratio = 0.9

def limit_feature_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    y_name = config.y_name
    feature_cols = [c for c in df.columns if c != y_name]
    limited = feature_cols[: config.feature_num]
    final_cols = limited + [y_name]
    return df[final_cols].copy()


def _parse_ratio(raw_value, default: float) -> float:
    try:
        ratio = float(raw_value) if raw_value is not None else float(default)
    except (TypeError, ValueError):
        ratio = float(default)
    if np.isnan(ratio):
        ratio = float(default)
    ratio = max(0.0, min(1.0, ratio))
    return ratio


def apply_semi_supervision(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """even 分割後の学習データにラベル欠損を導入する."""
    df = train_df.copy()
    label_col = config.y_name
    ratio = _parse_ratio(getattr(config, "labeling_ratio", None), 0.1)
    if ratio >= 1.0:
        return df
    num_inst = int(getattr(config, "num_institution", 0) or 0)
    per_inst = int(getattr(config, "num_institution_user", 0) or 0)
    total_rows = len(df)
    if total_rows == 0:
        return df
    rng = np.random.default_rng(getattr(config, "seed", 42))
    keep_mask = np.zeros(total_rows, dtype=bool)
    if num_inst > 0 and per_inst > 0 and num_inst * per_inst <= total_rows:
        for inst_idx in range(num_inst):
            start = inst_idx * per_inst
            end = min(start + per_inst, total_rows)
            inst_indices = np.arange(start, end)
            if inst_indices.size == 0:
                continue
            inst_labels = df[label_col].to_numpy()[inst_indices]
            unique_labels_inst = np.unique(inst_labels)
            required_min = min(len(unique_labels_inst), inst_indices.size)
            keep_count = int(round(inst_indices.size * ratio))
            if ratio > 0.0 and keep_count == 0:
                keep_count = 1
            keep_count = max(keep_count, required_min)
            keep_count = min(keep_count, inst_indices.size)
            selected = []
            for lab in unique_labels_inst:
                lab_positions = inst_indices[inst_labels == lab]
                if lab_positions.size == 0:
                    continue
                selected.append(int(rng.choice(lab_positions, size=1)))
            selected = list(dict.fromkeys(selected))
            if len(selected) > keep_count:
                keep_count = len(selected)
            remaining_needed = keep_count - len(selected)
            if remaining_needed > 0:
                remaining_candidates = np.array([idx for idx in inst_indices if idx not in selected], dtype=int)
                if remaining_candidates.size > 0:
                    take = min(remaining_needed, remaining_candidates.size)
                    picked = rng.choice(remaining_candidates, size=take, replace=False)
                    selected.extend([int(x) for x in picked])
            keep_mask[selected] = True
    else:
        all_indices = np.arange(total_rows)
        keep_count = int(round(total_rows * ratio))
        if ratio > 0.0 and keep_count == 0:
            keep_count = 1
        if keep_count >= total_rows:
            keep_mask[:] = True
        else:
            chosen = rng.choice(all_indices, size=keep_count, replace=False)
            keep_mask[chosen] = True
    unlabeled_indices = np.flatnonzero(~keep_mask)
    if unlabeled_indices.size == 0:
        return df
    label_pos = df.columns.get_loc(label_col)
    df.iloc[unlabeled_indices, label_pos] = np.nan
    return df


def apply_bias_mixing(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """division 分割後の学習データにバイアス（他機関データ混入）を導入する."""
    df = train_df.copy()
    label_col = config.y_name
    num_inst = int(getattr(config, "num_institution", 0) or 0)
    per_inst = int(getattr(config, "num_institution_user", 0) or 0)
    ratio = _parse_ratio(getattr(config, "bias_ratio", None), 0.8)
    if (
        df.empty
        or num_inst <= 1
        or per_inst <= 0
        or ratio >= 1.0
    ):
        return df
    contam_count = int(round(per_inst * (1.0 - ratio)))
    if (1.0 - ratio) > 0.0 and contam_count == 0:
        contam_count = 1
    if contam_count <= 0:
        return df
    rng = np.random.default_rng(getattr(config, "seed", 42))
    original = df.copy()
    labels = sorted(original[label_col].unique().tolist())
    for inst_idx, lab in enumerate(labels):
        start = inst_idx * per_inst
        end = min(start + per_inst, len(df))
        if start >= end:
            continue
        replace_count = min(contam_count, end - start)
        if replace_count <= 0:
            continue
        block_indices = np.arange(start, end)
        replace_positions = rng.choice(block_indices, size=replace_count, replace=False)
        other_candidates = original.index[original[label_col] != lab].to_numpy()
        if other_candidates.size == 0:
            continue
        sampled = rng.choice(
            other_candidates,
            size=replace_count,
            replace=other_candidates.size < replace_count,
        )
        df.iloc[replace_positions] = original.loc[sampled].to_numpy()
    return df


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
    """division モード: ユーザ仕様に従い 1 機関 = 1 ラベルの訓練データ・共通テストデータを構築。

    仕様:
      1. 各ラベルごとにシャッフルし 50% を train, 残り 50% を test に分割
      2. 機関数 num_institution = ラベル数 (num_label)
      3. 各機関は単一ラベルのみを保持。 per-label の train サンプル最小数を max_num_institution_user とし
         max_num_institution_user < config.num_institution_user の場合のみ更新
      4. Xs_train / ys_train: 各機関 (= 各ラベル) から num_institution_user 件（ラベル固有）
      5. テスト: 全ラベルについて num_institution_user 件ずつ抽出し結合 (サイズ = num_label * num_institution_user)
         これを num_institution 回 (機関数) で共有する想定 → DataFrame としては 1 回分のみ保持し
         配列化段階で複製する（to_institution_arrays で特殊処理）
    戻り値 train_df は 機関順 (ラベル昇順) にラベルブロックが連結された形。
            test_df は 1 セット (全ラベル均等) のみ。
    """
    rng = np.random.default_rng(getattr(config, "seed", random_state))
    label_col = config.y_name

    # 1. ラベルごとにインデックス分割
    labels = np.array(sorted(df[label_col].unique()))
    train_parts: List[pd.DataFrame] = []
    test_label_parts: List[pd.DataFrame] = []
    for lab in labels:
        idx = np.flatnonzero(df[label_col].to_numpy() == lab)
        rng.shuffle(idx)
        half = len(idx) // 2  # 下側切り捨て
        train_idx = idx[:half]
        test_idx = idx[half:]
        train_parts.append(df.iloc[train_idx])
        test_label_parts.append(df.iloc[test_idx])

    # 2. 機関数設定
    num_label = len(labels)
    config.num_institution = num_label

    # 3. 機関当たりサンプル数の上限 (train 側最小) を算出
    train_counts = [len(p) for p in train_parts]
    max_num_institution_user = int(min(train_counts)) if train_counts else 0
    if max_num_institution_user <= 0:
        raise ValueError("division_split: 有効な train サンプルがありません")
    if max_num_institution_user < config.num_institution_user:
        config.num_institution_user = max_num_institution_user

    per_inst = config.num_institution_user

    # 4. train_df 構築（ラベル昇順で per_inst 件ずつ）
    trimmed_train_blocks = []
    for lab, part in zip(labels, train_parts):
        if len(part) < per_inst:
            # 必要件数不足 → 上で per_inst 調整済のはずだがガード
            raise ValueError(f"label {lab}: train サンプル不足 {len(part)} < {per_inst}")
        trimmed_train_blocks.append(part.iloc[:per_inst].copy())
    train_df = pd.concat(trimmed_train_blocks, axis=0).reset_index(drop=True)

    # 5. test base セット構築
    test_counts = [len(p) for p in test_label_parts]
    min_test = min(test_counts) if test_counts else 0
    if min_test < per_inst:
        # テスト側が足りない場合は per_inst をさらに下げる (仕様上明記なしだが安全策)
        per_inst = min_test
        config.num_institution_user = per_inst
    trimmed_test_blocks = []
    for lab, part in zip(labels, test_label_parts):
        if len(part) < per_inst:
            raise ValueError(f"label {lab}: test サンプル不足 {len(part)} < {per_inst}")
        trimmed_test_blocks.append(part.iloc[:per_inst].copy())
    test_base_df = pd.concat(trimmed_test_blocks, axis=0).reset_index(drop=True)

    # test_df は 1 セットのみ（配列化で複製）
    test_df = test_base_df
    
    return train_df, test_df


# ------------------------- 配列化 ------------------------- #
def to_institution_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """DataFrame から機関配列へ変換。

    division モードの場合:
      - train_df はラベル毎ブロック (各 block = num_institution_user 行) が並ぶ
      - test_df は全ラベル * num_institution_user 行の 1 セットのみ
      - Xs_test / ys_test はこの 1 セットを機関数分複製（各機関同一テスト集合）
    even モードの場合は従来どおり連続スライス。
    """
    y_name = config.y_name
    n_inst = config.num_institution
    per_inst = config.num_institution_user
    Xs_train: List[np.ndarray] = []
    Xs_test: List[np.ndarray] = []
    ys_train: List[np.ndarray] = []
    ys_test: List[np.ndarray] = []

    dist = getattr(config, "data_distribution", None)

    if dist in ("division", "bias"):
        # --- train blocks (one block per label) ---
        y_train_ser = train_df[y_name]
        X_train_df = train_df.drop(columns=[y_name])
        labels = sorted(y_train_ser.unique())
        if len(train_df) != n_inst * per_inst:
            raise ValueError(
                f"division train_df 行数不整合: {len(train_df)} != {n_inst * per_inst}"
            )
        for i, lab in enumerate(labels):
            block = X_train_df.iloc[i * per_inst : (i + 1) * per_inst]
            y_block = y_train_ser.iloc[i * per_inst : (i + 1) * per_inst]
            if dist == "division" and len(set(y_block)) != 1:
                raise ValueError("division train block に単一ラベル以外が含まれています")
            Xs_train.append(block.to_numpy())
            ys_train.append(y_block.to_numpy())

        # --- test (共通セット) ---
        y_test_ser = test_df[y_name]
        X_test_df = test_df.drop(columns=[y_name])
        # ラベル均等性 (警告のみ) ※ 厳密には各ラベル per_inst 行が理想
        counts = y_test_ser.value_counts()
        if (counts < per_inst).any():
            self.logger.warning(
                f"[WARN] division test: 一部ラベル不足 counts={counts.to_dict()} < {per_inst}. 利用可能件数で進行"
            )
        # そのままの順序で base セット化
        X_base = X_test_df.to_numpy()
        y_base = y_test_ser.to_numpy()
        for _ in range(n_inst):
            Xs_test.append(X_base.copy())
            ys_test.append(y_base.copy())
        return Xs_train, Xs_test, ys_train, ys_test

    # ---- even (従来) ----
    y_train_ser = train_df[y_name]
    X_train_df = train_df.drop(columns=[y_name])
    y_test_ser = test_df[y_name]
    X_test_df = test_df.drop(columns=[y_name])
    total_needed = n_inst * per_inst
    if len(train_df) < total_needed or len(test_df) < total_needed:
        raise ValueError(
            f"サンプル不足: need per side={total_needed}, train={len(train_df)}, test={len(test_df)}"
        )
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
    elif dist == "bias":
        train_df, test_df = division_split(df_lim, config)
        train_df = apply_bias_mixing(train_df, config)
    elif dist == "semi":
        train_df, test_df = even_joint_split(
            df_lim,
            label_col=config.y_name,
            num_institution=config.num_institution,
            num_institution_user=config.num_institution_user,
            random_state=42,
        )
        train_df = apply_semi_supervision(train_df, config)
    else:  # even
        train_df, test_df = even_joint_split(
            df_lim,
            label_col=config.y_name,
            num_institution=config.num_institution,
            num_institution_user=config.num_institution_user,
            random_state=42,  # ベースライン
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
