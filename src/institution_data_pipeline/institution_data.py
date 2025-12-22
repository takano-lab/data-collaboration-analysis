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

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from config.config import Config

logger = logging.getLogger(__name__)


# ------------------------- 共通ヘルパ ------------------------- #
def _is_undefined(v) -> bool:
    return (
        v is None
        or (isinstance(v, str) and v.strip().lower() in ("undefined", "none", ""))
        or (isinstance(v, (int, float)) and v <= 0)
    )


def _reserve_smote_anchor_data(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    When anchor_method == \"smote\", reserve up to public_anchor_num rows as
    public data for SMOTE anchors, trying to balance label distribution.

    public_anchor_num defaults to num_institution_user when not specified.
    Returns (anchor_df, remaining_df).
    """
    anchor_method = getattr(config, "anchor_method", None)
    label_col = getattr(config, "y_name", "target")
    if anchor_method != "smote" or label_col not in df.columns or df.empty:
        return df.iloc[0:0].copy(), df

    raw_target = getattr(config, "public_anchor_num", None)
    if raw_target in (None, "", False):
        raw_target = getattr(config, "num_institution_user", 0)
    try:
        total_anchor = int(raw_target or 0)
    except (TypeError, ValueError):
        total_anchor = 0
    if total_anchor <= 0:
        return df.iloc[0:0].copy(), df
    # If reservation would exhaust the dataset, skip reserving to keep training data intact.
    if total_anchor >= len(df):
        print("[_reserve_smote_anchor_data] Requested public_anchor_num >= dataset size; skipping public reserve.")
        return df.iloc[0:0].copy(), df
    total_anchor = min(total_anchor, len(df))

    rng = np.random.default_rng(getattr(config, "seed", 42))
    y = df[label_col].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_classes == 0:
        return df.iloc[0:0].copy(), df

    base = total_anchor // n_classes
    rem = total_anchor % n_classes

    per_class_indices = {}
    for lab in classes:
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        per_class_indices[lab] = idx

    selected: list[int] = []

    # First pass: try to allocate roughly equal counts per class
    for i, lab in enumerate(classes):
        want = base + (1 if i < rem else 0)
        idx = per_class_indices[lab]
        take = min(want, idx.size)
        if take > 0:
            selected.extend(idx[:take])
            per_class_indices[lab] = idx[take:]

    # Remove duplicates while preserving order
    selected = list(dict.fromkeys(selected))

    # If still short, fill from remaining pool
    if len(selected) < total_anchor:
        remaining_lists = [idx for idx in per_class_indices.values() if idx.size > 0]
        if remaining_lists:
            all_remaining = np.concatenate(remaining_lists)
            need = min(total_anchor - len(selected), all_remaining.size)
            if need > 0:
                extra = rng.choice(all_remaining, size=need, replace=False)
                selected.extend(int(i) for i in extra)

    if not selected:
        return df.iloc[0:0].copy(), df

    selected_arr = np.array(sorted(set(selected)), dtype=int)
    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[selected_arr] = False

    anchor_df = df.iloc[selected_arr].reset_index(drop=True)
    remaining_df = df.iloc[keep_mask].reset_index(drop=True)
    return anchor_df, remaining_df


def _distribute_evenly(total: int, buckets: int, *, require_positive: bool = False) -> List[int]:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    if require_positive and total < buckets:
        raise ValueError(
            f"total({total}) < buckets({buckets}) -> 均等割が成立しません"
        )
    base = total // buckets
    rem = total % buckets
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def ensure_institution_params(df: pd.DataFrame, config: Config) -> None:
    """未設定パラメータを df に基づき安全に補完 (元 load_data の挙動を踏襲)"""
    if _is_undefined(config.feature_num):
        feature_num = len(df.columns) - 1
    if _is_undefined(config.dim_intermediate):
        config.dim_intermediate = feature_num - 1
    if _is_undefined(config.dim_integrate):
        config.dim_integrate = config.dim_intermediate
    if _is_undefined(config.num_institution_user):
        config.num_institution_user = 50
    y = df["target"].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # 'label_num' 特殊指定は後段（division_split など）で処理するためここでは数値確定しない
    if _is_undefined(config.num_institution):
        if _is_undefined(config.num_institution_user) or config.num_institution_user < n_classes:
            config.num_institution_user = max(int(config.num_institution_user or 0), n_classes)
        max_by_total = len(df) // (2 * config.num_institution_user)
        max_by_class = int(np.min(counts) // 2) if counts.size else 1
        config.num_institution = max(1, min(max_by_total, max_by_class))
    elif isinstance(getattr(config, 'num_institution'), str) and str(getattr(config, 'num_institution')).lower() == 'label_num':
        # 何もしない（division/bias モードでラベル数に置換）
        pass
    
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

    # if _is_undefined(getattr(config, "labeling_ratio", None)):
    #     config.labeling_ratio = 0.5
    # if _is_undefined(getattr(config, "bias_ratio", None)):
    #     config.bias_ratio = 0.9

    # Ensure total rows are sufficient for even split; adjust downward if needed
    total_rows = len(df)
    raw_inst = getattr(config, "num_institution", 1)
    if isinstance(raw_inst, str) and raw_inst.lower() == "label_num":
        num_inst = n_classes
    else:
        try:
            num_inst = int(raw_inst or 1)
        except (TypeError, ValueError):
            num_inst = 1
    num_inst = max(1, num_inst)
    try:
        num_inst_user = int(getattr(config, "num_institution_user", n_classes) or n_classes)
    except (TypeError, ValueError):
        num_inst_user = n_classes
    num_inst_user = max(n_classes, num_inst_user)

    max_per_user = total_rows // (2 * num_inst) if num_inst > 0 else 0
    if max_per_user < n_classes:
        # Reduce institutions to increase per-user capacity
        feasible_inst = total_rows // (2 * n_classes)
        if feasible_inst >= 1:
            num_inst = feasible_inst
            max_per_user = total_rows // (2 * num_inst)
        else:
            max_per_user = n_classes
    if max_per_user > 0:
        if num_inst_user > max_per_user:
            num_inst_user = max_per_user
        if num_inst_user < n_classes and max_per_user >= n_classes:
            num_inst_user = n_classes
    else:
        num_inst = 1
        num_inst_user = n_classes

    needed_total = 2 * num_inst * num_inst_user
    if needed_total > total_rows and num_inst > 1:
        max_inst = max(1, total_rows // (2 * num_inst_user))
        num_inst = min(num_inst, max_inst)
    needed_total = 2 * num_inst * num_inst_user
    if needed_total > total_rows and num_inst_user > n_classes:
        num_inst_user = max(n_classes, total_rows // (2 * num_inst))

    # 'label_num' の場合はここで上書きしない
    if not (isinstance(getattr(config, 'num_institution'), str) and str(getattr(config, 'num_institution')).lower() == 'label_num'):
        config.num_institution = max(1, num_inst)
    config.num_institution_user = max(n_classes, num_inst_user)

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
    # 機関ブロック単位で処理（num_institution がラベル数と異なるケース対応）
    for inst_idx in range(num_inst):
        start = inst_idx * per_inst
        end = min(start + per_inst, len(df))
        if end - start <= 0:
            continue
        block = original.iloc[start:end]
        block_labels = block[label_col].unique()
        if block_labels.size == 0:
            continue
        lab = block_labels[0]
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
        # データが足りない場合は num_institution_user を自動的に縮小して続行
        import warnings
        max_per_inst = max(1, len(df) // (2 * max(1, num_institution)))
        warnings.warn(
            f"rows={len(df)} < needed(total)={2*n_per_side}; "
            f"num_institution_user を {num_institution_user}->{max_per_inst} に縮小します",
            RuntimeWarning,
        )
        num_institution_user = max_per_inst
        n_per_side = num_institution * num_institution_user
        if n_per_side <= 0:
            raise ValueError(f"rows={len(df)} が少なすぎます (num_institution={num_institution})")
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

    # 2. 機関数設定 (指定可能)
    num_label = len(labels)
    desired_raw = getattr(config, "num_institution", None)
    if isinstance(desired_raw, str):
        if desired_raw.lower() == "label_num":
            n_inst = num_label
        else:
            try:
                n_inst = int(desired_raw)
            except ValueError:
                n_inst = num_label
    elif isinstance(desired_raw, (int, np.integer)) and int(desired_raw) > 0:
        n_inst = int(desired_raw)
    else:
        n_inst = num_label
    # 最低 1
    n_inst = max(1, n_inst)
    config.num_institution = n_inst

    # ラベル割当マップ (機関 i -> labels[i % num_label])
    institution_labels = [labels[i % num_label] for i in range(n_inst)]

    # 3. 機関当たりサンプル数上限 (ラベル毎の train サンプルを割当回数で割る)
    train_counts_map = {lab: len(part) for lab, part in zip(labels, train_parts)}
    occ_map = {lab: institution_labels.count(lab) for lab in labels}
    per_inst_limit_candidates = []
    for lab in labels:
        occ = occ_map[lab]
        if occ <= 0:
            continue
        per_inst_limit_candidates.append(train_counts_map[lab] // occ)
    max_num_institution_user = int(min(per_inst_limit_candidates)) if per_inst_limit_candidates else 0
    if max_num_institution_user <= 0:
        raise ValueError("division_split: 各ラベルの学習サンプルが機関割当数に足りません")
    if max_num_institution_user < config.num_institution_user:
        config.num_institution_user = max_num_institution_user

    per_inst = config.num_institution_user

    # 4. train_df 構築（機関順で per_inst 件ずつ）
    # ラベルごとに使用済みオフセットを管理
    train_parts_dict = {lab: part.reset_index(drop=True) for lab, part in zip(labels, train_parts)}
    used_offset = {lab: 0 for lab in labels}
    train_blocks = []
    for inst_lab in institution_labels:
        part = train_parts_dict[inst_lab]
        start = used_offset[inst_lab]
        end = start + per_inst
        if end > len(part):
            raise ValueError(f"label {inst_lab}: 割当不足 {len(part)} < {end}")
        block = part.iloc[start:end].copy()
        train_blocks.append(block)
        used_offset[inst_lab] = end
    train_df = pd.concat(train_blocks, axis=0).reset_index(drop=True)

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
        trimmed_test_blocks.append(part.reset_index(drop=True).iloc[:per_inst].copy())
    test_base_df = pd.concat(trimmed_test_blocks, axis=0).reset_index(drop=True)

    # test_df は 1 セットのみ（配列化で複製）
    test_df = test_base_df
    
    return train_df, test_df


def dirichlet_split(
    df: pd.DataFrame,
    config: Config,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dirichlet ベースの non-IID 分割 (機関ごとにラベル比率をサンプリング)."""
    label_col = config.y_name
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col} が存在しません")

    labels = np.array(sorted(df[label_col].unique()))
    num_label = len(labels)
    if num_label == 0:
        raise ValueError("dirichlet_split: ラベルが存在しません")

    try:
        num_inst = int(getattr(config, "num_institution", num_label) or num_label)
    except (TypeError, ValueError):
        num_inst = num_label
    try:
        per_inst = int(getattr(config, "num_institution_user", 0) or 0)
    except (TypeError, ValueError):
        per_inst = 0
    if num_inst <= 0 or per_inst <= 0:
        raise ValueError("dirichlet_split: num_institution と num_institution_user は正の整数が必要です")

    total_per_side = num_inst * per_inst

    beta_raw = getattr(config, "non_iid_beta", None)
    try:
        beta = float(beta_raw)
    except (TypeError, ValueError):
        beta = None
    if beta is None or beta <= 0.0:
        raise ValueError("dirichlet_split: config.non_iid_beta (>0) が必要です")

    rng = np.random.default_rng(getattr(config, "seed", random_state))
    y_arr = df[label_col].to_numpy()

    # --- テストデータをラベル均等に確保 ---
    test_parts: List[pd.DataFrame] = []
    pool_indices: dict[int, np.ndarray] = {}
    pool_ptr: dict[int, int] = {}
    for lab in labels:
        lab_indices = np.flatnonzero(y_arr == lab)
        if lab_indices.size == 0:
            raise ValueError(f"dirichlet_split: label {lab} のデータが存在しません")
        rng.shuffle(lab_indices)
        if lab_indices.size < per_inst:
            raise ValueError(f"dirichlet_split: label {lab} のテスト用サンプルが不足しています")
        test_idx = lab_indices[:per_inst]
        test_parts.append(df.iloc[test_idx])
        remaining = lab_indices[per_inst:]
        pool_indices[int(lab)] = remaining
        pool_ptr[int(lab)] = 0

    remaining_total = sum(arr.size for arr in pool_indices.values())
    if remaining_total < total_per_side:
        raise ValueError(
            "dirichlet_split: テスト確保後のデータが訓練に必要な件数を下回っています"
        )

    train_blocks: List[pd.DataFrame] = []
    for inst_idx in range(num_inst):
        probs = rng.dirichlet(np.full(num_label, beta))
        counts = rng.multinomial(per_inst, probs)
        block_parts: List[pd.DataFrame] = []
        block_size = 0

        for lab_idx, lab in enumerate(labels):
            need = int(counts[lab_idx])
            pool = pool_indices[int(lab)]
            ptr = pool_ptr[int(lab)]
            available = pool.size - ptr
            take = min(need, available)
            if take > 0:
                idx_slice = pool[ptr : ptr + take]
                block_parts.append(df.iloc[idx_slice])
                pool_ptr[int(lab)] += take
                block_size += take

        while block_size < per_inst:
            deficit = per_inst - block_size
            available_labels = [lab for lab in labels if pool_ptr[int(lab)] < pool_indices[int(lab)].size]
            if not available_labels:
                raise ValueError("dirichlet_split: 訓練データが不足しています")
            # 余りが最も多いラベルから追加で取得
            lab = max(
                available_labels,
                key=lambda l: pool_indices[int(l)].size - pool_ptr[int(l)],
            )
            pool = pool_indices[int(lab)]
            ptr = pool_ptr[int(lab)]
            take = min(deficit, pool.size - ptr)
            idx_slice = pool[ptr : ptr + take]
            block_parts.append(df.iloc[idx_slice])
            pool_ptr[int(lab)] += take
            block_size += take

        inst_df = pd.concat(block_parts, axis=0).reset_index(drop=True)
        if len(inst_df) != per_inst:
            raise ValueError(
                f"dirichlet_split: 機関 {inst_idx} のサンプル数 {len(inst_df)} != {per_inst}"
            )
        # ブロック内をランダムシャッフル
        perm_seed = int(rng.integers(0, 2**32 - 1))
        inst_df = inst_df.sample(frac=1.0, random_state=perm_seed).reset_index(drop=True)
        train_blocks.append(inst_df)

    for idx, block in enumerate(train_blocks):
        counts = block[label_col].value_counts(normalize=True) * 100
        ratios = ", ".join(f"{lab}:{count:.2f}%" for lab, count in counts.items())
        print(f"[dirichlet] inst {idx}: {ratios}")


    train_df = pd.concat(train_blocks, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)

    if len(train_df) != total_per_side:
        raise ValueError(
            f"dirichlet_split: train_df 行数 {len(train_df)} != {total_per_side}"
        )
    if len(test_df) != num_label * per_inst:
        raise ValueError(
            f"dirichlet_split: test_df 行数 {len(test_df)} != {num_label * per_inst}"
        )

    return train_df, test_df


def dirichlet_label_fixed_split(
    df: pd.DataFrame,
    config: Config,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dirichlet（固定 Ni）モード: 各ラベルの総数 Ni を均等にしつつ機関ごとの件数は可変にする."""
    label_col = config.y_name
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col} が存在しません")

    labels = np.array(sorted(df[label_col].unique()))
    num_label = len(labels)
    if num_label == 0:
        raise ValueError("dirichlet_label_fixed_split: ラベルが存在しません")

    try:
        num_inst = int(getattr(config, "num_institution", num_label) or num_label)
    except (TypeError, ValueError):
        num_inst = num_label
    try:
        per_inst = int(getattr(config, "num_institution_user", 0) or 0)
    except (TypeError, ValueError):
        per_inst = 0
    if num_inst <= 0 or per_inst <= 0:
        raise ValueError("dirichlet_label_fixed_split: num_institution と num_institution_user は正の整数が必要です")

    total_per_side = num_inst * per_inst
    if total_per_side % num_label != 0:
        raise ValueError(
            "dirichlet_label_fixed_split: num_institution*num_institution_user は num_label で割り切れる必要があります"
        )
    per_label_train = total_per_side // num_label
    per_label_test = per_label_train
    if per_label_train <= 0:
        raise ValueError("dirichlet_label_fixed_split: Ni が 0 以下です。パラメータを見直してください")

    beta_raw = getattr(config, "non_iid_beta", None)
    try:
        beta = float(beta_raw)
    except (TypeError, ValueError):
        beta = None
    if beta is None or beta <= 0.0:
        raise ValueError("dirichlet_label_fixed_split: config.non_iid_beta (>0) が必要です")

    rng = np.random.default_rng(getattr(config, "seed", random_state))
    y_arr = df[label_col].to_numpy()
    value_counts = pd.Series(y_arr).value_counts()
    counts_per_label = value_counts.reindex(labels, fill_value=0).to_numpy()
    needed = per_label_train + per_label_test
    lacking = labels[counts_per_label < needed]
    if lacking.size > 0:
        details = {
            int(lab): {
                "available": int(counts_per_label[i]),
                "needed": int(needed),
            }
            for i, lab in enumerate(labels)
            if counts_per_label[i] < needed
        }
        raise ValueError(f"dirichlet_label_fixed_split: ラベル別サンプル不足 {details}")

    train_parts_by_label: List[pd.DataFrame] = []
    test_parts_by_label: List[pd.DataFrame] = []
    assign_matrix = np.zeros((num_label, num_inst), dtype=int)

    for lab_idx, lab in enumerate(labels):
        lab_indices = np.flatnonzero(y_arr == lab)
        rng.shuffle(lab_indices)
        train_idx = lab_indices[:per_label_train]
        test_idx = lab_indices[per_label_train : per_label_train + per_label_test]
        train_parts_by_label.append(df.iloc[train_idx].reset_index(drop=True))
        test_parts_by_label.append(df.iloc[test_idx].reset_index(drop=True))
        probs = rng.dirichlet(np.full(num_inst, beta))
        counts = rng.multinomial(per_label_train, probs)
        assign_matrix[lab_idx] = counts

    inst_lengths = assign_matrix.sum(axis=0)
    zero_indices = np.flatnonzero(inst_lengths == 0)
    while zero_indices.size > 0:
        inst_idx = int(zero_indices[0])
        donor_candidates = np.flatnonzero(inst_lengths > 1)
        if donor_candidates.size == 0:
            raise ValueError("dirichlet_label_fixed_split: 再割当のための十分なサンプルがありません")
        donor_idx = int(donor_candidates[np.argmax(inst_lengths[donor_candidates])])
        donor_labels = np.flatnonzero(assign_matrix[:, donor_idx] > 0)
        if donor_labels.size == 0:
            raise ValueError("dirichlet_label_fixed_split: ドナー機関に割当済みラベルがありません")
        label_idx = int(donor_labels[np.argmax(assign_matrix[donor_labels, donor_idx])])
        assign_matrix[label_idx, donor_idx] -= 1
        assign_matrix[label_idx, inst_idx] += 1
        inst_lengths[donor_idx] -= 1
        inst_lengths[inst_idx] += 1
        zero_indices = np.flatnonzero(inst_lengths == 0)

    train_blocks_per_inst: List[List[pd.DataFrame]] = [[] for _ in range(num_inst)]
    for lab_idx, part in enumerate(train_parts_by_label):
        offset = 0
        for inst_idx in range(num_inst):
            take = int(assign_matrix[lab_idx, inst_idx])
            if take <= 0:
                continue
            next_offset = offset + take
            block = part.iloc[offset:next_offset]
            if len(block) != take:
                raise ValueError("dirichlet_label_fixed_split: ラベルブロック切り出しに失敗しました")
            train_blocks_per_inst[inst_idx].append(block)
            offset = next_offset
        if offset != per_label_train:
            raise ValueError("dirichlet_label_fixed_split: ラベル別サンプル消費量が一致しません")

    train_blocks: List[pd.DataFrame] = []
    for inst_idx, blocks in enumerate(train_blocks_per_inst):
        if blocks:
            inst_df = pd.concat(blocks, axis=0).reset_index(drop=True)
        else:
            inst_df = df.iloc[0:0].copy()
        train_blocks.append(inst_df)

    train_df = pd.concat(train_blocks, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts_by_label, axis=0).reset_index(drop=True)

    if len(train_df) != total_per_side:
        raise ValueError(
            f"dirichlet_label_fixed_split: train_df 行数 {len(train_df)} != {total_per_side}"
        )
    if len(test_df) != total_per_side:
        raise ValueError(
            f"dirichlet_label_fixed_split: test_df 行数 {len(test_df)} != {total_per_side}"
        )

    config.dirichlet_label_fixed_sizes = [int(v) for v in inst_lengths.tolist()]
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

    if dist == "fixed":
        # train_df / test_df には既に inst_id 等の列は含まれていない想定なので、
        # 分割は prepare_institutional_dataset 側で完了しており、ここでは even と同様に扱う。
        pass  # 下の even ロジックをそのまま使う

    if dist == "dirichlet_label_fixed":
        lengths = getattr(config, "dirichlet_label_fixed_sizes", None)
        if not isinstance(lengths, list) or len(lengths) != n_inst:
            raise ValueError("dirichlet_label_fixed: 機関ごとの件数情報が不足しています")
        if sum(lengths) != len(train_df):
            raise ValueError(
                f"dirichlet_label_fixed: 行数不一致 len(train_df)={len(train_df)} vs lengths={sum(lengths)}"
            )
        y_train_ser = train_df[y_name]
        X_train_df = train_df.drop(columns=[y_name])
        start = 0
        for inst_idx, size in enumerate(lengths):
            end = start + size
            block = X_train_df.iloc[start:end]
            y_block = y_train_ser.iloc[start:end]
            Xs_train.append(block.to_numpy())
            ys_train.append(y_block.to_numpy())
            start = end
        if start != len(train_df):
            raise ValueError("dirichlet_label_fixed: train_df スキャン長と件数が一致しません")
        y_test_ser = test_df[y_name]
        X_test_df = test_df.drop(columns=[y_name])
        if len(test_df) != n_inst * per_inst:
            raise ValueError(
                f"dirichlet_label_fixed: test_df 行数 {len(test_df)} != {n_inst * per_inst}"
            )
        X_base = X_test_df.to_numpy()
        y_base = y_test_ser.to_numpy()
        for _ in range(n_inst):
            Xs_test.append(X_base.copy())
            ys_test.append(y_base.copy())
        return Xs_train, Xs_test, ys_train, ys_test

    if dist in ("division", "bias", "dirichlet"):
        # --- train blocks (機関順で per_inst 件ずつ) ---
        print(f"[to_institution_arrays] division モード配列化: n_inst={n_inst}, per_inst={per_inst}")
        y_train_ser = train_df[y_name]
        X_train_df = train_df.drop(columns=[y_name])
        if len(train_df) != n_inst * per_inst:
            raise ValueError(
                f"{dist} train_df 行数不整合: {len(train_df)} != {n_inst * per_inst}"
            )
        for i in range(n_inst):
            start = i * per_inst
            end = start + per_inst
            block = X_train_df.iloc[start:end]
            y_block = y_train_ser.iloc[start:end]
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
            logger.warning(
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
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """前処理済み df から機関配列を構築 (even / division)"""
    ensure_institution_params(df, config)

    public_anchor = np.empty((0, 0), dtype=float)
    public_anchor_y = np.empty((0,), dtype=float)

    # If SMOTE anchors are requested, reserve public data for them first.
    anchor_method = getattr(config, "anchor_method", None)
    df_lim = df
    if anchor_method == "smote":
        anchor_df, df_lim = _reserve_smote_anchor_data(df_lim, config)
        if not anchor_df.empty:
            label_col = getattr(config, "y_name", "target")
            if label_col in anchor_df.columns:
                public_anchor = anchor_df.drop(columns=[label_col]).to_numpy()
                public_anchor_y = anchor_df[label_col].to_numpy()
    dist = getattr(config, "data_distribution", None)
    print(f"[prepare_institutional_dataset] data_distribution={dist}")
    # even / semi モードで 'label_num' 指定が来た場合はここで数値化
    if dist not in ("division", "bias") and isinstance(getattr(config, 'num_institution'), str) and str(getattr(config, 'num_institution')).lower() == 'label_num':
        labels_unique = df_lim[config.y_name].unique()
        config.num_institution = len(labels_unique)
    if dist == "division":
        train_df, test_df = division_split(df_lim, config)
    elif dist == "bias":
        train_df, test_df = division_split(df_lim, config)
        train_df = apply_bias_mixing(train_df, config)
    elif dist == "dirichlet":
        train_df, test_df = dirichlet_split(df_lim, config)
    elif dist == "dirichlet_label_fixed":
        train_df, test_df = dirichlet_label_fixed_split(df_lim, config)
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
    return Xs_train, Xs_test, ys_train, ys_test, train_df, test_df, public_anchor, public_anchor_y


__all__ = [
    "ensure_institution_params",
    "even_joint_split",
    "division_split",
    "dirichlet_split",
    "dirichlet_label_fixed_split",
    "to_institution_arrays",
    "prepare_institutional_dataset",
]
