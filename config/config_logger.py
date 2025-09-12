# config_logger.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Union

CsvPath = Union[str, Path]


# ------------------------------------------------------------
# 1. Config を丸ごと 1 行で記録
# ------------------------------------------------------------
def record_config(cfg: "Config", csv_path: CsvPath) -> None:
    """
    Config インスタンスの全フィールドを 1 行で CSV に追記する。
    ヘッダーに無いキーは自動で列追加。

    Parameters
    ----------
    cfg : Config
        保存したい設定オブジェクト
    csv_path : str | Path
        CSV ファイルの保存先（存在しなければ自動作成）
    """
    exclude = set(["output_path", "input_path", "name", "seed", "y_name", "eigenvalues", "nl_gammas", "g_abs_sum", "nl_lambda_opt", "nl_gamma_opt", "plot_name", "lambda_gen_eigen", "objective_direction_ratio", "lambda_pred", "lambda_offdiag", "semi_integ", "orth_ver", "f_seed_2", "jreg_gep", "g_norm_val_gep", "sum_objective_function", "g_mean_var", "g_condition_number", "集中解析", "個別解析", "now"])
    row = {k: v for k, v in cfg.__dict__.items() if k not in exclude}
    _append_row(row, csv_path)


# ------------------------------------------------------------
# 2. 任意の (カラム, 値) を 1 行で記録
# ------------------------------------------------------------
def record_value(column: str, value: Any, csv_path: CsvPath) -> None:
    """
    単一のカラムに値をセットした 1 行を追記。
    既存カラムでなければヘッダーを拡張する。

    Examples
    --------
    record_value("dataset", "har", "output/timing.csv")
    """
    _append_row({column: value}, csv_path)


# ------------------------------------------------------------
# ヘッダー拡張 & 追記を共通化した内部関数
# ------------------------------------------------------------
def _append_row(row_dict: Dict[str, Any], csv_path: CsvPath) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 既存ヘッダー読込 or 新規作成 --------------------------
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # 先頭行
    else:
        header = []

    # --- 新しい列があればヘッダー拡張 --------------------------
    new_cols = [k for k in row_dict if k not in header]
    if new_cols:
        header.extend(new_cols)
        _rewrite_header(csv_path, header)

    # --- ヘッダー順に行を作成して追記 -------------------------
    row = [row_dict.get(col, "") for col in header]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _rewrite_header(csv_path: Path, header: list[str]) -> None:
    """
    ヘッダーを書き換えてファイルを更新。
    既存データ行はそのまま保持し、改行が欠けないようにする。
    """
    if not csv_path.exists():
        # 新規作成
        csv_path.write_text(",".join(header) + "\n", encoding="utf-8")
        return

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[0] = ",".join(header)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------------------------------------------
# 1. Config → output_path の timing.csv へ保存
# ------------------------------------------------------------
# config_logger.py  ※差分だけ

def record_config_to_cfg(cfg: "Config", filename: str = "output.csv") -> None:
    """
    cfg.output_path / filename に Config の内容を 1 行で追記。
    """
    assert hasattr(cfg, "output_path") and cfg.output_path, "Config に output_path が必要です"
    csv_path = Path(cfg.output_path) / filename
    record_config(cfg, csv_path)

def record_value_to_cfg(
    cfg: "Config",
    column: str,
    value: Any,
    filename: str = "output.csv",
) -> None:
    """
    cfg.output_path/filename に (column,value) を格納。
    - 列が無ければ追加
    - 最後の行のその列が "" ならそこを書き換え
    - 既に値が入っていれば新しい行として追記
    """
    from pathlib import Path
    import csv

    assert hasattr(cfg, "output_path") and cfg.output_path, "Config に output_path が必要です"
    csv_path = Path(cfg.output_path) / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) ファイル読み込み (ヘッダーも) -----------------
    rows: list[list[str]]
    if csv_path.exists():
        rows = csv_path.read_text(encoding="utf-8").splitlines()
        rows = [r.split(",") for r in rows]
        header = rows[0]
    else:
        header, rows = [], []

    # ---------------- 2) ヘッダー拡張 -----------------
    if column not in header:
        header.append(column)
        # 古い行の足りない列を空白で埋める
        for r in rows[1:]:
            r.extend([""] * (len(header) - len(r)))
    n_cols = len(header)

    # ---------------- 3) 最終行をチェック -----------------
    if len(rows) <= 1:  # データ行が無い → 新行
        target_row = [""] * n_cols
        rows.append(target_row)
    else:
        target_row = rows[-1]
        target_row.extend([""] * (n_cols - len(target_row)))  # 列合わせ

    if target_row[header.index(column)] == "":
        # 空欄ならここへ格納
        target_row[header.index(column)] = str(value)
    else:
        # 既に埋まっていれば新たな行を append
        new_row = [""] * n_cols
        new_row[header.index(column)] = str(value)
        rows.append(new_row)

    # ---------------- 4) ファイル書き戻し -----------------
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows[1:] if rows and rows[0] == header else rows)

