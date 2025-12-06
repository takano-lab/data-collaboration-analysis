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
    exclude = set([
        "output_path",
        "input_path",
        "name",
        "seed",
        "y_name",
        "eigenvalues",
        "nl_gammas",
        "g_abs_sum",
        "nl_lambda_opt",
        "nl_gamma_opt",
        "plot_name",
        "lambda_gen_eigen",
        "lambda_pred",
        "lambda_offdiag",
        "semi_integ",
        "orth_ver",
        "f_seed_2",
        "jreg_gep",
        "g_norm_val_gep",
        "sum_objective_function",
        "g_mean_var",
        "g_condition_number",
        "集中解析",
        "個別解析",
        "now",
        "df_name",
        "intermediate_name",
        "integrated_name",
        "V_sel",
        "lambdas",
    ])
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
cfg.output_path/filename に (column,value) を追記する。
大きなファイルでもメモリを使いすぎないよう、1 行ずつ読み書きする。
    """
    from pathlib import Path
    import csv

    assert hasattr(cfg, "output_path") and cfg.output_path, "Config に output_path が必要です"
    csv_path = Path(cfg.output_path) / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as in_f, temp_path.open("w", newline="", encoding="utf-8") as out_f:
            reader = csv.reader(in_f)
            writer = csv.writer(out_f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            if column not in header:
                header.append(column)
            col_idx = header.index(column)
            writer.writerow(header)

            prev_row = None
            for row in reader:
                row.extend([""] * (len(header) - len(row)))
                if prev_row is not None:
                    writer.writerow(prev_row)
                prev_row = row

            if prev_row is None:
                prev_row = [""] * len(header)

            if prev_row[col_idx] == "":
                prev_row[col_idx] = str(value)
                writer.writerow(prev_row)
            else:
                writer.writerow(prev_row)
                new_row = [""] * len(header)
                new_row[col_idx] = str(value)
                writer.writerow(new_row)
    else:
        header = [column]
        with temp_path.open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerow([str(value)])

    temp_path.replace(csv_path)

