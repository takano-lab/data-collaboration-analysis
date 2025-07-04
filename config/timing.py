# timing.py
from __future__ import annotations
import csv
import inspect
from pathlib import Path
from time import perf_counter
from functools import wraps
from typing import Callable, Any


# ----------------------------- CSV タイマ ------------------------------
class _CsvTimer:
    """elapsed_ms / func / Config の各フィールドを CSV に追記するだけの超軽量クラス"""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.header = ["elapsed_ms", "func"]               # ★ 順番はここで決定
        self._ensure_header()

    # -- パブリック API --------------------------------------------------
    def log(self, elapsed_ms: float, func_name: str, cfg: "Config") -> None:
        row = {"elapsed_ms": f"{elapsed_ms:.3f}", "func": func_name, **cfg.__dict__}

        # 新しいキーが来たらヘッダーを拡張
        new_cols = [k for k in row if k not in self.header]
        if new_cols:
            self.header.extend(new_cols)
            self._rewrite_header()

        # ヘッダー順に並べて追記
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(col, "") for col in self.header])

    # -- 内部ヘルパ ------------------------------------------------------
    def _ensure_header(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.header)

    def _rewrite_header(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if lines:
            lines[0] = ",".join(self.header)
            # 行末改行が無くなると壊れるので必ず付ける
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- デコレータ ------------------------------
def timed(*, config: "Config", csv_path: str | Path = None) -> Callable:
    """
    Usage:
        @timed(config=config)                          ← OK（自動で config.output_path/timing.csv）
        @timed(config=config, csv_path="custom.csv")   ← OK（手動指定も可）
    """
    # ★ デフォルト保存先を config.output_path にする
    if csv_path is None:
        assert hasattr(config, "output_path"), "config に output_path が必要です"
        csv_path = Path(config.output_path) / "timing.csv"

    timer = _CsvTimer(Path(csv_path))

    def decorator(fn: Callable) -> Callable:
        short_name = fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (perf_counter() - start) * 1_000
                timer.log(elapsed, short_name, config)

        return wrapper

    return decorator


