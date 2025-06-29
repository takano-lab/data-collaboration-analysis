from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=False)
class Config:
    name: str  # 実験名
    dataset: str  # movielens or sushi
    num_institution: int  # 機関数
    num_institution_user: int  # 機関ごとのユーザー数
    num_anchor_data: int  # アンカーデータの数
    dim_intermediate: int  # 中間表現の次元数
    dim_integrate: int  # 統合表現の次元数[]
    y_name: str  # 目的変数の名前
    h_model: str  # 機械学習のモデル名
    output_path: Path  # 出力先のパス
    input_path: Path  # 入力先のパス
    seed: int = 10
