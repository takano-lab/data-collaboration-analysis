from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str  # 実験名
    dataset: str  # movielens or sushi
    num_institution: int  # 機関数
    num_institution_user: int  # 機関ごとのユーザー数
    num_anchor_data: int  # アンカーデータの数
    dim_intermediate: int  # 中間表現の次元数
    dim_integrate: int  # 統合表現の次元数
    neighbors_centlize: int  # 集中解析での協調フィルタリングの近傍数
    neighbors_individual: int  # 個別解析での協調フィルタリングの近傍数
    output_path: Path  # 出力先のパス
    input_path: Path  # 入力先のパス
    seed: int = 10
