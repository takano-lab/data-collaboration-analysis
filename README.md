# dca_yanagi
 「データ統合解析による推薦システム」の実験

# 実験方法

1. 以下コマンドを実行
```
git clone git@github.com:tomos7231/DataCollaborationAnalysis.git
cd DataCollaborationAnalysis
poetry shell
poetry install
```

2. configフォルダ内にyamlファイルを作成
```
# example(config/exp001.yaml)
name: exp001 # 実験名
dataset: movielens # movielens or sushi
seed: 42 # seed
num_institution: 2 # 機関数
num_institution_user: 10 # 機関ごとのユーザー数
num_anchor_data: 1000 # アンカーデータの数
dim_intermediate: 500 # 中間表現の次元数
dim_integrate: 300 # 統合表現の次元数
neighbors_centlize: 30 # 集中解析での協調フィルタリングの近傍数
neighbors_individual: 5 # 個別解析での協調フィルタリングの近傍数
```

3. 実験を実行
```
poetry run python main.py exp001
```

以上でoutputフォルダに使用したtrain/testテストと実験結果が保存される。