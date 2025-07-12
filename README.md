# dca_yanagi
 「データ統合解析による推薦システム」の実験

# 実験方法

1. 以下コマンドを実行
```
git clone git@github.com:takano-lab/dca_yanagi.git
cd DataCollaborationAnalysis
poetry shell
poetry install

gurobipy を使用する場合
pip install gurobipy
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
```

3. 実験を実行
```
poetry run python main.py exp001
```

以上でoutputフォルダに使用したtrain/testテストと実験結果が保存される。

ノートブックにカーネルを反映させるには以下のコードを使用

```
poetry run python -m ipykernel install --user --name=dca-project --display-name "Python (dca project)"

```

HARデータを使用する場合、input に格納する
例
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip
```


# コードフォーマット
isort, black, autoflakeによる整形
```
make format
```
