from __future__ import annotations

import argparse
from logging import INFO, FileHandler, getLogger

import yaml

from config.config import Config
from src.institutional_analysis import centralize_analysis, dca_analysis, individual_analysis
from src.data_collaboration import DataCollaborationAnalysis
from src.load_data import load_data
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, default="exp001")
args = parser.parse_args()

# yaml のパスと出力先パス
cfg_path    = CONFIG_DIR / f"{args.name}.yaml"
output_path = OUTPUT_DIR / args.name

# UTF-8 で読み込んで Config を生成
with cfg_path.open(encoding="utf-8") as f:
    cfg_dict = yaml.safe_load(f)

config = Config(**cfg_dict,
                output_path=output_path,
                input_path=INPUT_DIR)

# 出力ディレクトリ作成
output_path.mkdir(parents=True, exist_ok=True)

# ログの設定
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = FileHandler(filename=config.output_path / "result.log", encoding="utf-8")
logger.addHandler(handler)


def main():
    logger.info(f"データセット: {config.dataset}")
    
    # datasetの読み込み
    train_df, test_df = load_data(config=config)

    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)
    
    # データ分割 -> 統合表現の獲得まで一気に実行
    data_collaboration.run()

    # 集中解析
    centralize_analysis(config, logger, y_name=config.y_name)

    # 個別解析
    individual_analysis(
        config=config,
        logger=logger,
        Xs_train=data_collaboration.Xs_train,
        ys_train=data_collaboration.ys_train,
        Xs_test=data_collaboration.Xs_test,
        ys_test=data_collaboration.ys_test,
    )

    # 提案手法
    dca_analysis(
        X_train_integ=data_collaboration.X_train_integ,
        X_test_integ=data_collaboration.X_test_integ,
        y_train_integ=data_collaboration.y_train_integ,
        y_test_integ=data_collaboration.y_test_integ,
        config=config,
        logger=logger,
    )

def main_loop():
    LOADERS = [
        "statlog",
    #    "adult",
    #    "diabetes130",
    #    "credit_default",
        "bank_marketing",
    #    "har",
    ]
    for dataset in LOADERS:
        config.dataset = dataset
        main()

if __name__ == "__main__":
    main_loop()
    print(1)