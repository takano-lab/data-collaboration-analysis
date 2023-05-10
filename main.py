import argparse
from logging import INFO, FileHandler, getLogger

import yaml

from config.config import Config
from src.data_collaboration import DataCollaborationAnalysis
from src.load_data import load_data
from src.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR
from src.train_val import centralize_analysis, dca_analysis, individual_analysis

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, default="exp001")
args = parser.parse_args()

# args.nameで指定した名前のyamlファイルをpathlibで読み込む（configフォルダにyamlがある）
cfg_path = CONFIG_DIR / f"{args.name}.yaml"
output_path = OUTPUT_DIR / args.name
with open(cfg_path) as f:
    config = Config(**dict(**yaml.safe_load(f), **{"output_path": output_path, "input_path": INPUT_DIR}))

# ログの設定
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = FileHandler(filename=config["output_path"] / "result.log")
logger.addHandler(handler)


def main():
    # datasetの読み込み
    train_df, test_df = load_data(dataset=config.dataset, seed=config.seed, input_path=config.input_path)

    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config=config, logger=logger, train_df=train_df, test_df=test_df)

    # データ分割 -> 統合表現の獲得まで一気に実行
    data_collaboration.run()

    # 集中解析
    centralize_analysis(config, logger)

    # 個別解析
    individual_analysis(config, train_x_list, train_y_list, test_x_list, test_y_list, logger)

    # 提案手法
    dca_analysis(
        config,
        integrate_train_x,
        integrate_test_x,
        integrate_train_y,
        integrate_test_y,
        logger,
    )


if __name__ == "__main__":
    main()
