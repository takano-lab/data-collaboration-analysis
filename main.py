import argparse
import yaml
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

from config.config import Config
from src.load_data import load_data
from src.data_collaboration import DataCollaborationAnalysis
from src.train_val import centralize_analysis, individual_analysis, dca_analysis
from src.paths import OUTPUT_DIR, INPUT_DIR, CONFIG_DIR

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, default="exp001")
args = parser.parse_args()


# args.nameで指定した名前のyamlファイルをpathlibで読み込む（configフォルダにyamlがある）
cfg_path = CONFIG_DIR / f"{args.name}.yaml"
output_path = OUTPUT_DIR / args.name
with open(cfg_path) as f:
    config = Config(**dict(**yaml.safe_load(f), **{"output_path": output_path , "input_path": INPUT_DIR}))

# outputフォルダ作成&inputフォルダパスを取得
output_path.mkdir(parents=True, exist_ok=True)

# ログの設定
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = FileHandler(filename=config["output_path"] / "result.log")
logger.addHandler(handler)


def main():
    # datasetの読み込み
    train, test, train_rating, test_rating = load_data(config)

    # インスタンスの生成
    data_collaboration = DataCollaborationAnalysis(config, logger)

    # データ分割 -> 統合表現の獲得まで一気に実行
    (
        train_x_list,
        train_y_list,
        test_x_list,
        test_y_list,
        integrate_train_x,
        integrate_test_x,
        integrate_train_y,
        integrate_test_y,
    ) = data_collaboration.run(train, test, train_rating, test_rating)

    # 集中解析
    centralize_analysis(config, logger)

    # 個別解析
    individual_analysis(
        config, train_x_list, train_y_list, test_x_list, test_y_list, logger
    )

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
