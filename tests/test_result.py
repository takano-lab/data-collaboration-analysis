import pytest
from pathlib import Path
import pandas as pd
from config.config import Config
from src.paths import INPUT_DIR
from experiments.experiment import run_once
import logging

REF_CSV = Path(__file__).resolve().parents[1] / "output" / "result_qsar_default.csv"

def load_expected():
    if not REF_CSV.exists():
        pytest.skip(f"参照CSVが無いためスキップ: {REF_CSV}")
    ref = pd.read_csv(REF_CSV)
    mask = (
        (ref["dataset"] == "qsar")
        & (ref["h_model"] == "mlp")
        & (ref["F_type"] == "kernel_pca_svd_mixed")
        & (ref["gamma_type"] == "X_tuning")
        & (ref["gamma_ratio"] == 1)
        & (ref["gamma_ratio_krr"] == 1)
        & (ref["num_anchor_data"] == 100)
        & (ref["nl_lambda"] == 0.1)
        & (ref["dim_intermediate"] == 40)
        & (ref["num_institution_user"] == 50)
        & (ref["K_normalization"].astype(str) == "True")
        & (ref["anchor_method"] == "gaussian")
    )
    ref = ref[mask].copy()
    if ref.empty:
        pytest.skip("参照CSVに対象行が無いためスキップ")
    return {row.G_type: float(row.score_mean) for _, row in ref.iterrows()}


def get_test_logger() -> logging.Logger:
    logger = logging.getLogger("dca.tests.test_result")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())  # 出力は抑制
    return logger

@pytest.mark.slow
@pytest.mark.parametrize("g_type", ['centralize', 'individual', 'Imakura', 'GEP', 'ODC', 'nonlinear'])
def test_run_once_qsar_matches_reference_3dp(tmp_path, g_type):
    expected_by_g = load_expected()
    if g_type not in expected_by_g:
        pytest.skip(f"参照CSVに {g_type} が無いのでスキップ")

    # 実際の Config を使用して experiment.run_once を実行
    cfg = Config(output_path=tmp_path, input_path=INPUT_DIR)
    # 必要パラメータを指定（問題文どおり）
    cfg.seed = 0
    cfg.dataset = "qsar"
    cfg.h_model = "mlp"
    cfg.F_type = "kernel_pca_svd_mixed"
    cfg.True_F_type = "kernel_pca_svd_mixed"
    cfg.G_type = g_type
    cfg.gamma_type = "X_tuning"
    cfg.gamma_ratio = 1
    cfg.gamma_ratio_krr = 1
    cfg.num_anchor_data = 100
    cfg.nl_lambda = 0.1
    cfg.lw_alpha = 0
    cfg.lambda_pred = 0
    cfg.lambda_offdiag = 0
    cfg.metrics = "auc"
    cfg.visualize = False
    cfg.feature_num = 41
    cfg.dim_intermediate = 40
    cfg.num_institution_user = 50
    cfg.num_institution = 10
    cfg.K_normalization = True
    cfg.anchor_method = "gaussian"
    cfg.y_name = "target"
    cfg.lambda_gen_eigen = 0
    cfg.orth_ver = False

    try:
        val = float(run_once(cfg, get_test_logger()))
    except FileNotFoundError as e:
        pytest.skip(f"データ未配置のためスキップ: {e}")
    except Exception as e:
        pytest.fail(f"run_once 実行時に例外: {e}")

    assert round(val, 3) == round(expected_by_g[g_type], 3), f"{g_type}: {val} != {expected_by_g[g_type]}"