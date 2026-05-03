from config.config import Config
from src.task_utils import is_regression_task, resolve_task


def test_explicit_task_has_priority_over_dataset_and_metric_hints():
    cfg = Config(dataset="slice_localization", task="classification", metrics="rmse")
    assert resolve_task(cfg) == "classification"
    assert cfg.task == "classification"


def test_regression_dataset_sets_task_when_unspecified():
    cfg = Config(dataset="ujiindoorloc")
    assert is_regression_task(cfg)
    assert cfg.task == "regression"
