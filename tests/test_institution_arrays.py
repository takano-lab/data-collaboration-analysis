import logging

import pandas as pd
import pytest

from config.config import Config
from src.institution_data_pipeline.institution_data import to_institution_arrays


def test_division_regression_skips_label_shortage_warning(caplog):
    cfg = Config(
        data_distribution="division",
        dataset="slice_localization",
        task="regression",
        num_institution=2,
        num_institution_user=3,
        y_name="target",
    )
    train_df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "target": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3],
        }
    )
    test_df = pd.DataFrame(
        {
            "x": [10, 11, 12, 13],
            "target": [3.1, 3.2, 3.3, 3.4],
        }
    )

    with caplog.at_level(logging.WARNING):
        Xs_train, Xs_test, ys_train, ys_test = to_institution_arrays(train_df, test_df, cfg)

    assert "division test: 一部ラベル不足" not in caplog.text
    assert len(Xs_train) == 2
    assert len(Xs_test) == 2
    assert [len(y) for y in ys_train] == [3, 3]
    assert [len(y) for y in ys_test] == [4, 4]


def test_division_classification_still_requires_single_label_train_blocks():
    cfg = Config(
        data_distribution="division",
        task="classification",
        num_institution=2,
        num_institution_user=3,
        y_name="target",
    )
    train_df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "x": [10, 11, 12, 13, 14, 15],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="division train block"):
        to_institution_arrays(train_df, test_df, cfg)
