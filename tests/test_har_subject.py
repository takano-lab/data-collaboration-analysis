import numpy as np
import pandas as pd

from config.config import Config
from src.institution_data_pipeline.institution_data import prepare_institutional_dataset, subject_split
from src.institution_data_pipeline.load_data import load_data


def test_subject_split_uses_one_subject_per_institution_and_common_test():
    rows = []
    for split, subjects, n_rows in [("train", [1, 3, 5], 5), ("test", [2, 4], 4)]:
        for subject in subjects:
            for i in range(n_rows):
                rows.append(
                    {
                        "x0": float(subject),
                        "x1": float(i),
                        "subject": subject,
                        "split": split,
                        "target": i % 2,
                    }
                )
    df = pd.DataFrame(rows)
    cfg = Config(dataset="har_subject", y_name="target", num_institution=2, num_institution_user=3, seed=0)

    train_df, test_df = subject_split(df, cfg)

    assert cfg.num_institution == 2
    assert cfg.num_institution_user == 3
    assert len(cfg.subject_institutions) == 2
    assert train_df["split"].eq("train").all()
    assert test_df["split"].eq("test").all()
    assert train_df.groupby("subject").size().eq(3).all()

    train_df2, test_df2, anchor_pool = subject_split(df, cfg, return_anchor_pool=True)
    assert len(train_df2) == len(train_df)
    assert len(test_df2) == len(test_df)
    assert anchor_pool["split"].eq("train").all()
    assert set(anchor_pool["subject"].unique()).issubset(set(cfg.subject_institutions))


def test_har_subject_loader_and_prepare_keep_official_test_common():
    cfg = Config(
        dataset="har_subject",
        y_name="target",
        preprocess=True,
        num_institution=3,
        num_institution_user=50,
        seed=0,
        anchor_method=None,
        har_subject_test_from_train_remaining=False,
    )
    df = load_data(cfg)

    Xs_train, Xs_test, ys_train, ys_test, train_df, test_df, _, _ = prepare_institutional_dataset(df, cfg)

    assert cfg.data_distribution == "subject"
    assert len(Xs_train) == 3
    assert len(Xs_test) == 3
    assert all(X.shape == (50, 561) for X in Xs_train)
    assert all(X.shape == (2947, 561) for X in Xs_test)
    assert all(np.array_equal(Xs_test[0], X) for X in Xs_test[1:])
    assert train_df["split"].eq("train").all()
    assert test_df["split"].eq("test").all()
    assert train_df["subject"].nunique() == 3


def test_har_subject_reserves_public_anchor_from_remaining_train_subject_data():
    cfg = Config(
        dataset="har_subject",
        y_name="target",
        preprocess=True,
        num_institution=3,
        num_institution_user=50,
        seed=0,
        anchor_method="smote",
        public_anchor_num=60,
        use_public_anchor=True,
    )
    df = load_data(cfg)

    Xs_train, Xs_test, ys_train, ys_test, train_df, test_df, public_anchor, public_anchor_y = (
        prepare_institutional_dataset(df, cfg)
    )

    assert cfg.data_distribution == "subject"
    assert cfg.public_anchor_source == "train_subject_remaining"
    assert cfg.subject_test_source == "train_subject_remaining"
    assert public_anchor.shape == (60, 561)
    assert public_anchor_y.shape == (60,)
    assert all(X.shape[1] == 561 for X in Xs_test)
    assert all(X.shape[0] == len(test_df) for X in Xs_test)
    assert all(np.array_equal(Xs_test[0], X) for X in Xs_test[1:])
    assert train_df["split"].eq("train").all()
    assert test_df["split"].eq("train").all()
    assert set(test_df["subject"].unique()).issubset(set(train_df["subject"].unique()))
    assert len(test_df) > 0
