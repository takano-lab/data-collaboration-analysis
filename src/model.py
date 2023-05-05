import optuna.integration.lightgbm as tlgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, accuracy
from surprise import (
    NormalPredictor,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SVD,
    SVDpp,
    NMF,
    SlopeOne,
    CoClustering,
)

# from pyfm import pylibfm
# from scipy.sparse import csr_matrix


def run_lgbm(
    config, X_train, y_train, X_test, y_test, categorical_cols=[], use_optuna=False
):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": 0.1,
        "verbosity": -1,
        "random_state": config["seed"],
    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train[train_index, :]
        X_val = X_train[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        if use_optuna:
            lgb_train = tlgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

            lgb_eval = tlgb.Dataset(
                X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
            )

            model = tlgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(period=100),
                ],
            )

        else:
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

            lgb_eval = lgb.Dataset(
                X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
            )

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(period=100),
                ],
            )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )

        models.append(model)

    # testデータでの性能を評価
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


def run_surprise(config, train, test, neightbors=10):
    rating_columns = ["uid", "mid", "rating"]
    if config["dataset"] == "movielens":
        reader = Reader(rating_scale=(1, 5))
    elif config["dataset"] == "sushi":
        reader = Reader(rating_scale=(0, 4))

    # surpriseでデータを読み込む
    train_surprise_data = Dataset.load_from_df(train[rating_columns], reader)
    test_surprise_data = Dataset.load_from_df(test[rating_columns], reader)

    # データセットをビルド
    trainset = train_surprise_data.build_full_trainset()
    testset = test_surprise_data.build_full_trainset()
    testset = testset.build_testset()

    # アルゴリズムを設定
    algo = KNNBasic(k=neightbors)

    # 学習
    algo.fit(trainset)

    # testデータでの性能を評価
    pred = algo.test(testset)
    rmse = accuracy.rmse(pred)

    return rmse


# def run_fm(config, X_train, y_train, X_test, y_test):
#     # csr_matrixに変換
#     X_train = csr_matrix(X_train, dtype=np.float64)
#     X_test = csr_matrix(X_test, dtype=np.float64)

#     fm = pylibfm.FM(
#         num_factors=20,
#         num_iter=100,
#         task="regression",
#         initial_learning_rate=0.001,
#         learning_rate_schedule="optimal",
#         verbose=False,
#     )
#     fm.fit(X_train, y_train)

#     # testデータでの性能を評価
#     y_pred = fm.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#     return rmse
