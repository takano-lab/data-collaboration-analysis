from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.institution_data_pipeline.load_data import LOADERS


def load_har_Xy():
    df = LOADERS["har"]()
    # Exclude non-feature columns
    drop_cols = [c for c in ["target", "subject"] if c in df.columns]
    X = df.drop(columns=drop_cols).to_numpy()
    y = df["target"].to_numpy()
    return X, y


def evaluate_models():
    X, y = load_har_Xy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, multi_class="multinomial", solver="saga", n_jobs=None)),
        "LinearSVC": make_pipeline(StandardScaler(), LinearSVC()),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "kNN-5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "model": name,
            "acc_mean": scores["test_accuracy"].mean(),
            "acc_std": scores["test_accuracy"].std(),
            "f1_mean": scores["test_f1_macro"].mean(),
            "f1_std": scores["test_f1_macro"].std(),
        })
    return pd.DataFrame(rows).sort_values("acc_mean", ascending=False)


if __name__ == "__main__":
    df = evaluate_models()
    with pd.option_context('display.max_colwidth', None, 'display.precision', 4):
        print(df)

