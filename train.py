"""Day 1: baseline and one candidate on real Titanic passenger records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CATEGORICAL = ["Sex", "Embarked"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
REQUIRED = set(FEATURES + [TARGET, "PassengerId", "Name"])
EXPECTED_COLUMNS = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp",
    "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}
SPLIT_SEED = 42


def load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Real Titanic data not found at {path}. "
            "Follow the starter README and download train.csv from Kaggle."
        )
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def verify_real_data(path: Path) -> dict[str, object]:
    """Verify the exact real Titanic classroom file before model work."""
    frame = load_frame(path)
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if len(frame) != 891:
        raise ValueError(f"Expected 891 passenger rows but found {len(frame)}")
    counts = frame[TARGET].value_counts().sort_index().to_dict()
    if counts != {0: 549, 1: 342}:
        raise ValueError(f"Unexpected Survived counts: {counts}")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "survived_counts": counts,
        "missing_age": int(frame["Age"].isna().sum()),
        "missing_cabin": int(frame["Cabin"].isna().sum()),
        "missing_embarked": int(frame["Embarked"].isna().sum()),
    }


def make_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[FEATURES].copy(), frame[TARGET].astype(int)


def make_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=SPLIT_SEED,
        stratify=target,
    )


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("category", categorical, CATEGORICAL),
            ("number", numeric, NUMERIC),
        ]
    )


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            ("prepare", preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_candidate() -> Pipeline:
    """Return one stronger reproducible candidate model."""
    return Pipeline([
        ("prepare", preprocessor()),
        ("model", RandomForestClassifier(random_state=SPLIT_SEED))
    ])


def evaluate(y_true: pd.Series, y_pred: object) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_survived": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_survived": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_survived": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    if args.check_data:
        result = verify_real_data(args.data)
        print("REAL DATA CHECK PASSED")
        for name, value in result.items():
            print(f"{name}: {value}")
        return 0

    frame = load_frame(args.data)
    features, target = make_xy(frame)
    x_train, x_test, y_train, y_test = make_split(features, target)

    baseline = build_baseline()
    baseline.fit(x_train, y_train)
    result: dict[str, object] = {
        "dataset": {
            "name": "Titanic Dataset train.csv",
            "source": "Kaggle hesh97/titanicdataset-traincsv",
            "rows": len(frame),
            "features": FEATURES,
            "target": TARGET,
            "test_rows": len(x_test),
            "missing_values": {
                name: int(frame[name].isna().sum())
                for name in FEATURES
                if frame[name].isna().any()
            },
        },
        "baseline": evaluate(y_test, baseline.predict(x_test)),
        "candidate": None,
    }

    try:
        candidate = build_candidate()
        candidate.fit(x_train, y_train)
        prediction = candidate.predict(x_test)
        result["candidate"] = evaluate(y_test, prediction)
        errors = frame.loc[
            x_test.index, ["PassengerId", "Name", *FEATURES]
        ].copy()
        errors.insert(0, "source_row", errors.index)
        errors["true_survived"] = y_test
        errors["predicted_survived"] = prediction
        errors[errors["true_survived"] != errors["predicted_survived"]].to_csv(
            "errors.csv", index=False
        )
    except NotImplementedError as error:
        result["candidate_todo"] = str(error)

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
