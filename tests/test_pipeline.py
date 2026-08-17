from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import build_candidate, make_split, preprocessor


class PreparationTests(unittest.TestCase):
    def test_preprocessor_handles_missing_values(self):
        # This small frame tests the pipeline only. Course evidence comes from
        # the downloaded Titanic records.
        frame = pd.DataFrame(
            {
                "Pclass": [1, 3],
                "Sex": ["female", "male"],
                "Age": [38.0, None],
                "SibSp": [1, 0],
                "Parch": [0, 0],
                "Fare": [71.28, 7.25],
                "Embarked": ["C", None],
            }
        )
        transformed = preprocessor().fit_transform(frame)
        self.assertEqual(transformed.shape[0], 2)

    def test_split_keeps_all_rows(self):
        features = pd.DataFrame({"value": range(20)})
        target = pd.Series([0, 1] * 10)
        x_train, x_test, y_train, y_test = make_split(features, target)
        self.assertEqual(len(x_train) + len(x_test), 20)
        self.assertEqual(len(y_train) + len(y_test), 20)


class CandidateTests(unittest.TestCase):
    def test_candidate_is_a_pipeline(self):
        candidate = build_candidate()
        self.assertIsInstance(candidate, Pipeline)
        self.assertIn("prepare", candidate.named_steps)
        self.assertIn("model", candidate.named_steps)


if __name__ == "__main__":
    unittest.main()
