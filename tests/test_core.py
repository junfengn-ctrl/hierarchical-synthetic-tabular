from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_independent_sampling import independent_column_sampling
from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import prepare_features_and_target
from rule_provider import RuleProvider


class CorePipelineTests(unittest.TestCase):
    def test_default_rule_provider_has_expected_alignment(self) -> None:
        provider = RuleProvider.default()

        self.assertEqual(provider.required_target_values(), [0, 1])
        self.assertEqual(provider.allowed_text_labels(0), ["neutral", "negative"])
        self.assertEqual(provider.allowed_text_labels(1), ["positive", "neutral"])

    def test_independent_sampling_is_reproducible_and_preserves_schema(self) -> None:
        source_df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "category": ["a", "b", "a", "b", "a", "b"],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "input.csv"
            output_one = temp_path / "synthetic_one.csv"
            output_two = temp_path / "synthetic_two.csv"
            source_df.to_csv(input_csv, index=False)

            first = independent_column_sampling(input_csv, output_one, n_rows=20, random_state=7)
            second = independent_column_sampling(input_csv, output_two, n_rows=20, random_state=7)

        self.assertEqual(list(first.columns), list(source_df.columns))
        self.assertEqual(len(first), 20)
        assert_frame_equal(first, second)

    def test_target_preparation_removes_direct_leakage_columns(self) -> None:
        dataframe = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": [0, 1, 0, 1],
                "y": ["no", "yes", "no", "yes"],
                "salary": ["low", "high", "low", "high"],
                "credit_risk": ["bad", "good", "bad", "good"],
            }
        )

        features, target = prepare_features_and_target(dataframe, "target")

        self.assertEqual(list(features.columns), ["feature"])
        self.assertEqual(target.tolist(), [0, 1, 0, 1])

    def test_identical_data_has_zero_fidelity_distance(self) -> None:
        dataframe = pd.DataFrame(
            {
                "value": [float(index) for index in range(30)],
                "group": ["a", "b", "c"] * 10,
                "text_sentiment": ["negative", "positive"] * 15,
                "target": [0, 1] * 15,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            real_csv = temp_path / "real.csv"
            synth_csv = temp_path / "synthetic.csv"
            output_csv = temp_path / "fidelity.csv"
            dataframe.to_csv(real_csv, index=False)
            dataframe.to_csv(synth_csv, index=False)

            numeric_df, categorical_df, cross_modal_df = evaluate_fidelity(
                real_csv,
                synth_csv,
                output_csv,
            )

        self.assertTrue((numeric_df["abs_mean_diff"] == 0).all())
        self.assertTrue((numeric_df["abs_std_diff"] == 0).all())
        self.assertTrue((categorical_df["total_variation_distance"] == 0).all())
        self.assertTrue((cross_modal_df["abs_diff"] == 0).all())

    def test_main_cli_help_loads(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SRC_DIR / "main.py"), "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("hierarchical synthetic data benchmark pipeline", completed.stdout)


if __name__ == "__main__":
    unittest.main()
