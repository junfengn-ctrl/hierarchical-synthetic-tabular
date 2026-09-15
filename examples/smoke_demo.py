from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_independent_sampling import independent_column_sampling
from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import evaluate_utility


def build_demo_dataset() -> pd.DataFrame:
    """Create a small deterministic dataset that is independent of benchmark downloads."""
    rows: list[dict[str, object]] = []
    jobs = ["admin", "services", "technician", "management"]
    for index in range(80):
        target = index % 2
        rows.append(
            {
                "age": 21 + (index % 45),
                "balance": float(250 + index * 37 + target * 400),
                "job": jobs[index % len(jobs)],
                "text_sentiment": "positive" if target else "negative",
                "target": target,
                # Included deliberately to exercise the evaluator's leakage guard.
                "y": "yes" if target else "no",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hierarchical-synthetic-smoke-") as temp_dir:
        work_dir = Path(temp_dir)
        real_csv = work_dir / "real.csv"
        synthetic_csv = work_dir / "synthetic.csv"
        fidelity_csv = work_dir / "fidelity.csv"
        utility_csv = work_dir / "utility.csv"

        build_demo_dataset().to_csv(real_csv, index=False)
        synthetic_df = independent_column_sampling(
            input_csv=real_csv,
            output_csv=synthetic_csv,
            n_rows=120,
            random_state=42,
        )
        numeric_df, categorical_df, cross_modal_df = evaluate_fidelity(
            real_csv=real_csv,
            synth_csv=synthetic_csv,
            output_csv=fidelity_csv,
        )
        utility_df = evaluate_utility(
            real_csv=real_csv,
            synth_csv=synthetic_csv,
            output_csv=utility_csv,
            random_state=42,
        )

        if list(synthetic_df.columns) != list(build_demo_dataset().columns):
            raise RuntimeError("Smoke demo failed: synthetic schema differs from the input schema.")
        if len(synthetic_df) != 120:
            raise RuntimeError("Smoke demo failed: unexpected synthetic row count.")

        tstr_row = utility_df.loc[utility_df["setting"] == "train_synthetic_test_real"].iloc[0]
        print("Smoke demo passed.")
        print(f"Synthetic rows: {len(synthetic_df)}")
        print(f"TSTR AUROC: {tstr_row['auroc']:.4f}")
        print(f"Numeric fidelity rows: {len(numeric_df)}")
        print(f"Categorical fidelity rows: {len(categorical_df)}")
        print(f"Cross-modal fidelity rows: {len(cross_modal_df)}")


if __name__ == "__main__":
    main()
