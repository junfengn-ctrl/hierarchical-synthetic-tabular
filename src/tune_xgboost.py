from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import evaluate_utility
from synth_xgboost import conditional_xgboost_synthesis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CSV = PROJECT_ROOT / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "xgboost_tuning"


def summarize_cross_modal(cross_modal_df: pd.DataFrame) -> tuple[float, float]:
    if cross_modal_df.empty:
        return float("nan"), float("nan")

    mean_abs_diff = float(cross_modal_df["abs_diff"].mean())

    target1 = cross_modal_df[cross_modal_df["target"] == 1]
    if target1.empty:
        target1_mean = float("nan")
    else:
        target1_mean = float(target1["abs_diff"].mean())

    return mean_abs_diff, target1_mean


def run_single_config(
    name: str,
    real_csv: Path,
    output_dir: Path,
    n_rows: int,
    random_state: int,
    max_training_rows: int,
    max_condition_cols: int,
    n_estimators: int,
    max_depth: int,
) -> dict[str, float | int | str]:
    synth_csv = output_dir / f"{name}.csv"
    fidelity_csv = output_dir / f"{name}_fidelity.csv"
    utility_csv = output_dir / f"{name}_utility.csv"

    conditional_xgboost_synthesis(
        input_csv=real_csv,
        output_csv=synth_csv,
        n_rows=n_rows,
        random_state=random_state,
        max_training_rows=max_training_rows,
        max_condition_cols=max_condition_cols,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    numeric_df, categorical_df, cross_modal_df = evaluate_fidelity(
        real_csv=real_csv,
        synth_csv=synth_csv,
        output_csv=fidelity_csv,
    )
    utility_df = evaluate_utility(
        real_csv=real_csv,
        synth_csv=synth_csv,
        output_csv=utility_csv,
    )

    utility_map = utility_df.set_index("setting")
    tstr = utility_map.loc["train_synthetic_test_real"]
    gap = utility_map.loc["gap_tstr_minus_trtr"]

    mean_numeric_mean_diff = float(numeric_df["abs_mean_diff"].mean()) if not numeric_df.empty else float("nan")
    mean_categorical_tvd = float(categorical_df["total_variation_distance"].mean()) if not categorical_df.empty else float("nan")
    mean_cross_modal_diff, target1_cross_modal_diff = summarize_cross_modal(cross_modal_df)

    return {
        "config": name,
        "n_rows": n_rows,
        "max_training_rows": max_training_rows,
        "max_condition_cols": max_condition_cols,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "tstr_accuracy": float(tstr["accuracy"]),
        "tstr_f1": float(tstr["f1"]),
        "tstr_auroc": float(tstr["auroc"]),
        "gap_accuracy": float(gap["accuracy"]),
        "gap_f1": float(gap["f1"]),
        "gap_auroc": float(gap["auroc"]),
        "mean_numeric_abs_mean_diff": mean_numeric_mean_diff,
        "mean_categorical_tvd": mean_categorical_tvd,
        "mean_cross_modal_abs_diff": mean_cross_modal_diff,
        "target1_cross_modal_abs_diff": target1_cross_modal_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small XGBoost tuning sweep for synthetic data generation.")
    parser.add_argument(
        "--real-csv",
        type=Path,
        default=DEFAULT_REAL_CSV,
        help="Path to the real benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store tuning outputs.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=12000,
        help="Number of synthetic rows to generate for each config.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible tuning.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "xgb_cfg_a",
            "max_training_rows": 3000,
            "max_condition_cols": 8,
            "n_estimators": 30,
            "max_depth": 4,
        },
        {
            "name": "xgb_cfg_b",
            "max_training_rows": 5000,
            "max_condition_cols": 10,
            "n_estimators": 60,
            "max_depth": 5,
        },
        {
            "name": "xgb_cfg_c",
            "max_training_rows": 8000,
            "max_condition_cols": 12,
            "n_estimators": 80,
            "max_depth": 6,
        },
    ]

    rows: list[dict[str, float | int | str]] = []
    for config in configs:
        print(f"Running {config['name']} ...")
        summary = run_single_config(
            name=config["name"],
            real_csv=args.real_csv,
            output_dir=args.output_dir,
            n_rows=args.n_rows,
            random_state=args.random_state,
            max_training_rows=config["max_training_rows"],
            max_condition_cols=config["max_condition_cols"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
        )
        rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values(
        by=["tstr_auroc", "tstr_f1", "target1_cross_modal_abs_diff"],
        ascending=[False, False, True],
    )
    summary_csv = args.output_dir / "xgboost_tuning_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved tuning summary to: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
