from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from evaluate_fidelity import evaluate_fidelity
from evaluate_utility import evaluate_utility
from run_experiments import DATASETS, ensure_clean_datasets, parse_csv_list, summarize_fidelity
from synth_xgboost import conditional_xgboost_synthesis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "xgboost_ablation"


ABLATIONS = [
    {
        "ablation": "condition_cols_4",
        "max_training_rows": 8000,
        "max_condition_cols": 4,
        "n_estimators": 80,
        "max_depth": 6,
    },
    {
        "ablation": "condition_cols_8",
        "max_training_rows": 8000,
        "max_condition_cols": 8,
        "n_estimators": 80,
        "max_depth": 6,
    },
    {
        "ablation": "condition_cols_12",
        "max_training_rows": 8000,
        "max_condition_cols": 12,
        "n_estimators": 80,
        "max_depth": 6,
    },
    {
        "ablation": "training_rows_1000",
        "max_training_rows": 1000,
        "max_condition_cols": 12,
        "n_estimators": 80,
        "max_depth": 6,
    },
    {
        "ablation": "training_rows_3000",
        "max_training_rows": 3000,
        "max_condition_cols": 12,
        "n_estimators": 80,
        "max_depth": 6,
    },
]


def run_ablation(
    datasets: list[str],
    seeds: list[int],
    output_dir: Path,
    n_rows: int | None,
    keep_run_artifacts: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for dataset in datasets:
        real_csv = DATASETS[dataset]
        if not real_csv.exists():
            raise FileNotFoundError(f"Missing processed dataset: {real_csv}")

        for config in ABLATIONS:
            for seed in seeds:
                artifact_dir = output_dir / dataset / str(config["ablation"]) / f"seed_{seed}"
                if keep_run_artifacts:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    run_context = None
                    run_dir = artifact_dir
                else:
                    run_context = tempfile.TemporaryDirectory(
                        prefix=f"{dataset}_{config['ablation']}_{seed}_"
                    )
                    run_dir = Path(run_context.name)

                try:
                    synth_csv = run_dir / "synthetic.csv"
                    fidelity_csv = run_dir / "fidelity.csv"
                    utility_csv = run_dir / "utility.csv"

                    print(f"Running dataset={dataset} ablation={config['ablation']} seed={seed}")
                    conditional_xgboost_synthesis(
                        input_csv=real_csv,
                        output_csv=synth_csv,
                        n_rows=n_rows,
                        random_state=seed,
                        max_training_rows=int(config["max_training_rows"]),
                        max_condition_cols=int(config["max_condition_cols"]),
                        n_estimators=int(config["n_estimators"]),
                        max_depth=int(config["max_depth"]),
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
                        random_state=seed,
                    )
                finally:
                    if run_context is not None:
                        run_context.cleanup()

                utility_map = utility_df.set_index("setting")
                trtr = utility_map.loc["train_real_test_real"]
                tstr = utility_map.loc["train_synthetic_test_real"]
                gap = utility_map.loc["gap_tstr_minus_trtr"]

                rows.append(
                    {
                        "dataset": dataset,
                        "ablation": config["ablation"],
                        "seed": seed,
                        "max_training_rows": config["max_training_rows"],
                        "max_condition_cols": config["max_condition_cols"],
                        "trtr_accuracy": float(trtr["accuracy"]),
                        "trtr_f1": float(trtr["f1"]),
                        "trtr_auroc": float(trtr["auroc"]),
                        "tstr_accuracy": float(tstr["accuracy"]),
                        "tstr_f1": float(tstr["f1"]),
                        "tstr_auroc": float(tstr["auroc"]),
                        "gap_accuracy": float(gap["accuracy"]),
                        "gap_f1": float(gap["f1"]),
                        "gap_auroc": float(gap["auroc"]),
                        **summarize_fidelity(numeric_df, categorical_df, cross_modal_df),
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "xgboost_ablation_results.csv", index=False)

    summary_df = (
        results_df.groupby(["dataset", "ablation"], as_index=False)
        .agg(
            trtr_accuracy_mean=("trtr_accuracy", "mean"),
            trtr_accuracy_std=("trtr_accuracy", "std"),
            trtr_f1_mean=("trtr_f1", "mean"),
            trtr_f1_std=("trtr_f1", "std"),
            trtr_auroc_mean=("trtr_auroc", "mean"),
            trtr_auroc_std=("trtr_auroc", "std"),
            tstr_accuracy_mean=("tstr_accuracy", "mean"),
            tstr_accuracy_std=("tstr_accuracy", "std"),
            tstr_f1_mean=("tstr_f1", "mean"),
            tstr_f1_std=("tstr_f1", "std"),
            tstr_auroc_mean=("tstr_auroc", "mean"),
            tstr_auroc_std=("tstr_auroc", "std"),
            gap_accuracy_mean=("gap_accuracy", "mean"),
            gap_accuracy_std=("gap_accuracy", "std"),
            gap_f1_mean=("gap_f1", "mean"),
            gap_f1_std=("gap_f1", "std"),
            gap_auroc_mean=("gap_auroc", "mean"),
            gap_auroc_std=("gap_auroc", "std"),
            mean_numeric_abs_mean_diff=("mean_numeric_abs_mean_diff", "mean"),
            mean_numeric_abs_std_diff=("mean_numeric_abs_std_diff", "mean"),
            mean_categorical_tvd=("mean_categorical_tvd", "mean"),
            mean_cross_modal_abs_diff=("mean_cross_modal_abs_diff", "mean"),
        )
        .sort_values(["dataset", "tstr_auroc_mean"], ascending=[True, False])
    )
    summary_df.to_csv(output_dir / "xgboost_ablation_summary.csv", index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused ablations for XGBoost conditional synthesis.")
    parser.add_argument(
        "--datasets",
        type=parse_csv_list,
        default=list(DATASETS.keys()),
        help=f"Comma-separated dataset names. Available: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_list,
        default=["42", "123", "2024"],
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store ablation outputs.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=12000,
        help="Number of synthetic rows per run.",
    )
    parser.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep per-run synthetic, fidelity, and utility files for debugging.",
    )
    args = parser.parse_args()

    ensure_clean_datasets()
    summary_df = run_ablation(
        datasets=args.datasets,
        seeds=[int(seed) for seed in args.seeds],
        output_dir=args.output_dir,
        n_rows=args.n_rows,
        keep_run_artifacts=args.keep_run_artifacts,
    )

    print(f"Saved detailed results to: {args.output_dir / 'xgboost_ablation_results.csv'}")
    print(f"Saved summary to: {args.output_dir / 'xgboost_ablation_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
