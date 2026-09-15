from __future__ import annotations

import argparse
from pathlib import Path

from ablate_xgboost import DEFAULT_OUTPUT_DIR as DEFAULT_ABLATION_DIR
from ablate_xgboost import run_ablation
from build_text_features import build_text_features, text_feature_args_from_config
from build_weak_alignment import DEFAULT_OUTPUT_CSV as DEFAULT_WEAK_ALIGNMENT_CSV
from build_weak_alignment import build_weak_alignment
from prepare_tabular_dataset import DATASET_CONFIGS, clean_tabular_dataset, load_tabular_configs
from prepare_text_dataset import TEXT_DATASET_CONFIGS, clean_text_dataset, load_text_configs
from rule_provider import load_rule_provider
from run_experiments import DATASETS, DEFAULT_OUTPUT_DIR as DEFAULT_EXPERIMENT_DIR
from run_experiments import parse_csv_list, run_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDV_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experiments_sdv"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "result_comparison.png"
GEMINI_RULE_CONFIG_JSON = PROJECT_ROOT / "configs" / "rule_provider_gemini.json"
GEMINI_WEAK_ALIGNMENT_CSV = PROJECT_ROOT / "data" / "processed" / "weak_multimodal_gemini_benchmark.csv"

DEFAULT_TABULAR_DATASETS = list(DATASET_CONFIGS)
DEFAULT_TEXT_DATASETS = list(TEXT_DATASET_CONFIGS)
DEFAULT_EXPERIMENT_DATASETS = list(DATASETS)
DEFAULT_FAST_METHODS = ["independent", "gaussian_copula", "random_forest", "xgboost"]
DEFAULT_SDV_METHODS = ["ctgan", "tvae"]
DEFAULT_SEEDS = [42, 123, 2024]
DEFAULT_SDV_SEEDS = [42]
DEFAULT_ABLATION_DATASETS = ["weak_multimodal", "weak_multimodal_gemini"]
WEAK_ALIGNMENT_BENCHMARKS = [
    ("weak_multimodal", None, DEFAULT_WEAK_ALIGNMENT_CSV),
    ("weak_multimodal_gemini", GEMINI_RULE_CONFIG_JSON, GEMINI_WEAK_ALIGNMENT_CSV),
]


def run_prepare() -> None:
    tabular_configs = load_tabular_configs()
    text_configs = load_text_configs()

    print("Preparing tabular datasets...")
    for dataset in DEFAULT_TABULAR_DATASETS:
        df = clean_tabular_dataset(dataset, configs=tabular_configs)
        output_csv = tabular_configs[dataset]["output_csv"]
        print(f"  {dataset}: {len(df)} rows -> {output_csv}")

    print("Preparing text datasets...")
    for dataset in DEFAULT_TEXT_DATASETS:
        df = clean_text_dataset(dataset, configs=text_configs)
        output_csv = text_configs[dataset]["output_csv"]
        print(f"  {dataset}: {len(df)} rows -> {output_csv}")

    print("Building text features...")
    text_feature_outputs: dict[str, Path] = {}
    for dataset in DEFAULT_TEXT_DATASETS:
        feature_args = text_feature_args_from_config(text_configs[dataset])
        feature_df = build_text_features(**feature_args)
        output_csv = Path(feature_args["output_csv"])
        text_feature_outputs[dataset] = output_csv
        print(f"  {dataset} features: {len(feature_df)} rows -> {output_csv}")

    print("Building rule-guided weak multimodal benchmarks...")
    for benchmark_name, rule_config, output_csv in WEAK_ALIGNMENT_BENCHMARKS:
        provider = load_rule_provider(rule_config)
        text_source = str(provider.config["schema"]["text_source"])
        benchmark_df = build_weak_alignment(
            bank_csv=PROJECT_ROOT / "data" / "processed" / "bank_marketing_clean.csv",
            text_feature_csv=text_feature_outputs[text_source],
            output_csv=output_csv,
            rule_provider=provider,
        )
        print(f"  {benchmark_name}: {len(benchmark_df)} rows -> {output_csv}")


def run_standard_experiments(
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    n_rows: int,
    output_dir: Path,
    keep_run_artifacts: bool,
) -> None:
    summary_df = run_experiments(
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        output_dir=output_dir,
        n_rows=n_rows,
        keep_run_artifacts=keep_run_artifacts,
    )
    print(summary_df.to_string(index=False))


def run_xgboost_ablation(
    datasets: list[str],
    seeds: list[int],
    n_rows: int,
    output_dir: Path,
    keep_run_artifacts: bool,
) -> None:
    summary_df = run_ablation(
        datasets=datasets,
        seeds=seeds,
        output_dir=output_dir,
        n_rows=n_rows,
        keep_run_artifacts=keep_run_artifacts,
    )
    print(summary_df.to_string(index=False))


def run_result_plot(experiments_output_dir: Path, sdv_output_dir: Path, plot_output: Path) -> None:
    fast_summary_csv = experiments_output_dir / "experiment_summary.csv"
    sdv_summary_csv = sdv_output_dir / "experiment_summary.csv"
    missing = [path for path in [fast_summary_csv, sdv_summary_csv] if not path.exists()]
    if missing:
        print("Skipping result comparison plot; missing summary file(s):")
        for path in missing:
            print(f"  {path}")
        return

    from plot_result_comparison import load_weak_multimodal_results, plot_result_comparison

    try:
        results_df = load_weak_multimodal_results(
            fast_summary_csv=fast_summary_csv,
            sdv_summary_csv=sdv_summary_csv,
        )
        plot_result_comparison(results_df, plot_output)
    except ValueError as exc:
        print(f"Skipping result comparison plot: {exc}")
        return
    print(f"Saved result comparison plot to: {plot_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hierarchical synthetic data benchmark pipeline.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["prepare", "experiments", "ablation", "sdv", "all"],
        help="Pipeline stage to run. Defaults to all.",
    )
    parser.add_argument(
        "--datasets",
        type=parse_csv_list,
        default=DEFAULT_EXPERIMENT_DATASETS,
        help=f"Experiment datasets. Available: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--methods",
        type=parse_csv_list,
        default=DEFAULT_FAST_METHODS,
        help="Fast experiment methods.",
    )
    parser.add_argument(
        "--sdv-methods",
        type=parse_csv_list,
        default=DEFAULT_SDV_METHODS,
        help="SDV methods for the sdv command.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_list,
        default=[str(seed) for seed in DEFAULT_SEEDS],
        help="Comma-separated integer seeds for fast experiments and ablations.",
    )
    parser.add_argument(
        "--sdv-seeds",
        type=parse_csv_list,
        default=[str(seed) for seed in DEFAULT_SDV_SEEDS],
        help="Comma-separated integer seeds for SDV experiments.",
    )
    parser.add_argument(
        "--ablation-datasets",
        type=parse_csv_list,
        default=DEFAULT_ABLATION_DATASETS,
        help="Datasets for XGBoost ablation.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=12000,
        help="Synthetic rows per run.",
    )
    parser.add_argument(
        "--experiments-output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Output directory for main experiments.",
    )
    parser.add_argument(
        "--ablation-output-dir",
        type=Path,
        default=DEFAULT_ABLATION_DIR,
        help="Output directory for XGBoost ablation.",
    )
    parser.add_argument(
        "--sdv-output-dir",
        type=Path,
        default=DEFAULT_SDV_OUTPUT_DIR,
        help="Output directory for SDV experiments.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help="Output PNG path for the result comparison plot.",
    )
    parser.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep per-run synthetic, fidelity, utility, and metadata files for debugging.",
    )
    parser.add_argument(
        "--include-sdv",
        action="store_true",
        help="Also run SDV experiments after the default all pipeline.",
    )
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds]
    sdv_seeds = [int(seed) for seed in args.sdv_seeds]

    if args.command in {"prepare", "all"}:
        run_prepare()

    if args.command in {"experiments", "all"}:
        run_standard_experiments(
            datasets=args.datasets,
            methods=args.methods,
            seeds=seeds,
            n_rows=args.n_rows,
            output_dir=args.experiments_output_dir,
            keep_run_artifacts=args.keep_run_artifacts,
        )

    if args.command in {"ablation", "all"}:
        run_xgboost_ablation(
            datasets=args.ablation_datasets,
            seeds=seeds,
            n_rows=args.n_rows,
            output_dir=args.ablation_output_dir,
            keep_run_artifacts=args.keep_run_artifacts,
        )

    if args.command == "sdv" or (args.command == "all" and args.include_sdv):
        run_standard_experiments(
            datasets=args.datasets,
            methods=args.sdv_methods,
            seeds=sdv_seeds,
            n_rows=args.n_rows,
            output_dir=args.sdv_output_dir,
            keep_run_artifacts=args.keep_run_artifacts,
        )

    if args.command in {"experiments", "sdv", "all"}:
        run_result_plot(
            experiments_output_dir=args.experiments_output_dir,
            sdv_output_dir=args.sdv_output_dir,
            plot_output=args.plot_output,
        )


if __name__ == "__main__":
    main()
