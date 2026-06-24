from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

DEFAULT_FAST_SUMMARY_CSV = WORKSHOP_DIR / "data" / "processed" / "experiments" / "experiment_summary.csv"
DEFAULT_SDV_SUMMARY_CSV = WORKSHOP_DIR / "data" / "processed" / "experiments_sdv" / "experiment_summary.csv"
DEFAULT_OUTPUT = WORKSHOP_DIR / "data" / "processed" / "result_comparison.png"

METHOD_ORDER = ["independent", "gaussian_copula", "random_forest", "xgboost", "ctgan", "tvae"]
METHOD_LABELS = {
    "independent": "Ind.",
    "gaussian_copula": "Gauss.",
    "random_forest": "RF",
    "xgboost": "XGB",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
}
DATASET_LABELS = {
    "weak_multimodal": "Manual",
    "weak_multimodal_gemini": "Gemini",
}


def load_weak_multimodal_results(fast_summary_csv: Path, sdv_summary_csv: Path) -> pd.DataFrame:
    fast_df = pd.read_csv(fast_summary_csv)
    sdv_df = pd.read_csv(sdv_summary_csv)
    combined = pd.concat([fast_df, sdv_df], ignore_index=True)
    combined = combined[combined["dataset"].isin(DATASET_LABELS)].copy()
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()
    combined["method"] = pd.Categorical(combined["method"], categories=METHOD_ORDER, ordered=True)
    combined["benchmark"] = combined["dataset"].map(DATASET_LABELS)
    combined["method_label"] = combined["method"].map(METHOD_LABELS)
    return combined.sort_values(["benchmark", "method"])


def plot_result_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    methods = [METHOD_LABELS[method] for method in METHOD_ORDER]
    benchmarks = ["Manual", "Gemini"]
    colors = {"Manual": "#4C78A8", "Gemini": "#F58518"}
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.25), constrained_layout=True)
    metrics = [
        ("tstr_auroc_mean", "TSTR AUROC ↑", (0.0, 1.05)),
        ("tstr_f1_mean", "TSTR F1 ↑", (0.0, 1.05)),
        ("mean_cross_modal_abs_diff", "XModal ↓", (0.0, 0.36)),
    ]

    for axis, (metric, ylabel, ylim) in zip(axes, metrics):
        for offset, benchmark in zip([-width / 2, width / 2], benchmarks):
            subset = results_df[results_df["benchmark"] == benchmark].set_index("method")
            missing_methods = [method for method in METHOD_ORDER if method not in subset.index]
            if missing_methods:
                missing_labels = ", ".join(str(method) for method in missing_methods)
                raise ValueError(f"Missing methods for {benchmark} plot: {missing_labels}")
            values = [float(subset.loc[method, metric]) for method in METHOD_ORDER]
            axis.bar(x + offset, values, width=width, label=benchmark, color=colors[benchmark])

        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xticks(x)
        axis.set_xticklabels(methods, rotation=30, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot weak multimodal utility and cross-modal fidelity comparisons.")
    parser.add_argument("--fast-summary-csv", type=Path, default=DEFAULT_FAST_SUMMARY_CSV)
    parser.add_argument("--sdv-summary-csv", type=Path, default=DEFAULT_SDV_SUMMARY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results_df = load_weak_multimodal_results(args.fast_summary_csv, args.sdv_summary_csv)
    plot_result_comparison(results_df, args.output)
    print(f"Saved result comparison figure to: {args.output}")


if __name__ == "__main__":
    main()
