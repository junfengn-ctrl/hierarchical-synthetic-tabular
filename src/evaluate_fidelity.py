from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_SYNTH_CSV = WORKSHOP_DIR / "data" / "processed" / "synthetic_independent_sampling.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "fidelity_report.csv"


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for column in df.columns:
        is_integer_like = pd.api.types.is_integer_dtype(df[column])
        is_low_cardinality = df[column].nunique(dropna=True) <= 20
        if pd.api.types.is_numeric_dtype(df[column]) and not (is_integer_like and is_low_cardinality):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

    return categorical_cols, numeric_cols


def numeric_summary(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in numeric_cols:
        real_mean = float(real_df[col].mean())
        synth_mean = float(synth_df[col].mean())
        real_std = float(real_df[col].std())
        synth_std = float(synth_df[col].std())
        rows.append(
            {
                "column": col,
                "type": "numeric",
                "real_mean": real_mean,
                "synth_mean": synth_mean,
                "abs_mean_diff": abs(real_mean - synth_mean),
                "real_std": real_std,
                "synth_std": synth_std,
                "abs_std_diff": abs(real_std - synth_std),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(real_df: pd.DataFrame, synth_df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in categorical_cols:
        real_dist = real_df[col].astype(str).value_counts(normalize=True)
        synth_dist = synth_df[col].astype(str).value_counts(normalize=True)
        all_levels = sorted(set(real_dist.index) | set(synth_dist.index))
        total_variation = 0.5 * sum(abs(real_dist.get(level, 0.0) - synth_dist.get(level, 0.0)) for level in all_levels)
        rows.append(
            {
                "column": col,
                "type": "categorical",
                "num_levels": len(all_levels),
                "total_variation_distance": float(total_variation),
            }
        )
    return pd.DataFrame(rows)


def cross_modal_summary(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    if {"target", "text_sentiment"}.issubset(real_df.columns) and {"target", "text_sentiment"}.issubset(synth_df.columns):
        real_ct = pd.crosstab(real_df["target"], real_df["text_sentiment"], normalize="index")
        synth_ct = pd.crosstab(synth_df["target"], synth_df["text_sentiment"], normalize="index")
        sentiment_levels = sorted(set(real_ct.columns) | set(synth_ct.columns))
        target_levels = sorted(set(real_ct.index) | set(synth_ct.index))
        for target in target_levels:
            for sentiment in sentiment_levels:
                records.append(
                    {
                        "column": "target_vs_text_sentiment",
                        "type": "cross_modal",
                        "target": target,
                        "sentiment": sentiment,
                        "real_prob": float(real_ct.get(sentiment, pd.Series()).get(target, 0.0)),
                        "synth_prob": float(synth_ct.get(sentiment, pd.Series()).get(target, 0.0)),
                        "abs_diff": abs(
                            float(real_ct.get(sentiment, pd.Series()).get(target, 0.0))
                            - float(synth_ct.get(sentiment, pd.Series()).get(target, 0.0))
                        ),
                    }
                )
    return pd.DataFrame(records)


def evaluate_fidelity(real_csv: Path, synth_csv: Path, output_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_df = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    if list(real_df.columns) != list(synth_df.columns):
        raise ValueError("Real and synthetic CSVs must have identical columns in the same order.")

    categorical_cols, numeric_cols = infer_column_types(real_df)

    numeric_df = numeric_summary(real_df, synth_df, numeric_cols)
    categorical_df = categorical_summary(real_df, synth_df, categorical_cols)
    cross_modal_df = cross_modal_summary(real_df, synth_df)

    report_df = pd.concat([numeric_df, categorical_df, cross_modal_df], ignore_index=True, sort=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_csv, index=False)
    return numeric_df, categorical_df, cross_modal_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real and synthetic data fidelity with simple summary metrics.")
    parser.add_argument(
        "--real-csv",
        type=Path,
        default=DEFAULT_REAL_CSV,
        help="Path to the real benchmark CSV.",
    )
    parser.add_argument(
        "--synth-csv",
        type=Path,
        default=DEFAULT_SYNTH_CSV,
        help="Path to the synthetic CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save the fidelity report CSV.",
    )
    args = parser.parse_args()

    numeric_df, categorical_df, cross_modal_df = evaluate_fidelity(
        real_csv=args.real_csv,
        synth_csv=args.synth_csv,
        output_csv=args.output_csv,
    )

    print(f"Saved fidelity report to: {args.output_csv}")
    print("Numeric summary:")
    print(numeric_df.sort_values("abs_mean_diff", ascending=False).head(5).to_string(index=False))
    print("Categorical summary:")
    print(categorical_df.sort_values("total_variation_distance", ascending=False).head(5).to_string(index=False))
    if not cross_modal_df.empty:
        print("Cross-modal summary:")
        print(cross_modal_df.sort_values("abs_diff", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
