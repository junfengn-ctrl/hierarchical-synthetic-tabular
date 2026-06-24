from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "synthetic_independent_sampling.csv"


def independent_column_sampling(
    input_csv: Path,
    output_csv: Path,
    n_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input benchmark CSV is empty.")

    sample_size = n_rows if n_rows is not None else len(df)
    synthetic_columns: dict[str, pd.Series] = {}

    for col_idx, column in enumerate(df.columns):
        sampled = df[column].sample(
            n=sample_size,
            replace=True,
            random_state=random_state + col_idx,
        ).reset_index(drop=True)
        synthetic_columns[column] = sampled

    synthetic_df = pd.DataFrame(synthetic_columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_csv, index=False)
    return synthetic_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset by independently sampling each column."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the weak multimodal benchmark CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the output synthetic CSV.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Number of synthetic rows to generate. Defaults to the input row count.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    synthetic_df = independent_column_sampling(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        n_rows=args.n_rows,
        random_state=args.random_state,
    )

    print(f"Saved independent-sampling synthetic data to: {args.output_csv}")
    print(f"Rows: {len(synthetic_df)}")
    print(f"Columns: {len(synthetic_df.columns)}")
    print("Target by attached text sentiment:")
    print(pd.crosstab(synthetic_df["target"], synthetic_df["text_sentiment"], normalize="index"))


if __name__ == "__main__":
    main()
