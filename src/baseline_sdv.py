from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "synthetic_sdv.csv"
DEFAULT_METADATA_JSON = PROJECT_ROOT / "data" / "processed" / "synthetic_sdv_metadata.json"

SDVMethod = Literal["ctgan", "tvae"]


def low_cardinality_integer_columns(df: pd.DataFrame, max_levels: int = 20) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]) and df[column].nunique(dropna=True) <= max_levels:
            columns.append(column)
    return columns


def import_sdv_classes(method: SDVMethod):
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    except ImportError as exc:
        raise ImportError(
            "SDV is required for CTGAN/TVAE baselines. Install it in your environment with: "
            "pip install sdv"
        ) from exc

    synthesizer_classes = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }
    return Metadata, synthesizer_classes[method]


def update_sdv_metadata(metadata, real_df: pd.DataFrame) -> None:
    # SDV may infer binary or ordinal integer columns as numerical. For utility
    # evaluation, columns such as target must stay in their original label set.
    for column in low_cardinality_integer_columns(real_df):
        metadata.update_column(column_name=column, sdtype="categorical")


def coerce_generated_domains(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    synth_df = synth_df.copy()
    for column in real_df.columns:
        if column not in synth_df.columns:
            continue

        if pd.api.types.is_integer_dtype(real_df[column]):
            valid_values = np.array(sorted(real_df[column].dropna().unique()))
            numeric_values = pd.to_numeric(synth_df[column], errors="coerce")

            if real_df[column].nunique(dropna=True) <= 20:
                fallback = int(pd.Series(valid_values).mode().iloc[0])

                def nearest_valid(value: float) -> int:
                    if pd.isna(value):
                        return fallback
                    nearest_idx = int(np.abs(valid_values - value).argmin())
                    return int(valid_values[nearest_idx])

                synth_df[column] = numeric_values.map(nearest_valid).astype(real_df[column].dtype)
            else:
                synth_df[column] = numeric_values.round().fillna(real_df[column].median()).astype(real_df[column].dtype)
        elif pd.api.types.is_float_dtype(real_df[column]):
            synth_df[column] = pd.to_numeric(synth_df[column], errors="coerce")
        else:
            synth_df[column] = synth_df[column].astype(str)

    return synth_df


def sdv_synthesis(
    input_csv: Path,
    output_csv: Path,
    method: SDVMethod,
    n_rows: int | None = None,
    random_state: int = 42,
    epochs: int = 50,
    metadata_json: Path | None = None,
) -> pd.DataFrame:
    real_df = pd.read_csv(input_csv)
    if real_df.empty:
        raise ValueError("Input CSV is empty.")

    sample_size = n_rows if n_rows is not None else len(real_df)
    Metadata, Synthesizer = import_sdv_classes(method)

    metadata = Metadata.detect_from_dataframe(data=real_df)
    update_sdv_metadata(metadata, real_df)
    if metadata_json is not None:
        metadata_json.parent.mkdir(parents=True, exist_ok=True)
        if metadata_json.exists():
            metadata_json.unlink()
        metadata.save_to_json(metadata_json)

    synthesizer = Synthesizer(
        metadata,
        epochs=epochs,
        verbose=False,
    )
    if hasattr(synthesizer, "set_random_state"):
        synthesizer.set_random_state(random_state)

    synthesizer.fit(real_df)
    synth_df = synthesizer.sample(num_rows=sample_size)
    synth_df = synth_df.reindex(columns=real_df.columns)
    synth_df = coerce_generated_domains(real_df, synth_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output_csv, index=False)
    return synth_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data with SDV CTGAN or TVAE.")
    parser.add_argument(
        "method",
        choices=["ctgan", "tvae"],
        help="SDV synthesizer to run.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the real benchmark CSV.",
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
        help="Random seed for reproducible synthesis when supported by SDV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the SDV neural synthesizer.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=DEFAULT_METADATA_JSON,
        help="Path to save detected SDV metadata for reproducibility.",
    )
    args = parser.parse_args()

    try:
        synthetic_df = sdv_synthesis(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            method=args.method,
            n_rows=args.n_rows,
            random_state=args.random_state,
            epochs=args.epochs,
            metadata_json=args.metadata_json,
        )
    except ImportError as exc:
        parser.error(str(exc))

    print(f"Saved {args.method.upper()} synthetic data to: {args.output_csv}")
    print(f"Rows: {len(synthetic_df)}")
    print(f"Columns: {len(synthetic_df.columns)}")


if __name__ == "__main__":
    main()
