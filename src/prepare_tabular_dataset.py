from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from config_utils import CONFIG_DIR, load_json_config, resolve_path_fields

DEFAULT_CONFIG_JSON = CONFIG_DIR / "tabular_datasets.json"


def load_tabular_configs(config_json: Path = DEFAULT_CONFIG_JSON) -> dict[str, dict[str, Any]]:
    configs = load_json_config(config_json)
    return {
        name: resolve_path_fields(config, ["input_csv", "output_csv"])
        for name, config in configs.items()
    }


DATASET_CONFIGS = load_tabular_configs()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def clean_string_columns(df: pd.DataFrame, missing_tokens: list[str]) -> pd.DataFrame:
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    replacement_map = {token: "unknown" for token in missing_tokens}
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        if replacement_map:
            df[col] = df[col].replace(replacement_map)
    return df


def build_target(df: pd.DataFrame, target_col: str, target_map: dict[str, int] | None) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' was not found.")

    if target_map is not None:
        target = df[target_col].map(target_map)
        if target.isna().any():
            unexpected = sorted(df.loc[target.isna(), target_col].dropna().unique())
            raise ValueError(f"Unexpected values found in column '{target_col}': {unexpected}")
        return target.astype(int)

    unique_values = sorted(df[target_col].dropna().unique())
    if unique_values == [0, 1]:
        return df[target_col].astype(int)
    if unique_values == [1, 2]:
        return (df[target_col] == 2).astype(int)
    raise ValueError(f"Expected binary target values in '{target_col}', found: {unique_values}")


def clean_tabular_dataset(
    dataset: str,
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    dataset_configs = configs or DATASET_CONFIGS
    if dataset not in dataset_configs:
        available = ", ".join(sorted(dataset_configs))
        raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {available}")

    config = dataset_configs[dataset]
    source_csv = input_csv if input_csv is not None else Path(config["input_csv"])
    target_csv = output_csv if output_csv is not None else Path(config["output_csv"])

    df = pd.read_csv(source_csv, **dict(config["read_csv_kwargs"]))
    df = standardize_columns(df)
    df = clean_string_columns(df, list(config["missing_tokens"]))

    target_col = str(config["target_col"])
    target_map = config["target_map"]
    df["target"] = build_target(df, target_col, target_map)

    if bool(config["drop_source_target"]) and target_col in df.columns:
        df = df.drop(columns=[target_col])

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a configured tabular dataset and create a binary target column.")
    parser.add_argument(
        "dataset",
        help=f"Dataset key to clean. Defaults available in config: {', '.join(sorted(DATASET_CONFIGS))}",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional override for the raw input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional override for the cleaned output CSV.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=DEFAULT_CONFIG_JSON,
        help="Path to the tabular dataset config JSON.",
    )
    args = parser.parse_args()
    configs = load_tabular_configs(args.config_json)

    df = clean_tabular_dataset(
        dataset=args.dataset,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        configs=configs,
    )

    output_csv = args.output_csv or configs[args.dataset]["output_csv"]
    print(f"Saved cleaned {args.dataset} data to: {output_csv}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Target distribution:")
    print(df["target"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
