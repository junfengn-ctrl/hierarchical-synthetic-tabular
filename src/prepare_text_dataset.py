from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from config_utils import CONFIG_DIR, load_json_config, resolve_path_fields


DEFAULT_CONFIG_JSON = CONFIG_DIR / "text_datasets.json"


def load_text_configs(config_json: Path = DEFAULT_CONFIG_JSON) -> dict[str, dict[str, Any]]:
    configs = load_json_config(config_json)
    return {
        name: resolve_path_fields(config, ["input_csv", "output_csv", "feature_output_csv"])
        for name, config in configs.items()
    }


TEXT_DATASET_CONFIGS = load_text_configs()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def read_text_csv(input_csv: Path, config: dict[str, object]) -> pd.DataFrame:
    encodings = list(config["encodings"])
    read_csv_kwargs = dict(config["read_csv_kwargs"])
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(input_csv, encoding=str(encoding), **read_csv_kwargs)
            return standardize_columns(df)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not read text CSV: {input_csv}")


def clean_text_dataset(
    dataset: str,
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    text_configs = configs or TEXT_DATASET_CONFIGS
    if dataset not in text_configs:
        available = ", ".join(sorted(text_configs))
        raise ValueError(f"Unknown text dataset '{dataset}'. Available datasets: {available}")

    config = text_configs[dataset]
    source_csv = input_csv if input_csv is not None else Path(config["input_csv"])
    target_csv = output_csv if output_csv is not None else Path(config["output_csv"])

    df = read_text_csv(source_csv, config)
    text_col = str(config["text_col"])
    label_col = str(config["label_col"])
    valid_labels = set(config["valid_labels"])
    dedupe_cols = list(config["dedupe_cols"])

    missing = sorted({text_col, label_col} - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in text CSV: {missing}")

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[text_col] = df[text_col].map(normalize_text)

    df = df[df[label_col].isin(valid_labels)].copy()
    df = df[df[text_col].str.len() > 0].copy()
    df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    df["text_length"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()
    df["digit_ratio"] = df[text_col].str.count(r"\d") / df["text_length"].clip(lower=1)
    df["uppercase_ratio"] = df[text_col].str.count(r"[A-Z]") / df["text_length"].clip(lower=1)

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a configured text dataset.")
    parser.add_argument(
        "dataset",
        help=f"Text dataset key to clean. Defaults available in config: {', '.join(sorted(TEXT_DATASET_CONFIGS))}",
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
        help="Path to the text dataset config JSON.",
    )
    args = parser.parse_args()
    configs = load_text_configs(args.config_json)

    df = clean_text_dataset(
        dataset=args.dataset,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        configs=configs,
    )

    output_csv = args.output_csv or configs[args.dataset]["output_csv"]
    label_col = str(configs[args.dataset]["label_col"])
    print(f"Saved cleaned {args.dataset} data to: {output_csv}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Label distribution:")
    print(df[label_col].value_counts(dropna=False))


if __name__ == "__main__":
    main()
