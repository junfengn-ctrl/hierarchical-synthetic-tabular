from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare_text_dataset import TEXT_DATASET_CONFIGS, load_text_configs

DEFAULT_DATASET = "financial_phrasebank"


def count_hint_words(text: str, lexicon: set[str]) -> int:
    tokens = [tok.strip(".,;:!?()[]{}\"'").lower() for tok in str(text).split()]
    return sum(token in lexicon for token in tokens)


def build_text_features(
    input_csv: Path,
    output_csv: Path,
    text_col: str = "text",
    label_col: str = "sentiment",
    label_to_id: dict[str, int] | None = None,
    positive_hints: set[str] | None = None,
    negative_hints: set[str] | None = None,
    max_tfidf_features: int = 200,
    svd_components: int = 30,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    label_to_id = label_to_id or {}
    positive_hints = positive_hints or set()
    negative_hints = negative_hints or set()
    required_columns = {
        text_col,
        label_col,
        "text_length",
        "word_count",
        "digit_ratio",
        "uppercase_ratio",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in cleaned text CSV: {missing}")

    text_series = df[text_col].fillna("").astype(str)
    if label_to_id:
        unknown_labels = sorted(set(df[label_col].dropna().astype(str)) - set(label_to_id))
        if unknown_labels:
            raise ValueError(
                f"Labels missing from label_to_id for column '{label_col}': {unknown_labels}"
            )

    vectorizer = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(text_series)

    if tfidf_matrix.shape[1] <= 1:
        reduced_matrix = tfidf_matrix.toarray()
        reduced_columns = [f"text_svd_{i + 1}" for i in range(reduced_matrix.shape[1])]
    else:
        component_count = min(svd_components, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=component_count, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        reduced_columns = [f"text_svd_{i + 1}" for i in range(reduced_matrix.shape[1])]

    reduced_df = pd.DataFrame(reduced_matrix, columns=reduced_columns)

    feature_df = pd.DataFrame(
        {
            label_col: df[label_col],
            f"{label_col}_id": (
                df[label_col].astype(str).map(label_to_id)
                if label_to_id
                else pd.factorize(df[label_col])[0]
            ),
            "text_length": df["text_length"],
            "word_count": df["word_count"],
            "digit_ratio": df["digit_ratio"],
            "uppercase_ratio": df["uppercase_ratio"],
            "positive_hint_count": text_series.map(lambda x: count_hint_words(x, positive_hints)),
            "negative_hint_count": text_series.map(lambda x: count_hint_words(x, negative_hints)),
        }
    )

    feature_df = pd.concat([feature_df, reduced_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_csv, index=False)
    return feature_df


def text_feature_args_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_csv": Path(config["output_csv"]),
        "output_csv": Path(config["feature_output_csv"]),
        "text_col": str(config["text_col"]),
        "label_col": str(config["label_col"]),
        "label_to_id": {str(key): int(value) for key, value in dict(config.get("label_to_id", {})).items()},
        "positive_hints": {str(token).lower() for token in config.get("positive_hints", [])},
        "negative_hints": {str(token).lower() for token in config.get("negative_hints", [])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured text features for a configured text dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Text dataset key. Defaults available in config: {', '.join(sorted(TEXT_DATASET_CONFIGS))}",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional path to the cleaned text CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to the output feature CSV.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional path to a text dataset config JSON.",
    )
    parser.add_argument(
        "--max-tfidf-features",
        type=int,
        default=200,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=30,
        help="Number of SVD dimensions to keep.",
    )
    args = parser.parse_args()
    configs = load_text_configs(args.config_json) if args.config_json else TEXT_DATASET_CONFIGS
    if args.dataset not in configs:
        available = ", ".join(sorted(configs))
        raise ValueError(f"Unknown text dataset '{args.dataset}'. Available datasets: {available}")

    feature_args = text_feature_args_from_config(configs[args.dataset])
    if args.input_csv is not None:
        feature_args["input_csv"] = args.input_csv
    if args.output_csv is not None:
        feature_args["output_csv"] = args.output_csv

    feature_df = build_text_features(
        **feature_args,
        max_tfidf_features=args.max_tfidf_features,
        svd_components=args.svd_components,
    )

    print(f"Saved text features to: {feature_args['output_csv']}")
    print(f"Rows: {len(feature_df)}")
    print(f"Columns: {len(feature_df.columns)}")
    print("Feature preview:")
    print(feature_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
