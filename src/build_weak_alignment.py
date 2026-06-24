from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rule_provider import RuleProvider, load_rule_provider


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BANK_CSV = WORKSHOP_DIR / "data" / "processed" / "bank_marketing_clean.csv"
DEFAULT_TEXT_FEATURE_CSV = WORKSHOP_DIR / "data" / "processed" / "financial_phrasebank_features.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"


def sample_text_block(
    text_df: pd.DataFrame,
    sample_size: int,
    allowed_sentiments: list[str],
    random_state: int,
) -> pd.DataFrame:
    subset = text_df[text_df["sentiment"].isin(allowed_sentiments)].copy()
    if subset.empty:
        raise ValueError(f"No text rows found for sentiments: {allowed_sentiments}")

    sampled = subset.sample(
        n=sample_size,
        replace=True,
        random_state=random_state,
    ).reset_index(drop=True)
    return sampled


def build_weak_alignment(
    bank_csv: Path,
    text_feature_csv: Path,
    output_csv: Path,
    random_state: int = 42,
    rule_provider: RuleProvider | None = None,
) -> pd.DataFrame:
    provider = rule_provider or RuleProvider.default()
    bank_df = pd.read_csv(bank_csv)
    text_df = pd.read_csv(text_feature_csv)

    target_col = provider.target_column
    text_label_col = provider.text_label_column
    required_bank_cols = {target_col}
    required_text_cols = {text_label_col}
    missing_bank = sorted(required_bank_cols - set(bank_df.columns))
    missing_text = sorted(required_text_cols - set(text_df.columns))

    if missing_bank:
        raise ValueError(f"Missing required bank columns: {missing_bank}")
    if missing_text:
        raise ValueError(f"Missing required text feature columns: {missing_text}")

    merged_blocks: list[pd.DataFrame] = []
    for offset, target_value in enumerate(provider.required_target_values()):
        bank_block = bank_df[bank_df[target_col] == target_value].copy().reset_index(drop=True)
        if bank_block.empty:
            raise ValueError(f"No bank rows found for target value: {target_value}")

        text_block = sample_text_block(
            text_df=text_df.rename(columns={text_label_col: "sentiment"}),
            sample_size=len(bank_block),
            allowed_sentiments=provider.allowed_text_labels(target_value),
            random_state=random_state + offset,
        )
        text_block = text_block.add_prefix("text_")
        merged_blocks.append(pd.concat([bank_block, text_block], axis=1))

    benchmark_df = pd.concat(merged_blocks, axis=0, ignore_index=True)
    benchmark_df = benchmark_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(output_csv, index=False)
    return benchmark_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a weakly aligned multimodal benchmark from bank and finance-text features."
    )
    parser.add_argument(
        "--tabular-csv",
        "--bank-csv",
        dest="bank_csv",
        type=Path,
        default=DEFAULT_BANK_CSV,
        help="Path to the cleaned tabular CSV.",
    )
    parser.add_argument(
        "--text-feature-csv",
        type=Path,
        default=DEFAULT_TEXT_FEATURE_CSV,
        help="Path to the engineered finance-text feature CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the weakly aligned benchmark CSV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible weak alignment.",
    )
    parser.add_argument(
        "--rule-config",
        type=Path,
        default=None,
        help="Optional JSON rule provider config. Defaults to the built-in weak alignment rules.",
    )
    args = parser.parse_args()

    benchmark_df = build_weak_alignment(
        bank_csv=args.bank_csv,
        text_feature_csv=args.text_feature_csv,
        output_csv=args.output_csv,
        random_state=args.random_state,
        rule_provider=load_rule_provider(args.rule_config),
    )

    print(f"Saved weak multimodal benchmark to: {args.output_csv}")
    print(f"Rows: {len(benchmark_df)}")
    print(f"Columns: {len(benchmark_df.columns)}")
    print("Target by attached text sentiment:")
    print(pd.crosstab(benchmark_df["target"], benchmark_df["text_sentiment"], normalize="index"))


if __name__ == "__main__":
    main()
