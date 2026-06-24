from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "synthetic_xgboost.csv"


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


def sample_first_column(series: pd.Series, n_rows: int, random_state: int) -> pd.Series:
    return series.sample(n=n_rows, replace=True, random_state=random_state).reset_index(drop=True)


def conditional_xgboost_synthesis(
    input_csv: Path,
    output_csv: Path,
    n_rows: int | None = None,
    random_state: int = 42,
    max_training_rows: int = 5000,
    max_condition_cols: int = 12,
    n_estimators: int = 80,
    max_depth: int = 6,
) -> pd.DataFrame:
    real_df = pd.read_csv(input_csv)
    if real_df.empty:
        raise ValueError("Input benchmark CSV is empty.")

    sample_size = n_rows if n_rows is not None else len(real_df)
    if len(real_df) > max_training_rows:
        train_df = real_df.sample(n=max_training_rows, random_state=random_state).reset_index(drop=True)
    else:
        train_df = real_df.copy()

    column_order = list(real_df.columns)
    categorical_cols, numeric_cols = infer_column_types(real_df)

    synthetic_df = pd.DataFrame(index=range(sample_size))
    synthetic_df[column_order[0]] = sample_first_column(
        train_df[column_order[0]],
        n_rows=sample_size,
        random_state=random_state,
    )

    for step_idx, target_col in enumerate(column_order[1:], start=1):
        feature_cols = column_order[max(0, step_idx - max_condition_cols):step_idx]
        X_train = pd.get_dummies(train_df[feature_cols], drop_first=False)
        X_gen = pd.get_dummies(synthetic_df[feature_cols], drop_first=False)
        X_gen = X_gen.reindex(columns=X_train.columns, fill_value=0)

        if target_col in categorical_cols:
            y_train = train_df[target_col].astype(str)
            class_codes, class_labels = pd.factorize(y_train, sort=True)

            if len(class_labels) == 1:
                synthetic_df[target_col] = class_labels[0]
                continue

            if len(class_labels) == 2:
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state + step_idx,
                    n_jobs=1,
                    verbosity=0,
                )
                model.fit(X_train, class_codes)
                class_prob = model.predict_proba(X_gen)
            else:
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=len(class_labels),
                    random_state=random_state + step_idx,
                    n_jobs=1,
                    verbosity=0,
                )
                model.fit(X_train, class_codes)
                class_prob = model.predict_proba(X_gen)

            sampled_values = []
            label_series = pd.Series(class_labels)
            for row_idx, probs in enumerate(class_prob):
                sampled_values.append(
                    label_series.sample(
                        n=1,
                        weights=probs,
                        replace=True,
                        random_state=random_state + step_idx + row_idx,
                    ).iloc[0]
                )
            synthetic_df[target_col] = sampled_values
        else:
            y_train = train_df[target_col]
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state + step_idx,
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_gen)

            train_predictions = model.predict(X_train)
            residuals = y_train.to_numpy() - train_predictions
            residual_sample = pd.Series(residuals).sample(
                n=sample_size,
                replace=True,
                random_state=random_state + 1000 + step_idx,
            ).reset_index(drop=True)
            synthetic_df[target_col] = predictions + residual_sample

    for column in numeric_cols:
        if pd.api.types.is_integer_dtype(real_df[column]):
            synthetic_df[column] = synthetic_df[column].round().astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df = synthetic_df[column_order]
    synthetic_df.to_csv(output_csv, index=False)
    return synthetic_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset with sequential XGBoost conditional synthesis."
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
        help="Random seed for reproducible synthesis.",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=5000,
        help="Maximum number of real rows to use when fitting the sequential XGBoost models.",
    )
    parser.add_argument(
        "--max-condition-cols",
        type=int,
        default=12,
        help="Maximum number of previously generated columns to condition on for each new column.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=80,
        help="Number of boosting rounds per XGBoost model.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth for each XGBoost model.",
    )
    args = parser.parse_args()

    synthetic_df = conditional_xgboost_synthesis(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        n_rows=args.n_rows,
        random_state=args.random_state,
        max_training_rows=args.max_training_rows,
        max_condition_cols=args.max_condition_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    print(f"Saved XGBoost synthetic data to: {args.output_csv}")
    print(f"Rows: {len(synthetic_df)}")
    print(f"Columns: {len(synthetic_df.columns)}")
    print("Target by attached text sentiment:")
    print(pd.crosstab(synthetic_df["target"], synthetic_df["text_sentiment"], normalize="index"))


if __name__ == "__main__":
    main()
