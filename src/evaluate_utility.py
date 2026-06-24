from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REAL_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_SYNTH_CSV = WORKSHOP_DIR / "data" / "processed" / "synthetic_independent_sampling.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "utility_report.csv"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, categorical_cols),
            ("numeric", numeric_pipe, numeric_cols),
        ]
    )


def build_model(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_binary_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
    }


def evaluate_constant_binary_classifier(constant_class: int, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = np.full(len(X_test), constant_class)
    y_prob = np.full(len(X_test), float(constant_class))
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
    }


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    require_both_classes: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    leakage_cols = {target_col}
    # Exclude columns that directly reveal the binary target.
    if target_col == "target":
        leakage_cols.update({"y", "salary", "credit_risk"} & set(df.columns))
    X = df.drop(columns=sorted(leakage_cols))
    y = df[target_col].astype(int)
    unique_values = sorted(y.dropna().unique())
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Expected target values to be a subset of [0, 1] in '{target_col}', found: {unique_values}"
        )
    if require_both_classes and unique_values != [0, 1]:
        raise ValueError(f"Expected both binary target classes [0, 1] in '{target_col}', found: {unique_values}")
    return X, y


def evaluate_utility(
    real_csv: Path,
    synth_csv: Path,
    output_csv: Path,
    target_col: str = "target",
    test_size: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    real_df = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)

    X_real, y_real = prepare_features_and_target(real_df, target_col, require_both_classes=True)
    X_synth, y_synth = prepare_features_and_target(synth_df, target_col, require_both_classes=False)

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real,
        y_real,
        test_size=test_size,
        random_state=random_state,
        stratify=y_real,
    )

    # Train on real, test on real
    real_model = build_model(X_real_train)
    real_model.fit(X_real_train, y_real_train)
    trtr_metrics = evaluate_binary_classifier(real_model, X_real_test, y_real_test)

    # Train on synthetic, test on real. If the synthetic target collapses to one
    # class, evaluate it as a constant classifier instead of crashing the run.
    synth_classes = sorted(y_synth.dropna().unique())
    if len(synth_classes) == 1:
        tstr_metrics = evaluate_constant_binary_classifier(int(synth_classes[0]), X_real_test, y_real_test)
    else:
        synth_model = build_model(X_synth)
        synth_model.fit(X_synth, y_synth)
        tstr_metrics = evaluate_binary_classifier(synth_model, X_real_test, y_real_test)

    report_df = pd.DataFrame(
        [
            {
                "setting": "train_real_test_real",
                **trtr_metrics,
            },
            {
                "setting": "train_synthetic_test_real",
                **tstr_metrics,
            },
            {
                "setting": "gap_tstr_minus_trtr",
                "accuracy": tstr_metrics["accuracy"] - trtr_metrics["accuracy"],
                "f1": tstr_metrics["f1"] - trtr_metrics["f1"],
                "auroc": tstr_metrics["auroc"] - trtr_metrics["auroc"],
            },
        ]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_csv, index=False)
    return report_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downstream utility of synthetic data.")
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
        help="Path to save the utility report CSV.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of real data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    args = parser.parse_args()

    report_df = evaluate_utility(
        real_csv=args.real_csv,
        synth_csv=args.synth_csv,
        output_csv=args.output_csv,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Saved utility report to: {args.output_csv}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
