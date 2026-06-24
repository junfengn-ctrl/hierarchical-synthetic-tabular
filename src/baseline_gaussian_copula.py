from __future__ import annotations

import argparse
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "weak_multimodal_benchmark.csv"
DEFAULT_OUTPUT_CSV = WORKSHOP_DIR / "data" / "processed" / "synthetic_gaussian_copula.csv"


def normal_cdf(values: np.ndarray) -> np.ndarray:
    vectorized_erf = np.vectorize(erf)
    return 0.5 * (1.0 + vectorized_erf(values / np.sqrt(2.0)))


def make_normal_scores(n_rows: int) -> np.ndarray:
    # Deterministic normal-like scores are enough to estimate a rank correlation matrix.
    quantiles = (np.arange(n_rows, dtype=float) + 0.5) / n_rows
    return np.sqrt(2.0) * np.array([_inverse_erf(2.0 * q - 1.0) for q in quantiles])


def _inverse_erf(x: float) -> float:
    # Winitzki approximation. This avoids adding SciPy only for inverse normal scores.
    a = 0.147
    sign = -1.0 if x < 0 else 1.0
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    return float(sign * np.sqrt(np.sqrt(first * first - second) - first))


def fit_rank_gaussian(real_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_rows = len(real_df)
    score_template = make_normal_scores(n_rows)
    normal_matrix = np.zeros((n_rows, len(real_df.columns)), dtype=float)
    sorted_values: dict[str, np.ndarray] = {}

    for col_idx, column in enumerate(real_df.columns):
        series = real_df[column]
        ranks = series.rank(method="first").astype(int).to_numpy() - 1
        normal_matrix[:, col_idx] = score_template[ranks]
        sorted_values[column] = series.sort_values(kind="mergesort").to_numpy()

    corr = np.corrcoef(normal_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = nearest_positive_semidefinite(corr)
    return corr, sorted_values


def nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-6, None)
    adjusted = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diag = np.sqrt(np.diag(adjusted))
    adjusted = adjusted / np.outer(diag, diag)
    np.fill_diagonal(adjusted, 1.0)
    return adjusted


def gaussian_copula_synthesis(
    input_csv: Path,
    output_csv: Path,
    n_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    real_df = pd.read_csv(input_csv)
    if real_df.empty:
        raise ValueError("Input CSV is empty.")

    sample_size = n_rows if n_rows is not None else len(real_df)
    corr, sorted_values = fit_rank_gaussian(real_df)

    rng = np.random.default_rng(random_state)
    sampled_normal = rng.multivariate_normal(
        mean=np.zeros(len(real_df.columns)),
        cov=corr,
        size=sample_size,
        method="eigh",
    )
    sampled_uniform = np.clip(normal_cdf(sampled_normal), 0.0, 1.0 - np.finfo(float).eps)

    synthetic = {}
    source_size = len(real_df)
    for col_idx, column in enumerate(real_df.columns):
        indices = np.floor(sampled_uniform[:, col_idx] * source_size).astype(int)
        values = sorted_values[column][indices]
        synthetic[column] = values

    synth_df = pd.DataFrame(synthetic, columns=real_df.columns)
    for column in real_df.columns:
        synth_df[column] = synth_df[column].astype(real_df[column].dtype, copy=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output_csv, index=False)
    return synth_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data with a simple Gaussian copula baseline.")
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
        help="Random seed for reproducible synthesis.",
    )
    args = parser.parse_args()

    synthetic_df = gaussian_copula_synthesis(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        n_rows=args.n_rows,
        random_state=args.random_state,
    )

    print(f"Saved Gaussian copula synthetic data to: {args.output_csv}")
    print(f"Rows: {len(synthetic_df)}")
    print(f"Columns: {len(synthetic_df.columns)}")


if __name__ == "__main__":
    main()
