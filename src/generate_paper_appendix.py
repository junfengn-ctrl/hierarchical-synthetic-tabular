from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_FAST_SUMMARY_CSV = WORKSHOP_DIR / "data" / "processed" / "experiments" / "experiment_summary.csv"
DEFAULT_SDV_SUMMARY_CSV = WORKSHOP_DIR / "data" / "processed" / "experiments_sdv" / "experiment_summary.csv"
DEFAULT_ABLATION_SUMMARY_CSV = (
    WORKSHOP_DIR / "data" / "processed" / "xgboost_ablation" / "xgboost_ablation_summary.csv"
)
DEFAULT_OUTPUT_DIR = WORKSHOP_DIR / "data" / "processed" / "paper_artifacts"
DEFAULT_PAPER_TEX = WORKSHOP_DIR / "tabular_synthetic.tex"
APPENDIX_START_MARKER = "% BEGIN GENERATED APPENDIX RESULTS"
APPENDIX_END_MARKER = "% END GENERATED APPENDIX RESULTS"

METHOD_ORDER = ["independent", "gaussian_copula", "random_forest", "xgboost", "ctgan", "tvae"]
METHOD_LABELS = {
    "independent": "Independent",
    "gaussian_copula": "Gaussian",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
}
DATASET_LABELS = {
    "weak_multimodal": "Manual",
    "weak_multimodal_gemini": "Gemini",
    "adult_income": "Adult",
    "german_credit": "German",
}
ABLATION_LABELS = {
    "training_rows_1000": "1k rows",
    "training_rows_3000": "3k rows",
    "condition_cols_4": "4 cols",
    "condition_cols_8": "8 cols",
    "condition_cols_12": "12 cols",
}
ABLATION_ORDER = ["training_rows_1000", "training_rows_3000", "condition_cols_4", "condition_cols_8", "condition_cols_12"]


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.4f}"


def fmt_best(value: object, is_best: bool) -> str:
    formatted = fmt(value)
    if formatted == "--" or not is_best:
        return formatted
    return rf"\textbf{{{formatted}}}"


def best_masks(df: pd.DataFrame, columns: list[str], directions: dict[str, str], group_col: str) -> dict[str, pd.Series]:
    masks = {column: pd.Series(False, index=df.index) for column in columns}
    for _, group in df.groupby(group_col, observed=False):
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            values = values.dropna()
            if values.empty:
                continue
            if directions[column] == "max":
                best_value = values.max()
            else:
                best_value = values.min()
            masks[column].loc[values.index] = np.isclose(values, best_value)
    return masks


def read_available_summaries(fast_summary_csv: Path, sdv_summary_csv: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [fast_summary_csv, sdv_summary_csv]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No experiment summary CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["dataset", "method"])


def make_latex_table(
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    column_spec: str,
    resize: bool = False,
    after_vspace: str = r"0.03in",
    before_vspace: str = r"0.02in",
) -> str:
    lines = [
        r"\begin{table}[H]",
        rf"\vspace{{{before_vspace}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        ]
    )
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    if resize:
        lines.append(r"}")
    lines.extend([r"\vspace{-0.03in}", r"\end{table}", rf"\vspace{{{after_vspace}}}", ""])
    return "\n".join(lines)


def generate_trtr_tstr_gap_table(results_df: pd.DataFrame) -> str:
    df = results_df[results_df["dataset"].isin(["weak_multimodal", "weak_multimodal_gemini"])].copy()
    metric_cols = [
        "trtr_accuracy_mean",
        "tstr_accuracy_mean",
        "gap_accuracy_mean",
        "trtr_f1_mean",
        "tstr_f1_mean",
        "gap_f1_mean",
        "trtr_auroc_mean",
        "tstr_auroc_mean",
        "gap_auroc_mean",
    ]
    masks = best_masks(df, metric_cols, {column: "max" for column in metric_cols}, "dataset")
    rows: list[list[str]] = []
    for index, row in df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                METHOD_LABELS[row["method"]],
                fmt_best(row["trtr_accuracy_mean"], bool(masks["trtr_accuracy_mean"].loc[index])),
                fmt_best(row["tstr_accuracy_mean"], bool(masks["tstr_accuracy_mean"].loc[index])),
                fmt_best(row["gap_accuracy_mean"], bool(masks["gap_accuracy_mean"].loc[index])),
                fmt_best(row["trtr_f1_mean"], bool(masks["trtr_f1_mean"].loc[index])),
                fmt_best(row["tstr_f1_mean"], bool(masks["tstr_f1_mean"].loc[index])),
                fmt_best(row["gap_f1_mean"], bool(masks["gap_f1_mean"].loc[index])),
                fmt_best(row["trtr_auroc_mean"], bool(masks["trtr_auroc_mean"].loc[index])),
                fmt_best(row["tstr_auroc_mean"], bool(masks["tstr_auroc_mean"].loc[index])),
                fmt_best(row["gap_auroc_mean"], bool(masks["gap_auroc_mean"].loc[index])),
            ]
        )
    return make_latex_table(
        caption=(
            "Weak multimodal utility diagnostics. Gap is computed as TSTR minus TRTR."
        ),
        label="tab:appendix_trtr_tstr_gap",
        columns=[
            "Benchmark",
            "Method",
            r"TRTR Acc. $\uparrow$",
            r"TSTR Acc. $\uparrow$",
            r"Gap Acc. $\uparrow$",
            r"TRTR F1 $\uparrow$",
            r"TSTR F1 $\uparrow$",
            r"Gap F1 $\uparrow$",
            r"TRTR AUROC $\uparrow$",
            r"TSTR AUROC $\uparrow$",
            r"Gap AUROC $\uparrow$",
        ],
        rows=rows,
        column_spec="llrrrrrrrrr",
        resize=True,
        after_vspace=r"0.04in",
    )


def generate_tabular_only_table(results_df: pd.DataFrame) -> str:
    df = results_df[results_df["dataset"].isin(["adult_income", "german_credit"])].copy()
    metric_cols = [
        "trtr_auroc_mean",
        "tstr_auroc_mean",
        "gap_auroc_mean",
        "tstr_accuracy_mean",
        "tstr_f1_mean",
    ]
    masks = best_masks(df, metric_cols, {column: "max" for column in metric_cols}, "dataset")
    rows: list[list[str]] = []
    for index, row in df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                METHOD_LABELS[row["method"]],
                fmt_best(row["trtr_auroc_mean"], bool(masks["trtr_auroc_mean"].loc[index])),
                fmt_best(row["tstr_auroc_mean"], bool(masks["tstr_auroc_mean"].loc[index])),
                fmt_best(row["gap_auroc_mean"], bool(masks["gap_auroc_mean"].loc[index])),
                fmt_best(row["tstr_accuracy_mean"], bool(masks["tstr_accuracy_mean"].loc[index])),
                fmt_best(row["tstr_f1_mean"], bool(masks["tstr_f1_mean"].loc[index])),
            ]
        )
    return make_latex_table(
        caption="Additional tabular-only benchmark results for Adult Income and German Credit.",
        label="tab:appendix_tabular_only",
        columns=[
            "Dataset",
            "Method",
            r"TRTR AUROC $\uparrow$",
            r"TSTR AUROC $\uparrow$",
            r"Gap AUROC $\uparrow$",
            r"TSTR Acc. $\uparrow$",
            r"TSTR F1 $\uparrow$",
        ],
        rows=rows,
        column_spec="llrrrrr",
    )


def generate_fidelity_table(results_df: pd.DataFrame) -> str:
    metric_cols = [
        "mean_numeric_abs_mean_diff",
        "mean_numeric_abs_std_diff",
        "mean_categorical_tvd",
        "mean_cross_modal_abs_diff",
    ]
    masks = best_masks(results_df, metric_cols, {column: "min" for column in metric_cols}, "dataset")
    rows: list[list[str]] = []
    for index, row in results_df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                METHOD_LABELS[row["method"]],
                fmt_best(row["mean_numeric_abs_mean_diff"], bool(masks["mean_numeric_abs_mean_diff"].loc[index])),
                fmt_best(row["mean_numeric_abs_std_diff"], bool(masks["mean_numeric_abs_std_diff"].loc[index])),
                fmt_best(row["mean_categorical_tvd"], bool(masks["mean_categorical_tvd"].loc[index])),
                fmt_best(row["mean_cross_modal_abs_diff"], bool(masks["mean_cross_modal_abs_diff"].loc[index])),
            ]
        )
    return make_latex_table(
        caption=(
            "Additional fidelity diagnostics. Numeric columns report average absolute differences "
            "in means and standard deviations; categorical TVD is the average total variation "
            "distance across categorical columns; XModal is reported only for weak multimodal benchmarks."
        ),
        label="tab:appendix_fidelity",
        columns=[
            "Dataset",
            "Method",
            r"Num. mean diff $\downarrow$",
            r"Num. std diff $\downarrow$",
            r"Cat. TVD $\downarrow$",
            r"XModal $\downarrow$",
        ],
        rows=rows,
        column_spec="llrrrr",
        after_vspace=r"-0.04in",
    )


def generate_ablation_table(ablation_summary_csv: Path) -> str:
    if not ablation_summary_csv.exists():
        return ""
    df = pd.read_csv(ablation_summary_csv)
    df = df[df["dataset"].isin(["weak_multimodal", "weak_multimodal_gemini"])].copy()
    df["ablation"] = pd.Categorical(df["ablation"], categories=ABLATION_ORDER, ordered=True)
    df = df.sort_values(["dataset", "ablation"])
    metric_cols = [
        "trtr_auroc_mean",
        "tstr_auroc_mean",
        "gap_auroc_mean",
        "tstr_accuracy_mean",
        "tstr_f1_mean",
    ]
    masks = best_masks(df, metric_cols, {column: "max" for column in metric_cols}, "dataset")

    rows: list[list[str]] = []
    for index, row in df.iterrows():
        rows.append(
            [
                DATASET_LABELS[row["dataset"]],
                ABLATION_LABELS[str(row["ablation"])],
                fmt_best(row["trtr_auroc_mean"], bool(masks["trtr_auroc_mean"].loc[index])),
                fmt_best(row["tstr_auroc_mean"], bool(masks["tstr_auroc_mean"].loc[index])),
                fmt_best(row["gap_auroc_mean"], bool(masks["gap_auroc_mean"].loc[index])),
                fmt_best(row["tstr_accuracy_mean"], bool(masks["tstr_accuracy_mean"].loc[index])),
                fmt_best(row["tstr_f1_mean"], bool(masks["tstr_f1_mean"].loc[index])),
            ]
        )
    return make_latex_table(
        caption=(
            "Full XGBoost ablation results. Training-row ablations use the default conditioning setting; "
            "conditioning-column ablations use the default training-row setting."
        ),
        label="tab:appendix_xgboost_ablation_full",
        columns=[
            "Benchmark",
            "Setting",
            r"TRTR AUROC $\uparrow$",
            r"TSTR AUROC $\uparrow$",
            r"Gap AUROC $\uparrow$",
            r"TSTR Acc. $\uparrow$",
            r"TSTR F1 $\uparrow$",
        ],
        rows=rows,
        column_spec="llrrrrr",
    )


def generate_tabular_utility_plot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    df = results_df[results_df["dataset"].isin(["adult_income", "german_credit"])].copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)

    methods = [METHOD_LABELS[method] for method in METHOD_ORDER]
    datasets = ["Adult", "German"]
    colors = {"Adult": "#54A24B", "German": "#B279A2"}
    metrics = [
        ("tstr_accuracy_mean", "TSTR accuracy ↑", (0.0, 1.05)),
        ("tstr_f1_mean", "TSTR F1 ↑", (0.0, 1.05)),
        ("tstr_auroc_mean", "TSTR AUROC ↑", (0.0, 1.05)),
    ]
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for offset, dataset in zip([-width / 2, width / 2], datasets):
            subset = df[df["dataset_label"] == dataset].set_index("method")
            values = [
                float(subset.loc[method, metric]) if method in subset.index else np.nan
                for method in METHOD_ORDER
            ]
            ax.bar(x + offset, values, width=width, label=dataset, color=colors[dataset])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")
    output_path = output_dir / "appendix_tabular_utility.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def generate_fidelity_plot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    df = results_df[results_df["dataset"].isin(["weak_multimodal", "weak_multimodal_gemini"])].copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)

    methods = [METHOD_LABELS[method] for method in METHOD_ORDER]
    datasets = ["Manual", "Gemini"]
    colors = {"Manual": "#4C78A8", "Gemini": "#F58518"}
    metrics = [
        ("mean_categorical_tvd", "Categorical TVD", (0.0, 0.14)),
        ("mean_cross_modal_abs_diff", "XModal", (0.0, 0.36)),
    ]
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for offset, dataset in zip([-width / 2, width / 2], datasets):
            subset = df[df["dataset_label"] == dataset].set_index("method")
            values = [
                float(subset.loc[method, metric]) if method in subset.index else np.nan
                for method in METHOD_ORDER
            ]
            ax.bar(x + offset, values, width=width, label=dataset, color=colors[dataset])
        ax.set_title(f"{title} ↓", fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")
    output_path = output_dir / "appendix_fidelity_comparison.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def generate_ablation_plot(ablation_summary_csv: Path, output_dir: Path) -> Path | None:
    if not ablation_summary_csv.exists():
        return None
    df = pd.read_csv(ablation_summary_csv)
    df = df[df["dataset"].isin(["weak_multimodal", "weak_multimodal_gemini"])].copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    df["setting_label"] = df["ablation"].map(ABLATION_LABELS)
    df["ablation"] = pd.Categorical(df["ablation"], categories=ABLATION_ORDER, ordered=True)
    df = df.sort_values(["dataset", "ablation"])

    settings = [ABLATION_LABELS[setting] for setting in ABLATION_ORDER]
    datasets = ["Manual", "Gemini"]
    colors = {"Manual": "#4C78A8", "Gemini": "#F58518"}
    metrics = [
        ("tstr_auroc_mean", "TSTR AUROC", (0.72, 1.02)),
        ("tstr_f1_mean", "TSTR F1", (0.0, 1.05)),
    ]
    x = np.arange(len(settings))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for offset, dataset in zip([-width / 2, width / 2], datasets):
            subset = df[df["dataset_label"] == dataset].set_index("ablation")
            values = [
                float(subset.loc[setting, metric]) if setting in subset.index else np.nan
                for setting in ABLATION_ORDER
            ]
            ax.bar(x + offset, values, width=width, label=dataset, color=colors[dataset])
        ax.set_title(f"{title} ↑", fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")
    output_path = output_dir / "appendix_xgboost_ablation.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def generate_gap_plot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    df = results_df[results_df["dataset"].isin(DATASET_LABELS)].copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    df["method_label"] = df["method"].map(METHOD_LABELS)

    methods = [METHOD_LABELS[method] for method in METHOD_ORDER]
    datasets = ["Manual", "Gemini"]
    colors = {"Manual": "#4C78A8", "Gemini": "#F58518"}
    metrics = [
        ("gap_accuracy_mean", "Accuracy gap ↑", (-0.12, 0.0)),
        ("gap_f1_mean", "F1 gap ↑", (-0.70, 0.0)),
        ("gap_auroc_mean", "AUROC gap ↑", (-0.50, 0.0)),
    ]
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for offset, dataset in zip([-width / 2, width / 2], datasets):
            subset = df[df["dataset_label"] == dataset].set_index("method")
            values = []
            for method in METHOD_ORDER:
                if method in subset.index:
                    values.append(float(subset.loc[method, metric]))
                else:
                    values.append(np.nan)
            ax.bar(x + offset, values, width=width, label=dataset, color=colors[dataset])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("TSTR - TRTR")

    axes[0].legend(frameon=False, loc="lower left")

    output_path = output_dir / "appendix_utility_gap.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def make_figure_latex(
    path: Path,
    caption: str,
    label: str,
    width: str = r"0.95\textwidth",
    before_vspace: str = r"0.03in",
    after_vspace: str = r"-0.06in",
) -> str:
    figure_filename = path.name
    return "\n".join(
        [
            r"\begin{figure}[H]",
            rf"\vspace{{{before_vspace}}}",
            r"\centering",
            rf"\includegraphics[width={width}]{{{figure_filename}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\vskip -0.08in % Compact appendix spacing without attaching caption to text",
            r"\end{figure}",
            rf"\vspace{{{after_vspace}}}",
            "",
        ]
    )


def build_appendix_latex(
    results_df: pd.DataFrame,
    ablation_summary_csv: Path,
    utility_gap_figure: Path,
    tabular_utility_figure: Path,
    fidelity_figure: Path,
    ablation_figure: Path | None,
) -> str:
    parts = [
        APPENDIX_START_MARKER,
        r"\section{Additional Empirical Results}",
        r"\label{sec:appendix_results}",
        r"\setlength{\textfloatsep}{6pt}",
        r"\setlength{\floatsep}{6pt}",
        r"\setlength{\intextsep}{6pt}",
        r"\captionsetup[table]{skip=4pt}",
        r"\captionsetup[figure]{skip=5pt}",
        "",
        (
            r"All appendix tables and figures are generated directly from the experiment "
            r"summary CSV files produced by the pipeline, without manually editing numeric values."
        ),
        "",
        r"\subsection{TRTR, TSTR, and Gap Diagnostics}",
        "",
        (
            r"The main paper reports TSTR utility metrics in Tables~\ref{tab:manual_results} "
            r"and~\ref{tab:gemini_results}. For completeness, Appendix "
            r"Table~\ref{tab:appendix_trtr_tstr_gap} reports the corresponding TRTR reference, "
            r"TSTR value, and gap for accuracy, F1, and AUROC. Each gap is computed as TSTR "
            r"minus TRTR for the same metric, so values closer to zero indicate that training "
            r"on synthetic data approaches the performance of training on real data."
        ),
        "",
        generate_trtr_tstr_gap_table(results_df),
        make_figure_latex(
            utility_gap_figure,
            (
                r"Weak multimodal utility gaps. Bars closer to zero indicate smaller "
                r"degradation relative to the real-data reference."
            ),
            "fig:appendix_utility_gap",
            width=r"0.96\textwidth",
            before_vspace=r"0.02in",
            after_vspace=r"0.03in",
        ),
        "",
        r"\subsection{Tabular-Only Benchmark Results}",
        "",
        (
            r"Table~\ref{tab:appendix_tabular_only} reports additional results for the "
            r"tabular-only Adult Income and German Credit benchmarks. These results are used "
            r"as a robustness check because they remove the weak text-tabular alignment layer "
            r"and evaluate whether the same synthesis pipeline behaves reasonably on standard "
            r"tabular datasets."
        ),
        "",
        generate_tabular_only_table(results_df),
        make_figure_latex(
            tabular_utility_figure,
            (
                r"Tabular-only TSTR utility comparison on Adult Income and German Credit. "
                r"Higher accuracy, F1, and AUROC indicate better downstream utility."
            ),
            "fig:appendix_tabular_utility",
        ),
        r"\subsection{Fidelity Diagnostics}",
        "",
        (
            r"Table~\ref{tab:appendix_fidelity} reports additional fidelity metrics for all "
            r"benchmarks and methods. Figure~\ref{fig:appendix_fidelity_comparison} visualizes "
            r"the categorical and cross-modal fidelity metrics on the weak multimodal benchmarks."
        ),
        "",
        generate_fidelity_table(results_df),
        make_figure_latex(
            fidelity_figure,
            (
                r"Weak multimodal fidelity comparison. Lower categorical TVD and lower XModal "
                r"indicate closer agreement between real and synthetic data."
            ),
            "fig:appendix_fidelity_comparison",
            width=r"0.72\textwidth",
            before_vspace=r"-0.01in",
            after_vspace=r"0.02in",
        ),
        r"\subsection{Full XGBoost Ablation}",
        "",
        (
            r"Table~\ref{tab:appendix_xgboost_ablation_full} gives the full XGBoost ablation "
            r"used to produce the summary in Table~\ref{tab:xgboost_ablation}. The ablation "
            r"varies either the number of training rows or the number of conditioning columns "
            r"while keeping the remaining evaluation protocol fixed."
        ),
        "",
        generate_ablation_table(ablation_summary_csv),
        make_figure_latex(
            ablation_figure,
            (
                r"XGBoost ablation comparison. Training-row settings vary the amount of real data "
                r"used to fit the generator, while conditioning-column settings vary the number of "
                r"previous columns used during sequential conditional synthesis."
            ),
            "fig:appendix_xgboost_ablation",
            width=r"0.82\textwidth",
        )
        if ablation_figure is not None
        else "",
        APPENDIX_END_MARKER,
    ]
    return "\n".join(part for part in parts if part != "")


def update_paper_tex(paper_tex: Path, appendix_latex: str) -> Path:
    text = paper_tex.read_text(encoding="utf-8")
    if APPENDIX_START_MARKER not in text or APPENDIX_END_MARKER not in text:
        raise ValueError(
            f"Could not find appendix markers in {paper_tex}. "
            f"Expected {APPENDIX_START_MARKER!r} and {APPENDIX_END_MARKER!r}."
        )
    before, rest = text.split(APPENDIX_START_MARKER, 1)
    _, after = rest.split(APPENDIX_END_MARKER, 1)
    paper_tex.write_text(before + appendix_latex + after, encoding="utf-8")
    return paper_tex


def generate_appendix_artifacts(
    fast_summary_csv: Path = DEFAULT_FAST_SUMMARY_CSV,
    sdv_summary_csv: Path = DEFAULT_SDV_SUMMARY_CSV,
    ablation_summary_csv: Path = DEFAULT_ABLATION_SUMMARY_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    paper_tex: Path = DEFAULT_PAPER_TEX,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = read_available_summaries(fast_summary_csv, sdv_summary_csv)

    utility_gap_figure = generate_gap_plot(results_df, output_dir)
    tabular_utility_figure = generate_tabular_utility_plot(results_df, output_dir)
    fidelity_figure = generate_fidelity_plot(results_df, output_dir)
    ablation_figure = generate_ablation_plot(ablation_summary_csv, output_dir)
    appendix_latex = build_appendix_latex(
        results_df,
        ablation_summary_csv,
        utility_gap_figure,
        tabular_utility_figure,
        fidelity_figure,
        ablation_figure,
    )
    updated_paper = update_paper_tex(paper_tex, appendix_latex)
    generated = [utility_gap_figure, tabular_utility_figure, fidelity_figure, updated_paper]
    if ablation_figure is not None:
        generated.append(ablation_figure)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible LaTeX appendix tables and figures.")
    parser.add_argument("--fast-summary-csv", type=Path, default=DEFAULT_FAST_SUMMARY_CSV)
    parser.add_argument("--sdv-summary-csv", type=Path, default=DEFAULT_SDV_SUMMARY_CSV)
    parser.add_argument("--ablation-summary-csv", type=Path, default=DEFAULT_ABLATION_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-tex", type=Path, default=DEFAULT_PAPER_TEX)
    args = parser.parse_args()

    generated = generate_appendix_artifacts(
        fast_summary_csv=args.fast_summary_csv,
        sdv_summary_csv=args.sdv_summary_csv,
        ablation_summary_csv=args.ablation_summary_csv,
        output_dir=args.output_dir,
        paper_tex=args.paper_tex,
    )
    for path in generated:
        print(f"Saved appendix artifact: {path}")


if __name__ == "__main__":
    main()
