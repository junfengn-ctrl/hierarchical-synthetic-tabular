# Reference Results

These CSV files are compact, versioned snapshots of the benchmark results summarized in the repository README.

| file | contents |
| --- | --- |
| `fast_experiment_results.csv` | one row per dataset, fast synthesis method, and random seed |
| `fast_experiment_summary.csv` | mean and standard deviation across fast-method seeds |
| `sdv_experiment_results.csv` | CTGAN and TVAE results |
| `sdv_experiment_summary.csv` | CTGAN and TVAE summary |
| `xgboost_ablation_summary.csv` | summarized XGBoost conditioning ablations |

Regenerate the fast-method and ablation results from the repository root:

```bash
python src/main.py prepare
python src/main.py experiments
python src/main.py ablation
```

Run SDV baselines separately:

```bash
python src/main.py sdv
```

Generated working files remain under `data/processed/` and are intentionally ignored by Git. When publishing updated results, copy only the compact result tables needed for review into this directory and update the README values and figure in the same commit.
