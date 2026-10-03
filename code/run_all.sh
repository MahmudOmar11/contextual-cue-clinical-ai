#!/usr/bin/env bash
# Reproduce every table, figure and reported estimate from the released data.
# Run from anywhere: bash code/run_all.sh   (outputs go to results/)
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

Rscript code/analysis/01_matched_comparison.R
python3 code/analysis/02_supplementary_tables.py
python3 code/analysis/03_figures_comparison.py
Rscript code/analysis/04_robustness_matched.R
python3 code/analysis/04_robustness_llm.py
python3 code/analysis/04_robustness_clinician.py
python3 code/analysis/05_physician_validation.py
python3 code/analysis/06_figure_3.py
python3 code/analysis/07_explanation_screen.py
python3 code/analysis/08_model_table.py
python3 code/analysis/09_control_experiment.py
echo "Done: results/tables, results/figures, results/estimates"
