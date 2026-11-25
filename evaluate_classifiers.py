"""
evaluate_classifiers.py — Re-run the full experimental evaluation.

Uses the artifacts produced by `Resume_Screening_Project/src/train_models.py`
and the shared evaluation code in `Resume_Screening_Project/src/evaluation.py`:

  A. Category classification with linear SVMs on TF-IDF, BERT and hybrid
     features for 50–90 % training splits (all ten evaluation measures).
  B. Hyper-parameter sweep of the SVM's C (5-fold CV, train vs. validation).
  C. JD → resume ranking benchmark on held-out resumes.

Writes Resume_Screening_Project/models/evaluation_results.json (shown on the
dashboard's "Model evaluation" page), regenerates fig9.png / fig10.png /
fig11.png and prints Table 3 for the report.

Usage (from the repository root):
    python evaluate_classifiers.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Resume_Screening_Project", "src"))

import pandas as pd  # noqa: E402

from engine import MODEL_DIR, ScreeningEngine  # noqa: E402
from evaluation import run_full_evaluation  # noqa: E402
from generate_plots import RESULTS_PATH, plot_report_figures, table3_markdown  # noqa: E402
from sample_data import SAMPLE_JOBS  # noqa: E402


def main() -> None:
    print("🧪 Loading trained artifacts …")
    engine = ScreeningEngine(MODEL_DIR)
    holdout = pd.read_pickle(os.path.join(MODEL_DIR, "demo_resumes.pkl"))
    results = run_full_evaluation(engine, holdout, SAMPLE_JOBS)
    results["created_at"] = engine.meta.get("created_at")
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"💾 Saved {os.path.relpath(RESULTS_PATH, ROOT)}")

    print("📊 Generating report figures from the measured results …")
    for path in plot_report_figures(results, ROOT):
        print("   wrote", os.path.relpath(path, ROOT))

    print("\n--- TABLE 3 (80 % training split) ---")
    print(table3_markdown(results))
    print("\n--- RANKING BENCHMARK (held-out resumes) ---")
    for method, metrics in results["ranking"]["summary"].items():
        print(f"{method:9}", "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
