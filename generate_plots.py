"""
generate_plots.py — Report figures 9, 10 and 11 from the *real* evaluation results.

Reads Resume_Screening_Project/models/evaluation_results.json (written by
`python Resume_Screening_Project/src/train_models.py` or
`python evaluate_classifiers.py`) and plots the hybrid model's measures for
each training split, in the layout the assignment template asks for:

  Figure 9  — Sensitivity, Specificity, Accuracy, F1-score
  Figure 10 — FDR, FNR, NPV, FPR
  Figure 11 — Precision, MCC

Usage:
    python generate_plots.py [--method Hybrid|TF-IDF|BERT] [--out-dir .]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT, "Resume_Screening_Project", "models", "evaluation_results.json")

FIGURES = {
    "fig9.png": ["Sensitivity", "Specificity", "Accuracy", "F1-Score"],
    "fig10.png": ["FDR", "FNR", "NPV", "FPR"],
    "fig11.png": ["Precision", "MCC"],
}
TITLES = {
    "Sensitivity": "Sensitivity (recall)", "Specificity": "Specificity", "Accuracy": "Accuracy",
    "F1-Score": "F1-score", "FDR": "False discovery rate (FDR)", "FNR": "False negative rate (FNR)",
    "NPV": "Negative predictive value (NPV)", "FPR": "False positive rate (FPR)",
    "Precision": "Precision", "MCC": "Matthews correlation coefficient (MCC)",
}
ERROR_RATES = {"FDR", "FNR", "FPR"}
GOOD = "#257de0"  # blue — higher is better
BAD = "#ec4647"   # red  — error rates (lower is better)


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#191919")
    ax.set_xlabel("Training data split", fontsize=9, color="#666666")
    ax.yaxis.grid(True, color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c2c2c2")
    ax.tick_params(colors="#666666", labelsize=9, length=0)


def plot_report_figures(results: dict, out_dir: str = ROOT, method: str = "Hybrid") -> list[str]:
    per_split = results["classification"]["per_split"][method]
    splits = list(per_split.keys())
    written = []
    for filename, metrics in FIGURES.items():
        rows, cols = (2, 2) if len(metrics) == 4 else (1, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 7.6 if rows == 2 else 3.9))
        for ax, metric in zip(axes.ravel(), metrics):
            values = [per_split[s][metric] for s in splits]
            bars = ax.bar(splits, values, width=0.5, color=BAD if metric in ERROR_RATES else GOOD)
            top = max(values)
            if top > 0.5:
                ax.set_ylim(0, 1.12 if top > 0.92 else 1.0)  # headroom so labels clear the title
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                ax.set_ylim(0, max(top * 1.35, 0.05))
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8.5, color="#191919")
            _style(ax, TITLES[metric])
        fig.suptitle(f"{method} model — {', '.join(metrics)}", fontsize=10, color="#666666", y=1.0)
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


def table3_markdown(results: dict, split: str = "80%") -> str:
    per_split = results["classification"]["per_split"]
    order = ["Specificity", "Sensitivity", "Accuracy", "Precision", "FPR", "FNR", "NPV", "FDR", "F1-Score", "MCC"]
    lines = ["| Measures | TF-IDF (Lexical) | BERT (Semantic) | Hybrid |", "| :--- | :--- | :--- | :--- |"]
    for metric in order:
        lines.append(f"| **{metric}** | " + " | ".join(f"{per_split[m][split][metric]:.3f}" for m in ("TF-IDF", "BERT", "Hybrid")) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="Hybrid", choices=["Hybrid", "TF-IDF", "BERT"])
    parser.add_argument("--out-dir", default=ROOT)
    args = parser.parse_args()
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(f"{RESULTS_PATH} not found — run `python evaluate_classifiers.py` first.")
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        results = json.load(fh)
    for path in plot_report_figures(results, args.out_dir, args.method):
        print("🖼️  wrote", os.path.relpath(path, ROOT))
    print("\nTable 3 (80 % training split):\n" + table3_markdown(results))


if __name__ == "__main__":
    main()
