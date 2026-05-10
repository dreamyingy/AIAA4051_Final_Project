import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


warnings.filterwarnings("ignore")

# Plot style configuration
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})

# Dataset category mapping
TASK_MAPPING = {
    "sciq": "Short-form",
    "simple_questions_wiki": "Short-form",
    "nq": "Long-form",
    "truthfulQA": "Long-form",
}
RESULTS_DIR = Path("results")


def load_data():
    """Load similarity analysis results with basic diagnostics."""
    dfs, summaries = [], []
    found_datasets, missing_datasets = [], []

    for ds, task_type in TASK_MAPPING.items():
        csv_path = RESULTS_DIR / ds / "similarity_scores.csv"
        json_path = RESULTS_DIR / ds / "similarity_summary.json"

        if csv_path.exists() and json_path.exists():
            df = pd.read_csv(csv_path)
            df["task_type"] = task_type
            dfs.append(df)
            found_datasets.append(ds)

            with open(json_path, "r", encoding="utf-8") as f:
                summaries.append({**json.load(f), "task_type": task_type})
        else:
            missing_datasets.append(ds)

    print("\nDirectory diagnostics:")
    found_text = found_datasets if found_datasets else "none"
    print(f"  Found datasets ({len(found_datasets)}/4): {found_text}")
    if missing_datasets:
        print(f"  Missing datasets ({len(missing_datasets)}/4): {missing_datasets}")
        print("  Suggested fix: run the analysis script with the correct results path:")
        print(f"     python part2_similarity_analysis.py --datasets {' '.join(missing_datasets)} --results_dir {RESULTS_DIR}")

    if not dfs:
        raise FileNotFoundError("No analysis result files found. Run the evaluation analysis script first.")

    print(f"  Loaded detailed results for {len(dfs)} datasets.\n")
    return pd.concat(dfs, ignore_index=True), summaries


def plot_similarity_distribution(df):
    """Plot 1: Compare similarity distributions by task type and correctness."""
    plt.figure(figsize=(9, 6))
    for task in ["Short-form", "Long-form"]:
        sub = df[df["task_type"] == task]
        line_style = "--" if task == "Long-form" else "-"
        for label, color in zip([True, False], ["#2ca02c", "#d62728"]):
            sns.kdeplot(
                data=sub[sub["is_correct"] == label],
                x="cosine_similarity",
                label=f"{task} - {'Correct' if label else 'Incorrect'}",
                color=color,
                fill=True,
                alpha=0.4,
                linewidth=2,
                linestyle=line_style,
            )
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.6, label="Similarity=0.5")
    plt.title("Distribution of Cosine Similarity by Task Type & Correctness")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(title="Task / Label", loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_1_distribution.png", dpi=300)
    plt.show()


def plot_roc_curves(df):
    """Plot 2: Compare ROC curves for similarity-based correctness prediction."""
    plt.figure(figsize=(8, 6))
    for ds, task_type in TASK_MAPPING.items():
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        fpr, tpr, _ = roc_curve(sub["is_correct"].astype(int), sub["cosine_similarity"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2.5, label=f"{ds} ({task_type}) | AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Can Similarity Predict Answer Correctness?")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_2_roc.png", dpi=300)
    plt.show()


def plot_correlation(df):
    """Plot 3: HHEM score versus cosine similarity with regression lines."""
    # Use lmplot instead of regplot to support hue.
    g = sns.lmplot(
        data=df,
        x="hhem_score",
        y="cosine_similarity",
        hue="task_type",
        scatter_kws={"alpha": 0.4, "s": 40, "edgecolor": "w"},
        line_kws={"linewidth": 2.5},
        ci=95,
        palette="Set1",
        height=6,
        aspect=1.4,
    )
    # Set axis labels and title.
    g.set_axis_labels("HHEM Score (Ground Truth Correctness)", "Predicted Cosine Similarity")
    g.fig.suptitle("Correlation: HHEM Factuality Score vs Cosine Similarity", y=0.97, fontsize=13)
    g.fig.subplots_adjust(top=0.90)

    # Save the FacetGrid figure returned by lmplot.
    g.fig.savefig(RESULTS_DIR / "plot_3_correlation.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric_comparison(summaries):
    """Plot 4: Compare core metrics, including AUC, AP, and best F1."""
    records = []
    for s in summaries:
        best_f1 = s["best_f1_similarity_threshold"]
        records.append({
            "dataset": s["dataset"],
            "task_type": s["task_type"],
            "roc_auc": s.get("roc_auc_similarity_predicts_correct", 0),
            "avg_precision": s.get("average_precision_similarity_predicts_correct", 0),
            "best_f1": best_f1["f1"] if best_f1 else 0,
            "best_threshold": best_f1["threshold"] if best_f1 else 0,
        })
    df_m = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(df_m))
    w = 0.25
    plt.bar(x - w, df_m["roc_auc"], w, label="ROC-AUC", color="#1f77b4", alpha=0.85)
    plt.bar(x, df_m["avg_precision"], w, label="Avg Precision", color="#2ca02c", alpha=0.85)
    plt.bar(x + w, df_m["best_f1"], w, label="Best F1", color="#ff7f0e", alpha=0.85)

    plt.xticks(x, [f"{d}\n({t})" for d, t in zip(df_m["dataset"], df_m["task_type"])])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Predictive Performance Metrics by Task Type")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_4_metrics.png", dpi=300)
    plt.show()
    return df_m


def print_comparison_report(df_m, df):
    """Print a quantitative short-form versus long-form QA comparison."""
    print("\n" + "=" * 60)
    print("SHORT-FORM vs LONG-FORM QA COMPARISON REPORT")
    print("=" * 60)
    for metric in ["roc_auc", "avg_precision", "best_f1"]:
        short_avg = df_m[df_m["task_type"] == "Short-form"][metric].mean()
        long_avg = df_m[df_m["task_type"] == "Long-form"][metric].mean()
        gap = short_avg - long_avg
        print(f"- {metric.upper():<15} | Short: {short_avg:.3f} | Long: {long_avg:.3f} | Delta: {gap:+.3f}")

    # Similarity distribution statistics
    print("-" * 60)
    for task in ["Short-form", "Long-form"]:
        sub = df[df["task_type"] == task]
        mean_sim = sub["cosine_similarity"].mean()
        corr = np.corrcoef(sub["hhem_score"], sub["cosine_similarity"])[0, 1]
        print(f"- {task:<10} | Mean Similarity: {mean_sim:.3f} | Pearson Corr: {corr:.3f}")
    print("=" * 60 + "\n")


def main():
    print("Loading data from results/ ...")
    df, summaries = load_data()

    print("Generating plots...")
    plot_similarity_distribution(df)
    plot_roc_curves(df)
    plot_correlation(df)
    df_m = plot_metric_comparison(summaries)

    print_comparison_report(df_m, df)
    print("All figures saved to `results/` directory.")


if __name__ == "__main__":
    main()
