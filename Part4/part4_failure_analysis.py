import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASETS = ["sciq", "simple_questions_wiki", "nq", "truthfulQA"]


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text):
    return re.findall(r"\b\d+(?:\.\d+)?\b", str(text))


def has_negation(text):
    text = normalize_text(text)
    neg_words = {"no", "not", "never", "none", "neither", "nor", "cannot", "can't", "without"}
    return any(w in text.split() for w in neg_words)


def token_jaccard(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def guess_failure_type(row):
    pred = str(row.get("merged_prediction", ""))
    ref = str(row.get("merged_true_answer", ""))

    pred_nums = extract_numbers(pred)
    ref_nums = extract_numbers(ref)

    if pred_nums != ref_nums and (pred_nums or ref_nums):
        return "numeric_or_date_mismatch"

    if has_negation(pred) != has_negation(ref):
        return "negation_or_contradiction"

    jac = token_jaccard(pred, ref)

    if row["failure_type_auto"] == "false_positive_high_similarity_wrong":
        if jac > 0.65:
            return "entity_or_relation_mismatch"
        else:
            return "semantic_ambiguity"

    if row["failure_type_auto"] == "false_negative_low_similarity_correct":
        if jac < 0.35:
            return "lexical_variation_or_paraphrase"
        else:
            return "overly_strict_threshold_or_hhem_artifact"

    return "not_failure"


def load_thresholds(results_dir):
    summary_path = results_dir / "part2_similarity_summary.json"

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        return {
            item["dataset"]: item["best_f1_similarity_threshold"]["threshold"]
            for item in summaries
        }

    thresholds = {}
    for dataset in DATASETS:
        path = results_dir / dataset / "similarity_summary.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
            thresholds[dataset] = item["best_f1_similarity_threshold"]["threshold"]

    if not thresholds:
        raise FileNotFoundError(
            "Cannot find part2_similarity_summary.json or dataset-level similarity_summary.json files."
        )

    return thresholds


def load_scores(results_dir):
    dfs = []
    for dataset in DATASETS:
        path = results_dir / dataset / "similarity_scores.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"[Warning] Missing file: {path}")

    if not dfs:
        raise FileNotFoundError("No similarity_scores.csv files found under results/{dataset}/.")

    return pd.concat(dfs, ignore_index=True)


def analyze(results_dir, out_dir, top_k):
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(results_dir)
    df = load_scores(results_dir)

    df["is_correct"] = df["is_correct"].apply(str_to_bool)
    df["dataset_threshold"] = df["dataset"].map(thresholds)
    df["similarity_predict_correct"] = df["cosine_similarity"] >= df["dataset_threshold"]

    conditions = [
        (~df["is_correct"]) & (df["similarity_predict_correct"]),
        (df["is_correct"]) & (~df["similarity_predict_correct"]),
        (df["is_correct"]) & (df["similarity_predict_correct"]),
        (~df["is_correct"]) & (~df["similarity_predict_correct"]),
    ]

    names = [
        "false_positive_high_similarity_wrong",
        "false_negative_low_similarity_correct",
        "true_positive",
        "true_negative",
    ]

    df["failure_type_auto"] = np.select(conditions, names, default="unknown")
    df["candidate_failure_reason"] = df.apply(guess_failure_type, axis=1)
    df["token_jaccard_merged"] = df.apply(
        lambda r: token_jaccard(r.get("merged_prediction", ""), r.get("merged_true_answer", "")),
        axis=1,
    )

    # Save all samples with labels
    all_path = out_dir / "all_scores_with_failure_flags.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")

    # Confusion summary
    summary_rows = []
    for dataset, g in df.groupby("dataset"):
        tp = ((g["failure_type_auto"] == "true_positive")).sum()
        tn = ((g["failure_type_auto"] == "true_negative")).sum()
        fp = ((g["failure_type_auto"] == "false_positive_high_similarity_wrong")).sum()
        fn = ((g["failure_type_auto"] == "false_negative_low_similarity_correct")).sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        summary_rows.append({
            "dataset": dataset,
            "threshold": thresholds[dataset],
            "num_samples": len(g),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive_high_similarity_wrong": fp,
            "false_negative_low_similarity_correct": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_similarity_correct": g.loc[g["is_correct"], "cosine_similarity"].mean(),
            "mean_similarity_incorrect": g.loc[~g["is_correct"], "cosine_similarity"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "part4_confusion_summary.csv", index=False, encoding="utf-8-sig")

    # Select top failure cases for manual inspection
    selected_cases = []

    for dataset, g in df.groupby("dataset"):
        threshold = thresholds[dataset]

        fp_cases = g[g["failure_type_auto"] == "false_positive_high_similarity_wrong"].copy()
        fp_cases["distance_from_threshold"] = fp_cases["cosine_similarity"] - threshold
        fp_cases = fp_cases.sort_values(
            ["cosine_similarity", "distance_from_threshold"],
            ascending=False
        ).head(top_k)

        fn_cases = g[g["failure_type_auto"] == "false_negative_low_similarity_correct"].copy()
        fn_cases["distance_from_threshold"] = threshold - fn_cases["cosine_similarity"]
        fn_cases = fn_cases.sort_values(
            ["cosine_similarity", "distance_from_threshold"],
            ascending=[True, False]
        ).head(top_k)

        selected_cases.append(fp_cases)
        selected_cases.append(fn_cases)

    cases_df = pd.concat(selected_cases, ignore_index=True)

    # Add blank columns for manual annotation
    cases_df["manual_failure_type"] = ""
    cases_df["manual_notes"] = ""
    cases_df["should_use_in_report"] = ""

    cols = [
        "dataset",
        "index",
        "failure_type_auto",
        "candidate_failure_reason",
        "manual_failure_type",
        "manual_notes",
        "should_use_in_report",
        "question",
        "prediction",
        "true_answer",
        "merged_prediction",
        "merged_true_answer",
        "hhem_score",
        "is_correct",
        "cosine_similarity",
        "dataset_threshold",
        "token_jaccard_merged",
    ]

    cases_df[cols].to_csv(
        out_dir / "failure_cases_for_manual_annotation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    reason_summary = (
        df[df["failure_type_auto"].str.startswith("false")]
        .groupby(["dataset", "failure_type_auto", "candidate_failure_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "failure_type_auto", "count"], ascending=[True, True, False])
    )
    reason_summary.to_csv(out_dir / "auto_failure_reason_summary.csv", index=False, encoding="utf-8-sig")

    # Plot 1: failure counts by dataset
    plot_df = summary_df.set_index("dataset")[
        ["false_positive_high_similarity_wrong", "false_negative_low_similarity_correct"]
    ]

    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Failure Cases by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of cases")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_counts_by_dataset.png", dpi=300)
    plt.close()

    # Plot 2: similarity vs HHEM score
    for dataset, g in df.groupby("dataset"):
        plt.figure(figsize=(6, 5))
        plt.scatter(g["cosine_similarity"], g["hhem_score"], s=10, alpha=0.5)
        plt.axvline(thresholds[dataset], linestyle="--", linewidth=1)
        plt.axhline(0.5, linestyle="--", linewidth=1)
        plt.title(f"Cosine Similarity vs HHEM Score: {dataset}")
        plt.xlabel("Cosine similarity")
        plt.ylabel("HHEM score")
        plt.tight_layout()
        plt.savefig(out_dir / f"similarity_vs_hhem_{dataset}.png", dpi=300)
        plt.close()

    print(f"Done. Outputs saved to: {out_dir}")
    print("\nMain files:")
    print(f"- {out_dir / 'part4_confusion_summary.csv'}")
    print(f"- {out_dir / 'failure_cases_for_manual_annotation.csv'}")
    print(f"- {out_dir / 'auto_failure_reason_summary.csv'}")
    print(f"- {out_dir / 'failure_counts_by_dataset.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="results/part4_failure_analysis")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    analyze(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out_dir),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()