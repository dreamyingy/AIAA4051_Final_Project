"""
Part 4 improvement experiment: Question-Conditioned Structured Similarity Filter.

Input:
  all_scores_with_failure_flags.csv generated from Part 4 failure-analysis script.

This script evaluates a lightweight implementation of the proposed improvement:
  1) keep the original cosine-threshold decision as the baseline;
  2) recover likely false negatives using exact/containment answer matching,
     high token overlap, and refusal-style answer matching;
  3) block likely false positives using structured checks for critical mismatches:
     numbers/dates, negation, contrastive category pairs, and comparison direction;
  4) use an adaptive mode: if the cosine threshold is extremely high (>= 0.95),
     keep the original baseline decision, because aggressive recovery tends to hurt
     datasets such as TruthfulQA where automatic labels are noisy and the tuned
     threshold is already very conservative.

The experiment still uses HHEM-derived labels as the evaluation target, so the
results should be interpreted as a proof-of-concept improvement over pure cosine
thresholding rather than a human-gold factuality benchmark.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONTRAST_PAIRS = [
    ("male", "female"),
    ("yes", "no"),
    ("true", "false"),
    ("mitosis", "meiosis"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("larger", "smaller"),
    ("left", "right"),
    ("north", "south"),
    ("east", "west"),
    ("positive", "negative"),
    ("above", "below"),
]

COMPARATORS = [
    "denser than",
    "larger than",
    "smaller than",
    "greater than",
    "less than",
    "higher than",
    "lower than",
    "older than",
    "younger than",
]

NEGATION_WORDS = {
    "no", "not", "never", "none", "neither", "nor", "cannot", "can't",
    "without", "unable", "unknown", "impossible",
}

REFUSAL_PATTERNS = [
    "unable", "cannot", "can not", "no comment", "not know", "unknown",
    "not available", "do not have", "don t have", "i have no",
]


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return normalize_text(text).split()


def extract_numbers(text: str):
    return re.findall(r"\b\d+(?:\.\d+)?\b", str(text))


def number_or_date_mismatch(pred: str, ref: str) -> bool:
    pred_nums = extract_numbers(pred)
    ref_nums = extract_numbers(ref)
    return bool(pred_nums or ref_nums) and pred_nums != ref_nums


def has_negation(text: str) -> bool:
    return any(tok in NEGATION_WORDS for tok in tokenize(text))


def negation_mismatch(pred: str, ref: str) -> bool:
    return has_negation(pred) != has_negation(ref)


def contrast_pair_mismatch(pred: str, ref: str) -> bool:
    pred_tokens = set(tokenize(pred))
    ref_tokens = set(tokenize(ref))
    for a, b in CONTRAST_PAIRS:
        if (a in pred_tokens and b in ref_tokens) or (b in pred_tokens and a in ref_tokens):
            return True
    return False


def comparison_direction_mismatch(pred: str, ref: str) -> bool:
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    for comp in COMPARATORS:
        if comp not in pred_norm or comp not in ref_norm:
            continue
        pred_parts = pred_norm.split(comp)
        ref_parts = ref_norm.split(comp)
        if len(pred_parts) < 2 or len(ref_parts) < 2:
            continue
        pred_left = set(pred_parts[0].split()[-4:])
        pred_right = set(pred_parts[1].split()[:4])
        ref_left = set(ref_parts[0].split()[-4:])
        ref_right = set(ref_parts[1].split()[:4])
        if pred_left & ref_right and pred_right & ref_left:
            return True
    return False


def exact_or_containment_match(pred: str, ref: str) -> bool:
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    if not pred_norm or not ref_norm:
        return False
    if pred_norm == ref_norm:
        return True
    if len(pred_norm) >= 3 and pred_norm in ref_norm:
        return True
    if len(ref_norm) >= 3 and ref_norm in pred_norm:
        return True
    return False


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def is_refusal_like(text: str) -> bool:
    text = normalize_text(text)
    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["baseline_pred"] = df["similarity_predict_correct"].astype(str).str.lower().isin(["true", "1", "yes"])

    df["number_or_date_mismatch"] = df.apply(
        lambda r: number_or_date_mismatch(r["merged_prediction"], r["merged_true_answer"]), axis=1
    )
    df["negation_mismatch"] = df.apply(
        lambda r: negation_mismatch(r["merged_prediction"], r["merged_true_answer"]), axis=1
    )
    df["contrast_pair_mismatch"] = df.apply(
        lambda r: contrast_pair_mismatch(r["merged_prediction"], r["merged_true_answer"]), axis=1
    )
    df["comparison_direction_mismatch"] = df.apply(
        lambda r: comparison_direction_mismatch(r["merged_prediction"], r["merged_true_answer"]), axis=1
    )
    df["critical_mismatch"] = (
        df["number_or_date_mismatch"]
        | df["negation_mismatch"]
        | df["contrast_pair_mismatch"]
        | df["comparison_direction_mismatch"]
    )

    df["exact_or_containment_match"] = df.apply(
        lambda r: exact_or_containment_match(r["prediction"], r["true_answer"])
        or exact_or_containment_match(r["merged_prediction"], r["merged_true_answer"]),
        axis=1,
    )
    df["answer_token_jaccard"] = df.apply(
        lambda r: token_jaccard(r["prediction"], r["true_answer"]), axis=1
    )
    df["refusal_style_match"] = df.apply(
        lambda r: (is_refusal_like(r["prediction"]) and is_refusal_like(r["true_answer"]))
        or (is_refusal_like(r["merged_prediction"]) and is_refusal_like(r["merged_true_answer"])),
        axis=1,
    )
    return df


def predict_balanced(df: pd.DataFrame, jaccard_threshold: float = 0.80) -> pd.Series:
    """Balanced structured rule: recover likely FNs and block critical FPs."""
    recover = (
        df["exact_or_containment_match"]
        | (df["answer_token_jaccard"] >= jaccard_threshold)
        | df["refusal_style_match"]
    )
    pred = df["baseline_pred"] | recover
    pred = pred & ~(df["critical_mismatch"] & ~df["exact_or_containment_match"])
    return pred


def predict_adaptive(df: pd.DataFrame, jaccard_threshold: float = 0.80, high_threshold_cutoff: float = 0.95) -> pd.Series:
    """
    Adaptive structured rule.
    If a dataset has an extremely high tuned similarity threshold, keep the baseline decision.
    This avoids over-correcting datasets where the tuned threshold is already conservative.
    """
    balanced = predict_balanced(df, jaccard_threshold=jaccard_threshold)
    return pd.Series(
        np.where(df["dataset_threshold"] >= high_threshold_cutoff, df["baseline_pred"], balanced),
        index=df.index,
    ).astype(bool)


def evaluate(df: pd.DataFrame, pred_col: str, method_name: str):
    rows = []
    overall = compute_metrics(df["is_correct"], df[pred_col])
    overall.update({"dataset": "overall", "method": method_name, "num_samples": len(df)})
    rows.append(overall)
    for dataset, g in df.groupby("dataset"):
        item = compute_metrics(g["is_correct"], g[pred_col])
        item.update({"dataset": dataset, "method": method_name, "num_samples": len(g)})
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/part4_failure_analysis/all_scores_with_failure_flags.csv")
    parser.add_argument("--out_dir", type=str, default="results/part4_improvement")
    parser.add_argument("--jaccard_threshold", type=float, default=0.80)
    parser.add_argument("--high_threshold_cutoff", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = add_features(df)
    df["balanced_structured_pred"] = predict_balanced(df, args.jaccard_threshold)
    df["adaptive_structured_pred"] = predict_adaptive(
        df,
        jaccard_threshold=args.jaccard_threshold,
        high_threshold_cutoff=args.high_threshold_cutoff,
    )

    rows = []
    rows.extend(evaluate(df, "baseline_pred", "cosine_threshold_baseline"))
    rows.extend(evaluate(df, "balanced_structured_pred", "balanced_structured_filter"))
    rows.extend(evaluate(df, "adaptive_structured_pred", "adaptive_structured_filter"))
    metrics = pd.DataFrame(rows)

    # Put columns in a report-friendly order.
    metric_cols = [
        "method", "dataset", "num_samples", "tp", "tn", "fp", "fn",
        "precision", "recall", "f1", "accuracy",
    ]
    metrics = metrics[metric_cols]
    metrics.to_csv(out_dir / "improvement_metrics.csv", index=False)
    df.to_csv(out_dir / "all_scores_with_improvement_predictions.csv", index=False)

    feature_summary = (
        df.groupby("dataset")[[
            "number_or_date_mismatch",
            "negation_mismatch",
            "contrast_pair_mismatch",
            "comparison_direction_mismatch",
            "critical_mismatch",
            "exact_or_containment_match",
            "refusal_style_match",
        ]]
        .sum()
        .reset_index()
    )
    feature_summary.to_csv(out_dir / "structured_feature_summary.csv", index=False)

    with open(out_dir / "improvement_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "jaccard_threshold": args.jaccard_threshold,
                "high_threshold_cutoff": args.high_threshold_cutoff,
                "evaluation_target": "HHEM-derived correctness labels",
                "note": "The adaptive rule keeps the cosine baseline when dataset_threshold >= high_threshold_cutoff.",
            },
            f,
            indent=2,
        )

    print("Saved outputs to", out_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
