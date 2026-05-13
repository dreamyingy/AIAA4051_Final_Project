import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


DATASETS = ["sciq", "simple_questions_wiki", "nq", "truthfulQA"]


def load_pickle_file(path):
    objs = []
    with open(path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break
    return objs


def load_predictions(path):
    chunks = load_pickle_file(path)
    rows = [x for chunk in chunks for x in chunk]
    out = []
    for item in rows:
        out.append(item[0] if isinstance(item, list) else item)
    return out


def to_numpy(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def load_embeddings(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    pred = np.stack([to_numpy(x) for x in obj["pred_embeddings"]])
    true = np.stack([to_numpy(x) for x in obj["true_embeddings"]])
    return normalize(pred), normalize(true)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def corr(a, b, method):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if method == "spearman":
        a = rankdata(a)
        b = rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(labels)) + 1)
    return float(precision[sorted_labels].sum() / n_pos)


def threshold_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_threshold(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None

    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        metrics = threshold_metrics(labels, scores, threshold)
        key = (metrics["f1"], metrics["accuracy"])
        if best is None or key > (best["f1"], best["accuracy"]):
            best = metrics
    return best


def analyze_dataset(dataset, results_dir, label_threshold, fixed_similarity_threshold):
    dataset_dir = results_dir / dataset
    predictions = load_predictions(dataset_dir / "prediction.pkl")
    pred_emb, true_emb = load_embeddings(dataset_dir / "embeddings.pkl")

    with open(dataset_dir / "correctness.json", "r", encoding="utf-8") as f:
        hhem_scores = np.asarray(json.load(f), dtype=np.float64)

    n = len(predictions)
    if not (len(pred_emb) == len(true_emb) == len(hhem_scores) == n):
        raise ValueError(f"{dataset}: prediction, embedding, and HHEM lengths do not match.")

    import sys; sys.path.append(".")
    from hyperbolic_metrics import mixture_of_curvature_similarity

    pred_lengths = np.array([len(str(item.get("prediction", ""))) for item in predictions])
    true_lengths = np.array([len(str(item.get("true_answer", ""))) for item in predictions])
    avg_lengths = (pred_lengths + true_lengths) / 2.0

    cosine = mixture_of_curvature_similarity(pred_emb, true_emb, seq_lengths=avg_lengths)

    labels = hhem_scores >= label_threshold

    rows = []
    for i, item in enumerate(predictions):
        rows.append(
            {
                "dataset": dataset,
                "index": i,
                "question": item.get("question", ""),
                "prediction": item.get("prediction", ""),
                "true_answer": item.get("true_answer", ""),
                "merged_prediction": item.get("merged_prediction", ""),
                "merged_true_answer": item.get("merged_true_answer", ""),
                "hhem_score": float(hhem_scores[i]),
                "is_correct": bool(labels[i]),
                "cosine_similarity": float(cosine[i]),
            }
        )

    write_csv(dataset_dir / "similarity_scores.csv", rows)

    summary = {
        "dataset": dataset,
        "num_samples": int(n),
        "label_threshold": float(label_threshold),
        "num_correct": int(labels.sum()),
        "num_incorrect": int((~labels).sum()),
        "similarity_all": describe(cosine),
        "similarity_correct": describe(cosine[labels]),
        "similarity_incorrect": describe(cosine[~labels]),
        "mean_gap_correct_minus_incorrect": (
            float(np.mean(cosine[labels]) - np.mean(cosine[~labels]))
            if labels.sum() and (~labels).sum()
            else None
        ),
        "pearson_hhem_vs_similarity": corr(hhem_scores, cosine, "pearson"),
        "spearman_hhem_vs_similarity": corr(hhem_scores, cosine, "spearman"),
        "roc_auc_similarity_predicts_correct": roc_auc(labels, cosine),
        "average_precision_similarity_predicts_correct": average_precision(labels, cosine),
        "best_f1_similarity_threshold": best_threshold(labels, cosine),
    }

    if fixed_similarity_threshold is not None:
        summary["fixed_similarity_threshold"] = threshold_metrics(labels, cosine, fixed_similarity_threshold)

    with open(dataset_dir / "similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def flatten_summary(summary):
    best = summary["best_f1_similarity_threshold"] or {}
    return {
        "dataset": summary["dataset"],
        "num_samples": summary["num_samples"],
        "num_correct": summary["num_correct"],
        "num_incorrect": summary["num_incorrect"],
        "correct_mean_similarity": summary["similarity_correct"]["mean"],
        "incorrect_mean_similarity": summary["similarity_incorrect"]["mean"],
        "mean_gap": summary["mean_gap_correct_minus_incorrect"],
        "pearson": summary["pearson_hhem_vs_similarity"],
        "spearman": summary["spearman_hhem_vs_similarity"],
        "roc_auc": summary["roc_auc_similarity_predicts_correct"],
        "average_precision": summary["average_precision_similarity_predicts_correct"],
        "best_threshold": best.get("threshold"),
        "best_accuracy": best.get("accuracy"),
        "best_precision": best.get("precision"),
        "best_recall": best.get("recall"),
        "best_f1": best.get("f1"),
    }


def main():
    parser = argparse.ArgumentParser(description="Part 2 similarity analysis.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--label_threshold", default=0.5, type=float)
    parser.add_argument("--similarity_threshold", default=None, type=float)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summaries = [
        analyze_dataset(dataset, results_dir, args.label_threshold, args.similarity_threshold)
        for dataset in args.datasets
    ]

    write_csv(results_dir / "part2_similarity_summary.csv", [flatten_summary(x) for x in summaries])

    with open(results_dir / "part2_similarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
