import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 🎨 绘图样式配置
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 13,
    'figure.dpi': 300
})

# 🗂️ 数据集分类映射（根据你的推理脚本逻辑）
TASK_MAPPING = {
    "sciq": "Short-form",
    "simple_questions_wiki": "Short-form",
    "nq": "Long-form",
    "truthfulQA": "Long-form"
}
RESULTS_DIR = Path("results")

def load_data():
    """带诊断功能的健壮数据加载器"""
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
            
    print("\n📂 目录诊断报告:")
    print(f"  ✅ 已找到数据集 ({len(found_datasets)}/4): {found_datasets if found_datasets else '无'}")
    if missing_datasets:
        print(f"  ❌ 缺失数据集 ({len(missing_datasets)}/4): {missing_datasets}")
        print("  💡 修复建议: 运行分析脚本并指定正确路径:")
        print(f"     python part2_analysis.py --datasets {' '.join(missing_datasets)} --results_dir {RESULTS_DIR}")
        
    if not dfs:
        raise FileNotFoundError("未找到任何分析结果文件，请先完整运行评估分析脚本。")
        
    print(f"  📊 成功加载 {len(dfs)} 个数据集的明细数据。\n")
    return pd.concat(dfs, ignore_index=True), summaries


def plot_similarity_distribution(df):
    """图1：相似度分布对比（按任务类型与正误标签）"""
    plt.figure(figsize=(9, 6))
    for task in ["Short-form", "Long-form"]:
        sub = df[df["task_type"] == task]
        for label, color in zip([True, False], ["#2ca02c", "#d62728"]):
            sns.kdeplot(
                data=sub[sub["is_correct"] == label],
                x="cosine_similarity",
                label=f"{task} - {'Correct' if label else 'Incorrect'}",
                color=color,
                fill=True, alpha=0.4, linewidth=2
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
    """图2：ROC 曲线对比（评估相似度预测正确性的能力）"""
    plt.figure(figsize=(8, 6))
    for ds, task_type in TASK_MAPPING.items():
        sub = df[df["dataset"] == ds]
        if sub.empty: continue
        fpr, tpr, _ = roc_curve(sub["is_correct"].astype(int), sub["cosine_similarity"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2.5, 
                 label=f"{ds} ({task_type}) | AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Can Similarity Predict Answer Correctness?")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_2_roc.png", dpi=300)
    plt.show()

def plot_correlation(df):
    """图3：HHEM分数 vs 余弦相似度（散点+回归线）"""
    # 使用 lmplot 替代 regplot，以支持 hue 参数
    g = sns.lmplot(
        data=df, x="hhem_score", y="cosine_similarity",
        hue="task_type",
        scatter_kws={"alpha": 0.4, "s": 40, "edgecolor": "w"},
        line_kws={"linewidth": 2.5},
        ci=95,
        palette="Set1",
        height=6,
        aspect=1.4
    )
    # 设置坐标轴标签与标题
    g.set_axis_labels("HHEM Score (Ground Truth Correctness)", "Predicted Cosine Similarity")
    plt.suptitle("Correlation: HHEM Factuality Score vs Cosine Similarity", y=1.01, fontsize=13)
    plt.tight_layout()
    
    # 保存图片（lmplot 返回的是 FacetGrid 对象）
    plt.savefig(RESULTS_DIR / "plot_3_correlation.png", dpi=300)
    plt.show()


def plot_metric_comparison(summaries):
    """图4：核心指标柱状图对比（AUC, AP, 最佳F1阈值）"""
    records = []
    for s in summaries:
        best_f1 = s["best_f1_similarity_threshold"]
        records.append({
            "dataset": s["dataset"],
            "task_type": s["task_type"],
            "roc_auc": s.get("roc_auc_similarity_predicts_correct", 0),
            "avg_precision": s.get("average_precision_similarity_predicts_correct", 0),
            "best_f1": best_f1["f1"] if best_f1 else 0,
            "best_threshold": best_f1["threshold"] if best_f1 else 0
        })
    df_m = pd.DataFrame(records)
    
    plt.figure(figsize=(8, 5))
    x = np.arange(len(df_m))
    w = 0.25
    plt.bar(x - w, df_m["roc_auc"], w, label="ROC-AUC", color="#1f77b4", alpha=0.85)
    plt.bar(x, df_m["avg_precision"], w, label="Avg Precision", color="#2ca02c", alpha=0.85)
    plt.bar(x + w, df_m["best_f1"], w, label="Best F1", color="#ff7f0e", alpha=0.85)
    
    plt.xticks(x, [f"{d}\n({t})" for d, t in zip(df_m["dataset"], df_m["task_type"])])
    plt.ylabel("Score"); plt.ylim(0, 1)
    plt.title("Predictive Performance Metrics by Task Type")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_4_metrics.png", dpi=300)
    plt.show()
    return df_m

def print_comparison_report(df_m, df):
    """终端输出：短问答 vs 长问答的量化对比"""
    print("\n" + "="*60)
    print("📊 SHORT-FORM vs LONG-FORM QA COMPARISON REPORT")
    print("="*60)
    for metric in ["roc_auc", "avg_precision", "best_f1"]:
        short_avg = df_m[df_m["task_type"]=="Short-form"][metric].mean()
        long_avg = df_m[df_m["task_type"]=="Long-form"][metric].mean()
        gap = short_avg - long_avg
        print(f"🔹 {metric.upper():<15} | Short: {short_avg:.3f} | Long: {long_avg:.3f} | Δ: {gap:+.3f}")
        
    # 相似度分布统计
    print("-" * 60)
    for task in ["Short-form", "Long-form"]:
        sub = df[df["task_type"] == task]
        mean_sim = sub["cosine_similarity"].mean()
        corr = np.corrcoef(sub["hhem_score"], sub["cosine_similarity"])[0, 1]
        print(f"🔹 {task:<10} | Mean Similarity: {mean_sim:.3f} | Pearson Corr: {corr:.3f}")
    print("="*60 + "\n")

def main():
    print("🔍 Loading data from results/ ...")
    df, summaries = load_data()
    
    print("📈 Generating plots...")
    plot_similarity_distribution(df)
    plot_roc_curves(df)
    plot_correlation(df)
    df_m = plot_metric_comparison(summaries)
    
    print_comparison_report(df_m, df)
    print("✅ All figures saved to `results/` directory.")

if __name__ == "__main__":
    main()
