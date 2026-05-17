## Introduction

The project is designed to investigate whether the **embedding similarity between predictions and correct answers** can properly indicate **prediction correctness**.

## Preparation

### Download the Following Language Models

Please download the following models to your local device:

1. **Llama-3.2-3B-Instruct**  
https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct/files  

This model is used as the **base model for inference**.

2. **Qwen2.5-7B-Instruct**  
https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/files  

This model merges the **response from the base model and its corresponding query** into a **statement for evaluation**.

3. **HHEM-2.1-Open**  
https://huggingface.co/vectara/hallucination_evaluation_model  

This is a **hallucination evaluation model**. It produces scores between **0 and 1** to indicate whether statements are hallucinated given the ground truth.

4. **all-mpnet-base-v2**  
https://huggingface.co/sentence-transformers/all-mpnet-base-v2  

This is an **embedding model** that encodes text into vector representations.

After downloading the models, please **replace the model locations in the scripts** with your local paths.


### Create an Environment

The project is mainly based on **LlamaFactory**:  
https://github.com/hiyouga/LlamaFactory

Please create a **virtual environment** containing the required packages. Also check that your environment includes other necessary components, such as:

- seaborn (for visualization)
- torch
- sentence-transformers
- other dependencies required by LlamaFactory

# Part 1：Prediction and Representation

## Run the Scripts

### Run Inference

```
python llama_Inference.py --dataset nq --max_samples 2000
python llama_Inference.py --dataset sciq --max_samples 2000
python llama_Inference.py --dataset simple_questions_wiki --max_samples 2000
python llama_Inference.py --dataset truthfulQA
```

Run `llama_inference.py` to conduct inference on the following four datasets:

- nq
- sciq
- simple_questions_wiki
- truthfulQA

The data is stored in the `processed_data` folder.

You may decide how many samples to process for each dataset, as long as the number is **sufficient to support your conclusions**.


### Evaluate Predictions

```
conda activate hhem_eval
python eval_hem.py --dataset nq 
python eval_hem.py --dataset sciq 
python eval_hem.py --dataset simple_questions_wiki
python eval_hem.py --dataset truthfulQA
```

Run `eval_hem.py` to determine whether the predictions are **correct or hallucinated**.

### Compute Embeddings

```
python encoder_embedding.py --dataset sciq 
python encoder_embedding.py --dataset simple_questions_wiki 
python encoder_embedding.py --dataset nq 
python encoder_embedding.py --dataset truthfulQA 
```

Run `encoder_embedding.py` to compute embeddings for:

- predicted statements
- ground-truth statements

# Part 2：Similarity Analysis

Part 2 analyzes whether embedding similarity between model predictions and ground-truth answers can indicate prediction correctness.

This step uses the outputs generated in Part 1:

```
results/{dataset}/prediction.pkl
results/{dataset}/correctness.json
results/{dataset}/embeddings.pkl
```

The analysis script is:

```
part2_similarity_analysis.py
```

## What This Script Does

For each dataset, the script:

1. Loads model predictions and rewritten factual statements from `prediction.pkl`.
2. Loads HHEM consistency scores from `correctness.json`.
3. Loads prediction/reference embeddings from `embeddings.pkl`.
4. Computes cosine similarity between each prediction embedding and its corresponding ground-truth embedding.
5. Uses the HHEM score to define correctness labels:

```
HHEM score >= label_threshold  -> correct
HHEM score <  label_threshold  -> incorrect
```
By default, `label_threshold = 0.5`.

6. Compares cosine similarity distributions between correct and incorrect predictions.
7. Computes Pearson and Spearman correlation between HHEM scores and cosine similarity.
8. Evaluates whether cosine similarity can serve as a threshold-based correctness indicator using:
- ROC-AUC
- Average Precision
- best-F1 similarity threshold
- optional fixed similarity threshold metrics

## Main Command

Run the full Part 2 analysis on all four datasets:

```
python part2_similarity_analysis.py --label_threshold 0.5
```

## Recommended Command With Fixed Similarity Threshold
To additionally evaluate a fixed cosine-similarity threshold, run:
```
python part2_similarity_analysis.py --label_threshold 0.5 --similarity_threshold 0.75
```
Here, predictions with cosine similarity greater than or equal to `0.75` are classified as correct by the similarity-based rule.

## Output Files
For each dataset, the script writes:
```
results/{dataset}/similarity_scores.csv
results/{dataset}/similarity_summary.json
```

The file `similarity_scores.csv` contains per-sample results, including:

```
dataset
index
question
prediction
true_answer
merged_prediction
merged_true_answer
hhem_score
is_correct
cosine_similarity
```

The file `similarity_summary.json` contains dataset-level statistics, including:

```
num_samples
num_correct
num_incorrect
similarity_all
similarity_correct
similarity_incorrect
mean_gap_correct_minus_incorrect
pearson_hhem_vs_similarity
spearman_hhem_vs_similarity
roc_auc_similarity_predicts_correct
average_precision_similarity_predicts_correct
best_f1_similarity_threshold
fixed_similarity_threshold
```

The script also writes cross-dataset summary files:
```
results/part2_similarity_summary.csv
results/part2_similarity_summary.json
```

# Part 3: Empirical Study and Visualization

Part 3 visualizes the Part 2 similarity-analysis results and compares the behavior of embedding-based evaluation across short-form and long-form QA tasks.

This step uses the outputs generated by `part2_similarity_analysis.py`:

```
results/{dataset}/similarity_scores.csv
results/{dataset}/similarity_summary.json
```

The visualization script is:

```
part2_visualization.py
```

## Main Command

Run the visualization script after completing Part 2:

```
python part2_visualization.py
```

The script loads all available datasets from `results/`, generates four plots, and prints a short-form vs. long-form comparison report in the terminal.

## Output Figures

The generated figures are saved to the `results/` directory:

```
results/plot_1_distribution.png
results/plot_2_roc.png
results/plot_3_correlation.png
results/plot_4_metrics.png
```

## Figure Descriptions

### Plot 1: Cosine Similarity Distribution

```
results/plot_1_distribution.png
```

This plot compares cosine-similarity distributions across task type and correctness label:

- Short-form correct predictions
- Short-form incorrect predictions
- Long-form correct predictions
- Long-form incorrect predictions

This plot helps show whether correct predictions tend to have higher embedding similarity than incorrect predictions.

### Plot 2: ROC Curves

```
results/plot_2_roc.png
```

This plot evaluates whether cosine similarity can rank correct predictions above incorrect predictions. 

### Plot 3: HHEM Score vs Cosine Similarity

```
results/plot_3_correlation.png
```

This scatter plot shows the relationship between the HHEM factuality score and cosine similarity. Regression lines are drawn separately for short-form and long-form QA.

This plot helps reveal whether embedding similarity increases consistently as HHEM factuality scores increase.

### Plot 4: Metric Comparison

```
results/plot_4_metrics.png
```

This bar chart compares the main threshold-analysis metrics across datasets:

- ROC-AUC
- Average Precision
- Best F1

It provides a compact view of whether similarity-based correctness prediction works better for short-form QA or long-form QA.

Second, it evaluates whether cosine similarity can be used as a threshold-based correctness indicator. ROC-AUC and Average Precision measure ranking quality, while the best-F1 threshold and optional fixed threshold evaluate direct binary classification performance.

The best-F1 threshold is selected on the same dataset being evaluated, so it should be interpreted as an in-sample exploratory threshold rather than a guaranteed general threshold.

# Part 4: Failure Analysis & Structured Evaluation

This part focuses on analyzing the limitations of embedding-based similarity for LLM prediction evaluation and implementing a structured improvement framework.

It consists of two main components:

- Failure case analysis (diagnosing when cosine similarity fails)
- Adaptive Structured Similarity Evaluator (a lightweight improvement over cosine thresholding)

---

## 1. Failure Analysis

File: `part4_failure_analysis.py` 

### Overview

This script analyzes when cosine similarity fails as a proxy for correctness.

It:
- Loads similarity scores and HHEM labels
- Applies dataset-specific best-F1 thresholds
- Identifies failure cases:
  - False positives (high similarity but incorrect)
  - False negatives (low similarity but correct)
- Automatically classifies failure types

---

### Key Features

#### 1. Failure Type Detection

The script identifies common failure patterns:

- `numeric_or_date_mismatch`
- `negation_or_contradiction`
- `entity_or_relation_mismatch`
- `semantic_ambiguity`
- `lexical_variation_or_paraphrase`
- `overly_strict_threshold_or_hhem_artifact`

These are inferred using heuristics such as:

- number extraction
- negation detection
- token-level Jaccard similarity

---

#### 2. Dataset-Level Thresholding

Thresholds are loaded from:
results/part2_similarity_summary.json 
or dataset-level summaries.

Each sample is classified using:
cosine_similarity >= dataset_threshold


---

#### 3. Outputs

The script generates:

- `all_scores_with_failure_flags.csv`
- `part4_confusion_summary.csv`
- `failure_cases_for_manual_annotation.csv`
- `auto_failure_reason_summary.csv`
- `failure_counts_by_dataset.png`

These outputs support both quantitative analysis and manual inspection.

---

## 2. Adaptive Structured Similarity Evaluator

File: `part4_improved_evaluator.py` :contentReference[oaicite:1]{index=1}

### Overview

This module implements a lightweight improvement over the cosine-threshold evaluator.

Instead of relying only on embedding similarity, it introduces:

- structured factual checks
- recovery rules for likely false negatives
- adaptive control based on dataset characteristics

---

### 2.1 Baseline

The original evaluator uses:
y = 1 if cosine_similarity >= threshold else 0


---

### 2.2 Structured Rules

#### (A) Blocking Modules (reduce false positives)

Reject predictions with critical factual mismatches:

- number/date mismatch
- negation mismatch
- contrastive category mismatch (e.g., male vs female)
- comparison direction mismatch (e.g., A > B vs B > A)

---

#### (B) Recovery Modules (reduce false negatives)

Recover likely correct predictions using:

- exact or containment match
- high token-level Jaccard similarity
- refusal-style answer matching

---

### 2.3 Balanced Evaluator

Applies all structured rules uniformly:
prediction = baseline OR recovery
prediction = prediction AND NOT critical_mismatch


---

### 2.4 Adaptive Evaluator

Adds dataset-level control:
if dataset_threshold >= 0.95:
use baseline
else:
use structured evaluator


This prevents over-correction on datasets like TruthfulQA, where thresholds are already very strict.

---

### 2.5 Outputs

The script generates:

- `improvement_metrics.csv`
- `all_scores_with_improvement_predictions.csv`
- `structured_feature_summary.csv`
- `improvement_config.json`

---

## 3. Key Insights

From the experiments:

- Cosine similarity is effective but incomplete
- Major failure modes include:
  - numeric mismatch
  - entity mismatch
  - comparison-direction errors
  - long-form partial answers
- Structured rules significantly improve performance
- Adaptive control is necessary for noisy or ambiguous datasets

---

## 4. Usage

### Step 1: Run failure analysis

```bash
python part4_failure_analysis.py \
    --results_dir results \
    --out_dir results/part4_failure_analysis
```

### Step 2: Run structured evaluator
```python part4_improved_evaluator.py \
    --input results/part4_failure_analysis/all_scores_with_failure_flags.csv \
    --out_dir results/part4_improvement
```

## 5. Notes
- Evaluation uses HHEM-derived labels (not human annotations)
- The structured evaluator is a lightweight proof-of-concept
- No additional model training is required
- Designed for interpretability and reproducibility

## 6. Hyperbolic Mixture-of-Curvature Embeddings
File: hyperbolic_metrics.py, part2_similarity_analysis_hyperbolic.py

### Overview
This part shows a new method embedding vectors into hyperbolic space and a new semantic similarity evaluation method.
To evaluate if the long-form dataset truly has a high confidence prediction, we also  introduce the difference of Pearson r as a metric.
It introduces:
- adaptive hyperbolic space depending on length of two answers
- Poincaré distance dD(u, v)
- new semantic similarity method S_hyp

### 6.1 Pipeline

#### Step1:
Avoid boundary singu-larities
new vector = vector * scale=0.95
#### Step2：
- Input Preprocessing: Ensure inputs are L2-normalized with norms < 1.0. If coming from a neural network, apply:
x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8) * 0.95.
- Training vs. Inference: This is a NumPy evaluation/inference implementation. For training loops, replace np.* with torch.* or tf.* to maintain gradient flow.
- Threshold Tuning: mask_long = seq_lengths > 80 is based on typical English/Chinese token distributions. 
- Decay Coefficient: The 1.5 in exp(-1.5 * dist) acts as a temperature parameter.
  - Increase to 2.0~3.0 for sharper discrimination.
  - Decrease to 0.8~1.2 for smoother similarity transitions.
### Step3:
Auto select curvature for different lengths of answers. Adopt heuristic thresholds based on token/word counts, or fix moderate-to-low curvature values to avoid distortion.
c = 0.1 for short and 0.3 for long.
### 6.2 Similarity analysis for hyperbolic (part2_similarity_analysis_hyperbolic.py)
Same as part2_similarity_analysis but import hyperbolic for similarity calculation.

### 6.3 Outcome

The script generates:
- part2_similarity_summary.csv
- part2_similarity_summary.json
