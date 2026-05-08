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

## Interpretation

The Part 2 analysis addresses two main questions.

First, it compares similarity scores between correct and incorrect predictions. If correct predictions have clearly higher cosine similarity than incorrect predictions, embedding similarity is useful for distinguishing correctness.

Second, it evaluates whether cosine similarity can be used as a threshold-based correctness indicator. ROC-AUC and Average Precision measure ranking quality, while the best-F1 threshold and optional fixed threshold evaluate direct binary classification performance.

The best-F1 threshold is selected on the same dataset being evaluated, so it should be interpreted as an in-sample exploratory threshold rather than a guaranteed general threshold.
