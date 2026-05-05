# Project title: Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation

## Project Description
Large Language Models (LLMs) have demonstrated strong performance across a wide range of question-answering (QA) tasks. However, assessing the correctness of their generated responses remains a challenging problem, especially in the presence of hallucinations or semantically diverse valid answers.

This project investigates whether semantic similarity in embedding space can serve as a reliable indicator of prediction correctness. Specifically, students will explore whether the similarity between model-generated answers and ground-truth answers, measured using pretrained embedding models, correlates with factual correctness across different QA settings.

The study will involve both short-form QA tasks (with concise answers) and long-form QA tasks (with more complex and diverse responses), enabling a comprehensive analysis of when embedding-based similarity succeeds or fails as a proxy for correctness.

## Objectives
- Evaluate whether embedding similarity can distinguish correct and incorrect model predictions
- Compare performance across different QA task types (short-form vs. long-form)
- Analyze the limitations of embedding-based similarity metrics
- Propose potential improvements for more robust semantic evaluation

## Datasets
- **Short-form QA datasets**
  - SciQ
  - Simple Questions (Wiki-based)
- **Long-form QA datasets**
  - Natural Questions (NQ)
  - TruthfulQA

## Tasks

### Part 1: Prediction and Representation (25%)
- Generate model predictions for selected datasets
- Transform predictions and ground-truth answers into comparable textual forms
- Encode both predictions and references into embedding vectors

### Part 2: Similarity Analysis (25%)
- Compute similarity scores (e.g., cosine similarity) between prediction and ground truth embeddings
- Analyze how similarity scores differ between correct and incorrect predictions
- Investigate whether similarity can serve as a threshold-based correctness indicator

### Part 3: Empirical Study (25%)
- Compare results across Short-form vs. long-form QA tasks
- Visualize results using appropriate plots (e.g., distributions, ROC curves, correlation plots)

### Part 4: Failure Analysis and Improvement (25%)
- Identify scenarios where embedding similarity fails
- Analyze potential causes, such as:
  - Semantic ambiguity
  - Lexical variation
  - Long-form reasoning complexity
- Propose a well-motivated solution to improve robustness
  - This may involve alternative embedding strategies, structured representations, or advanced modeling techniques

## Related work for potential solutions
- [1] He, Neil, et al. Helm: Hyperbolic Large Language Models via Mixture-of-Curvature Experts. arXiv:2505.24722 (2025).
- [2] Patil, Sarang, et al. Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings. arXiv:2505.18973 (2025).

## Final Report
A well-structured report including:
- Introduction
- Methodology
- Experiments
- Results and analysis

**Expected GPU hours:** <50 hours