# cafa5-protein-function-prediction
End-to-end protein function prediction on CAFA5 benchmark using ESM-2 embeddings and multi-label classification
Overview
Predicts Gene Ontology (GO) terms for proteins using sequence-derived features and machine learning. Compared multiple feature-model pipelines on 142,246 real proteins across 1,000 GO term labels.
Problem

Input: Protein amino acid sequence
Output: Multiple GO term labels (biological function annotations)
Metric: Fmax (protein-centric F1 score)
Challenge: 97% label sparsity, 1000 output classes, class imbalance

Results
Model                      Features                   Fmax
Logistic Regression        K-mer (420 dims)         0.337
Logistic Regression        ESM-2 (1280 dims)        0.321
MLP                        ESM-2 (1280 dims)        0.298

Key Finding
Simple logistic regression on 420-dimensional k-mer features outperformed deep MLP on 1280-dimensional ESM-2 transformer embeddings. Analysis revealed the bottleneck is label sparsity on specific GO terms (child terms Fmax 0.15) rather than feature quality.
Pipeline
Protein sequences (FASTA)
        ↓
Feature Engineering
├── K-mer frequency vectors (420 dims)
└── ESM-2 embeddings (1280 dims, GPU)
        ↓
Multi-label Classification
├── Logistic Regression
└── MLP (PyTorch, GPU)
        ↓
Evaluation (Fmax, threshold sweep)
        ↓
GO Hierarchy Analysis (parent vs child terms)



Engineering Highlights
Generated ESM-2 embeddings for 142k proteins using batched GPU inference
Reduced embedding generation from 21 hours to 6 hours via sorted batching
Resumable checkpoint pipeline — safe across Kaggle session resets
Masked mean pooling for variable-length sequence embeddings

Dataset
CAFA5 Protein Function Prediction — Kaggle competition dataset
Notebooks

esm_embedding_generation.ipynb — ESM-2 embedding pipeline
protein_function_prediction.ipynb — Training, evaluation, analysis


## Tech Stack

**ML & Deep Learning:** PyTorch, scikit-learn, LightGBM

**Protein Language Model:** ESM-2 (650M parameters, Facebook/Meta)

**Data Processing:** NumPy, Pandas, BioPython

**GPU Infrastructure:** Kaggle T4 x2, CUDA

**Evaluation:** Custom Fmax implementation (CAFA5 metric)
