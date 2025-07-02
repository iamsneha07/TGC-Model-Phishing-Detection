# 📈 Transaction Graph Contrastive (TGC) Model for Phishing Detection

This project implements a graph-based phishing detection pipeline using a Transaction Graph Contrastive Learning approach. It leverages Deep Graph Library (DGL) and PyTorch to learn node embeddings with a Graph Attention Network (GAT) encoder and contrastive loss. The final embeddings are used with an XGBoost classifier to detect phishing nodes with improved precision and recall.

## ➡️ Key features:

🔹Builds a directed transaction graph from raw CSV data

🔹Generates node-level features and ego-graphs

🔹Trains a GAT-based encoder using InfoNCE contrastive loss

🔹Extracts robust embeddings for nodes

🔹Classifies nodes as phishing/non-phishing using a boosted tree model

🔹Rich, color-coded console output with detailed metrics and confusion matrix

## ⚙️ Tech stack:
PyTorch · DGL · NetworkX · XGBoost · Scikit-learn · Rich · TQDM

## 🛡️ Purpose: 
Helps improve phishing detection accuracy in transaction networks by leveraging self-supervised graph learning.

## ✅ Status: 
Working prototype with tested Colab integration.

## 📂 Usage: 
Clone, install requirements, run the training pipeline, and evaluate phishing detection performance.
