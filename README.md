# Airline Review Sentiment Analysis & Multi-Label Aspect Detection Using Fine-Tuned Transformer Models and SHAP

## 📌 1. Project Overview
This repository contains an advanced deep learning pipeline developed for a collaborative academic research project centered on aviation customer experience analytics. Utilizing raw customer reviews scraped from the **Skytrax** platform, this project fine-tunes state-of-the-art **Transformer** architectures to simultaneously solve two complex computational linguistics tasks:
1. **Multi-class Sentiment Classification:** Categorizing customer reviews into Positive, Negative, or Neutral classes based on rating-to-sentiment mapping.
2. **Multi-label Aspect Detection:** Identifying the simultaneous presence of 5 distinct core airline service dimensions (*Food & Beverage, Staff Service, Wifi Inflight, Ground Service, Seat & Cabin*) within a single review text.

This project follows strict scientific methodologies and is part of an ongoing **academic publication pipeline**.

## 🛠️ 2. Transformer Architectures & Tokenization Specifications
A rigorous benchmarking study was conducted across 3 major Transformer variants:
* **BERT (`bert-base-uncased`):** Implements WordPiece tokenization, yielding 55 tokens on the standard benchmark sample and breaking down out-of-vocabulary words using the `##` subword prefix.
* **RoBERTa (`roberta-base`):** Formulated on a robustly optimized pretraining approach with dynamic masking and Byte-Pair Encoding (BPE), yielding 59 tokens with a specialized `Ġ` space-prefix marker.
* **DeBERTa (`microsoft/deberta-v3-base`):** Features a SentencePiece tokenizer and introduces a *Disentangled Attention* mechanism that separates content and relative position encodings, yielding the most mathematically efficient representation at 52 tokens.

## 📊 3. Benchmarking & Comparative Performance Evaluation
Model evaluation was strictly conducted on a stratified test set of 600 unseen documents to ensure a highly consistent and fair label distribution between the training and testing phases.

### Performance Benchmark Table (Middle Eastern Airlines Dataset):
| Model Architecture | Accuracy | F1-Macro | AUC Score |
| :--- | :---: | :---: | :---: |
| **BERT (Base Uncased)** | 85.17% | 0.6760 | 0.9090 |
| **RoBERTa (Base)** | 84.17% | **0.7154** | 0.9190 |
| **DeBERTa (Base v3)** | **86.50%** | 0.7203 | **0.9238** |

### Evaluation Insights & Classification Diagnosis:
* **DeBERTa** emerged as the top-performing model, capturing the highest **Overall Accuracy (86.50%)** and a leading **AUC Score (0.9238)**. It proved exceptional at identifying explicit complaints, capturing a peak recall of 0.95 for the negative class, confirming that its disentangled attention layer is highly sensitive to nuanced consumer grievances.
* **RoBERTa** demonstrated superior capability in capturing linguistic ambiguity within moderate opinions, achieving the highest **F1-Macro score of 0.7154**. It significantly outperformed the other models in the minority **Neutral class**, securing a recall of 0.32 (compared to BERT's 0.20 and DeBERTa's 0.17).
* **The Neutral Class Challenge:** All three Transformer models faced a noticeable drop in performance on the Neutral class. This is primarily caused by data imbalance (neutral samples ~9.8%) and the inherently mixed sentiment structure of airline reviews, where praise and complaints often coexist in a single sentence.

## 🧠 4. Semantic Interpretation via SHAP (XAI Module)
To understand the inner mathematical alignment of the Transformers, the **SHAP (SHapley Additive exPlanations)** framework was integrated to visualize token-level attribution:

* **Negative Class Attributions:** SHAP analysis revealed that negative predictions are strongly driven by explicit emotional triggers such as `poor` (+0.156), `disappointing` (+0.143), and `dirty` (+0.106). DeBERTa also showed high sensitivity to intensifiers like `very` (+0.142) which amplify negative confidence.
* **Positive Class Attributions:** Positive predictions were influenced by satisfaction-related terms such as `hospitality` (+0.072), `good`, `maintained`, and comfort-related features like `food`. RoBERTa additionally emphasized certainty markers such as `definitely` (+0.031) as strong positive drivers.
