# Airlines Review Sentiment Analysis & Multi-Label Aspect Detection Using Fine-Tuned Transformer Models and SHAP

## 📌 1. Project Overview
This repository contains an advanced deep learning and Explainable AI (XAI) pipeline developed for a comprehensive, comparative text analytics study on international aviation customer experience. Utilizing raw customer reviews scraped from the **Skytrax** platform, this project evaluates state-of-the-art **Transformer** architectures by fine-tuning them across two distinct computational linguistics tasks simultaneously:
1. **Multi-class Sentiment Classification:** Categorizing customer reviews into Positive, Negative, or Neutral classes based on explicit rating-to-sentiment mapping.
2. **Multi-label Aspect Detection:** Identifying the simultaneous presence of 5 distinct core airline service quality dimensions (*Food & Beverage, Staff Service, Wifi Inflight, Ground Service, and Seat & Cabin*) within a single review text.

The project evaluates and benchmarks performance across two regional focus datasets:
* **Asian Airlines Dataset:** Focused on *Cathay Pacific Airways, Singapore Airlines, and Turkish Airlines*.
* **Middle Eastern Airlines Dataset:** Focused on *Qatar Airways, Emirates, and Etihad Airways*.

This study follows strict scientific methodologies and is part of an ongoing **academic publication pipeline**.

## ⚙️ 2. Dataset Profiles & Stratified Partitioning
Both datasets utilize a programmatic thresholding system to convert passenger numerical ratings (1-10) into objective sentiment targets:
* **Negative (Label 0):** Ratings from 1 to 4/10 (clear service dissatisfaction).
* **Neutral (Label 1):** Ratings from 5 to 6/10 (moderate or ambiguous experiences).
* **Positive (Label 2):** Ratings from 7 to 10/10 (high customer satisfaction).

### A. Asian Airlines Dataset Profile
* **Total Volume:** 2,999 raw review records.
* **Class Distribution:** Negative: 1,384 reviews (46.1%) | Positive: 1,318 reviews (43.9%) | Neutral: 297 reviews (9.9%).
* **Data Splitting:** Partitioned using an **80:20 Stratified Train-Test Split** (2,399 training records and 600 validation records) to guarantee a perfectly uniform class distribution across both subsets.

### B. Middle Eastern Airlines Dataset Profile
* **Total Volume:** Programmatically structured validation corpus evaluated across 600 unseen documents.
* **Class Distribution:** Heavily skewed with Negative and Positive domains acting as the primary classes, while the Neutral class remains a distinct minority making up exactly 9.8% of the data (59 out of 600 validation data points).
* **Data Splitting:** Evaluated using a matching stratified test architecture to maintain rigorous evaluation standards across regional domains.

## 🛠️ 3. Transformer Tokenization Layer Comparison
Prior to model training, a comparative subword tokenization analysis was performed to evaluate how each architecture represents complex, specialized aviation feedback:
* **BERT (`bert-base-uncased`):** Implements WordPiece tokenization, yielding 55 tokens on the standard baseline sample. Out-of-vocabulary words are broken down using the `##` prefix (e.g., splitting `lavatories` into `lava##tori##es` or `cathay` into `cat##hay`).
* **RoBERTa (`roberta-base`):** Formulated on Byte-Pair Encoding (BPE) with dynamic masking, yielding 59 tokens on the baseline sample. It utilizes a specialized `Ġ` character as a prefix to preserve spacing data.
* **DeBERTa (`microsoft/deberta-v3-base`):** Features a SentencePiece tokenizer. By isolating token content from relative positioning, it achieves the highest syntactic efficiency, yielding only 52 tokens and preserving words like `lavatories` intact as a single block (`▁lavatories`).

## 📊 4. Benchmarking & Comparative Performance Evaluation
All fine-tuned Transformer networks were thoroughly evaluated across both regional datasets using identical training parameters.

### Performance Benchmark Table:
| Regional Dataset | Model Architecture | Validation Accuracy | F1-Macro | AUC Score |
| :--- | :--- | :---: | :---: | :---: |
| **Asian Airlines** | BERT (Base Uncased) | 85.00% | 0.6864 | 0.9103 |
| | RoBERTa (Base) | 84.17% | **0.7080** | 0.9065 |
| | DeBERTa (Base v3) | **85.67%** | 0.6808 | **0.9238** |
| **Middle Eastern Airlines** | BERT (Base Uncased) | 85.17% | 0.6760 | 0.9090 |
| | RoBERTa (Base) | 86.33% | 0.7154 | 0.9190 |
| | DeBERTa (Base v3) | **86.50%** | **0.7203** | **0.9167** |

### Evaluation Insights & Classification Diagnosis:
* **DeBERTa Model Superiority:** DeBERTa consistently led overall **Accuracy (85.67% / 86.50%)** and **AUC Scores (0.9238)** across both studies. Driven by its disentangled attention layers, it proved exceptional at capturing highly explicit consumer grievances, capturing a peak recall of 0.95 for the negative class in the Middle Eastern dataset. However, it demonstrated a highly aggressive behavior towards complaints, frequently misclassifying neutral reviews containing minor critique into the fully negative category.
* **RoBERTa Model Balance:** RoBERTa showed a superior capability for capturing structural and semantic ambiguity within moderate opinions. It achieved the highest **F1-Macro score of 0.7080** on the Asian dataset and a strong 0.7154 on the Middle Eastern dataset. This performance was driven by its ability to resolve the neutral class, securing a neutral class recall of 0.32 compared to BERT's 0.20 and DeBERTa's 0.17.
* **The Neutral Class Challenge:** Across both datasets, all three Transformer models faced a noticeable drop in performance on the Neutral class. This is a collective result of severe data imbalance (~9.8% of the data across sets) and the mixed, dual-polarity nature of customer reviews, where passengers frequently combine positive praise and negative complaints in a single sentence.

## 🧠 5. Explainable AI (XAI Framework via SHAP)
To avoid the "black box" dilemma, token-level mathematical attributions were generated using the **SHAP (SHapley Additive exPlanations)** framework to isolate exactly how structural and semantic combinations guide model behavior:

* **Negative Class Attributions:** SHAP analysis revealed that the models' negative classification boundaries are strongly driven by explicit emotional triggers such as `poor` (+0.156), `disappointing` (+0.143), and `dirty` (+0.106). DeBERTa, in particular, showed high sensitivity to sentence intensifiers like `very` (+0.142) to heavily scale up model confidence.
* **Neutral Class Attributions:** The SHAP framework captured a complex "tug-of-war" effect. Mixed evaluations were triggered when negative operators like `not` (+0.021) were mathematically balanced by moderate qualifiers like `okay` (+0.007) or `expected` (+0.053), successfully pulling predictions back to the baseline expectation.
* **Positive Class Attributions:** The models distributed heavy positive weights across explicit service-satisfaction keywords like `hospitality` (+0.072), `good`, `maintained`, and physical comfort features like `food`. RoBERTa uniquely emphasized certainty markers like `definitely` (+0.031) as a major driver to lock down positive predictions.
