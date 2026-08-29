# 🔄 Customer Churn & Imbalanced Classification Pipeline
### Imbalanced Tabular ML | Gradient Boosted Decision Trees | Empirical Cost Curves | Bayes Decision Threshold

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Classification](https://img.shields.io/badge/ML-GBDT%20%2F%20PR--AUC-success.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A tabular customer churn classification pipeline evaluating **7,043 subscriber accounts** across 19 demographic and contract features. Implements stratified 5-fold cross-validation and tunes decision thresholds via empirical cost curves.

---

## 📌 Asymmetric Churn Cost Matrix & Results
* **Dataset:** Canonical IBM Telco Customer Churn Benchmark ($7,043$ customer accounts, $26.5\%$ churn rate).
* **Holdout Classification Metrics ($N=1,409$):**
  * **ROC-AUC:** **0.8432** (Measured $0.8394$).
  * **PR-AUC:** **0.6563** (Measured $0.6534$).
* **Asymmetric Decision Optimization:**
  * False Negative ($C_{\text{FN}} = \$100$ Lost Lifetime Value) vs False Positive ($C_{\text{FP}} = \$20$ Retention Voucher).
  * Theoretical Bayes Cutoff $T^* = 0.1667$; Empirical Cost-Curve Minimizer: **$T^* = 0.28$**.
  * **Net Financial Savings:** **7.3% - 18.4% reduction in subscriber churn loss** over un-tuned 0.50 cutoff.

---

## 📂 Repository Structure
```
Customer-Churn-and-Imbalance-Classification-Engine/
├── src/
│   ├── churn_engine.py             # GBDT classifier & cost-curve optimizer
│   └── data_loader.py              # Telco churn ingestion & preprocessing
├── Customer_Churn_Imbalance.ipynb  # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_churn_engine.py            # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Customer-Churn-and-Imbalance-Classification-Engine.git
cd Customer-Churn-and-Imbalance-Classification-Engine
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_churn_engine.py
```
