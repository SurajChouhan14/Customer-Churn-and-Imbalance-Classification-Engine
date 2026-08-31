# 📉 Customer Churn & Imbalanced Classification Pipeline
> **Gradient Boosted Decision Trees, Stratified 5-Fold Cross-Validation, and Decision-Theoretic Threshold Optimization**  
> *Tabular ML · Imbalanced Classification · scikit-learn · 5-Fold CV · Cost-Curve Optimization · ROC-AUC & PR-AUC*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/SurajChouhan14/Customer-Churn-and-Imbalance-Classification-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/Customer-Churn-and-Imbalance-Classification-Engine/actions)
[![Classification](https://img.shields.io/badge/ML-HistGradientBoosting-blue.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Mathematical Architecture
An enterprise customer churn prediction and retention optimization engine ingesting **7,043 subscriber records** across 19 demographic, service, and contract features with a $73.5\% / 26.5\%$ class imbalance. Integrates `HistGradientBoostingClassifier` with balanced class weights, train-side 5-fold stratified cross-validation, and Bayesian decision theory to establish asymmetric cost-optimal operational cutoffs.

```
                   Telecom Customer Records (7,043 Accounts)
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
  Categorical & Numeric Preprocessing                       Class Imbalance Weighting
   • One-Hot Encoding (16 contract/service features)          • 73.5% Retained vs 26.5% Churned
   • StandardScaler (tenure, monthly, total charges)          • class_weight='balanced' (2.77:1)
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       ▼
                 Train-Side 5-Fold Stratified Cross-Validation
                  [ Cost Minimization: C_FN=$100, C_FP=$20 ]
                                       │
                                       ▼
                 Out-of-Sample Holdout Evaluation (N=1,409)
              [ ROC-AUC = 0.8394 | PR-AUC = 0.6534 | Cutoff T* = 0.30 ]
```

### 1. Asymmetric Decision Theory & Cost Optimization
Under asymmetric commercial misclassification penalties ($C_{\text{FN}} = \$100$ customer LTV loss vs $C_{\text{FP}} = \$20$ retention incentive cost), standard $0.50$ probability thresholds produce sub-optimal commercial loss. The optimal decision threshold $T^*$ is selected by minimizing total expected loss across train-side 5-fold cross-validation:
$$T^* = \arg\min_{T} \left[ C_{\text{FN}} \cdot \text{FN}(T) + C_{\text{FP}} \cdot \text{FP}(T) \right], \quad T^*_{\text{theoretical}} = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{20}{120} = 0.1667$$

---

## 📊 Benchmark Performance & Holdout Evaluation

### Holdout Validation Performance ($80/20\text{ Stratified Holdout Split}, N = 1,409\text{ Accounts}$)

| Metric / Parameter | Resume Baseline | Measured Out-of-Sample Performance | Status |
|---|:---:|:---:|:---:|
| **Area Under ROC Curve (ROC-AUC)** | $0.8432$ | **$0.8394$** | **PASSED (Noise-Level $\Delta = 0.0038$)** |
| **Precision-Recall AUC (PR-AUC)** | $0.6563$ | **$0.6534$** | **PASSED (Noise-Level $\Delta = 0.0029$)** |
| **Brier Calibration Score Loss** | Baseline Loss | **$0.1620$** | High Probabilistic Calibration |
| **CV Train-Tuned Optimal Cutoff ($T^*$)** | $0.28$ | **$0.30$** | **PASSED (Leakage-Free Train CV)** |
| **Out-of-Sample Expected Cost @ $T^*$** | Cost Baseline | **$\$12,460.00$** | Optimal Asymmetric Policy |
| **Out-of-Sample Expected Cost @ $T=0.50$** | Default Policy | **$\$13,200.00$** | Standard Default Cutoff |
| **Net Cost Reduction vs Default Cutoff** | Theoretical Uplift | **$+5.61\%$** | **$\$740\text{ Saved per 1,409 Accounts}$** |

---

## 📁 Repository Structure

```text
Customer-Churn-and-Imbalance-Classification-Engine/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI test & validation workflow
├── .gitignore                          # Git exclusions (pycache, logs)
├── Customer_Churn_Classification.ipynb # Interactive evaluation & EDA notebook
├── README.md                           # Documentation & mathematical architecture
├── data/
│   └── customer_churn.csv              # 7,043 telecom customer subscriber records
├── figures/
│   ├── figure_1_roc_and_pr_curves.png  # ROC and Precision-Recall evaluation curves
│   └── figure_2_cost_curve_threshold.png # Asymmetric cost optimization curve
├── requirements.txt                    # Production dependencies
├── run_pipeline.py                     # 3-phase execution pipeline with 5-fold CV
├── src/
│   ├── __init__.py                     # Package init
│   └── churn_classifier.py             # ColumnTransformer & GBDT classifier engine
└── test_churn_engine.py                # 4 automated unit & pipeline tests
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Customer-Churn-and-Imbalance-Classification-Engine.git
cd Customer-Churn-and-Imbalance-Classification-Engine
pip install -r requirements.txt
```

### 2. Run Pipeline Benchmark
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
python test_churn_engine.py
```
