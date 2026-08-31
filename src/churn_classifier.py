"""
Customer Churn & Imbalanced Classification Engine.
Ingests 7,043 telecommunications subscriber records, handles 73.5%/26.5% class imbalance,
trains Gradient Boosted Decision Trees with 5-fold cross-validation, and tunes Bayesian cost-optimal decision thresholds.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, brier_score_loss
from typing import Dict, Any, Tuple


class CustomerChurnClassifierEngine:
    """
    Imbalanced Machine Learning & Decision-Theoretic Customer Churn Classifier.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cat_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
        ]
        self.num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.pipeline = None
        self.optimal_threshold = 0.30

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Cleans TotalCharges whitespace/missing values and builds binary churn label.
        """
        df_clean = df.copy()
        # Coerce TotalCharges from string to numeric and impute missing
        total_num = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"] = total_num.fillna(total_num.median())
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

        y = (df_clean["Churn"] == "Yes").astype(int)
        X = df_clean[self.num_cols + self.cat_cols]
        return X, y

    def train_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.20) -> Tuple[Any, Dict[str, Any]]:
        """
        Trains Gradient Boosted Trees, selects cost-optimal threshold via train-side 5-fold CV,
        and evaluates out-of-sample ROC-AUC, PR-AUC, and financial cost savings on held-out test data.
        """
        X, y = self.preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)
            ]
        )

        # Gradient Boosted Classifier tuned for imbalanced churn dynamics
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            learning_rate=0.06,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.random_state
        )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # 1. Train-Side 5-Fold Stratified Cross-Validation for Threshold Optimization
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_train_probs = cross_val_predict(self.pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

        # Asymmetric Decision Theory Cost Optimization (FN=$100 LTV Loss, FP=$20 Promotion Cost)
        c_fn = 100.0
        c_fp = 20.0
        thresholds = np.linspace(0.05, 0.95, 91)
        train_costs = []

        for t in thresholds:
            y_train_pred_t = (cv_train_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred_t).ravel()
            total_cost = (fn * c_fn) + (fp * c_fp)
            train_costs.append(total_cost)

        opt_idx = np.argmin(train_costs)
        self.optimal_threshold = float(thresholds[opt_idx])

        # 2. Fit Full Pipeline on Training Set
        self.pipeline.fit(X_train, y_train)

        # 3. Evaluate on Held-Out Test Set with Frozen T*
        y_test_prob = self.pipeline.predict_proba(X_test)[:, 1]

        roc_auc = float(roc_auc_score(y_test, y_test_prob))
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        pr_auc = float(auc(recall, precision))
        brier = float(brier_score_loss(y_test, y_test_prob))

        # Evaluate Out-of-Sample Financial Cost Savings at Frozen T* vs Default 0.50
        y_test_pred_opt = (y_test_prob >= self.optimal_threshold).astype(int)
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_test_pred_opt).ravel()
        test_cost_opt = (fn_opt * c_fn) + (fp_opt * c_fp)

        y_test_pred_50 = (y_test_prob >= 0.50).astype(int)
        tn_50, fp_50, fn_50, tp_50 = confusion_matrix(y_test, y_test_pred_50).ravel()
        test_cost_50 = (fn_50 * c_fn) + (fp_50 * c_fp)

        cost_savings_pct = float(((test_cost_50 - test_cost_opt) / test_cost_50) * 100.0)

        metrics = {
            "total_accounts": len(df),
            "retained_pct": float((1 - y.mean()) * 100.0),
            "churn_pct": float(y.mean() * 100.0),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "optimal_threshold": round(self.optimal_threshold, 2),
            "test_cost_optimal_usd": float(test_cost_opt),
            "test_cost_default_usd": float(test_cost_50),
            "cost_savings_pct": round(cost_savings_pct, 2)
        }

        return self.pipeline, metrics

    def predict_churn_risk(self, account_dict: dict) -> Dict[str, Any]:
        """Predicts probability of churn and optimal commercial retention action."""
        df_single = pd.DataFrame([account_dict])
        total_num = pd.to_numeric(df_single["TotalCharges"], errors="coerce")
        df_single["TotalCharges"] = total_num.fillna(df_single["MonthlyCharges"] * df_single["tenure"])
        df_single["SeniorCitizen"] = df_single["SeniorCitizen"].astype(str)

        prob = float(self.pipeline.predict_proba(df_single[self.num_cols + self.cat_cols])[:, 1][0])
        action = "High Risk - Dispatch Retention Offer" if prob >= self.optimal_threshold else "Low Risk - Standard Engagement"
        return {
            "churn_probability": round(prob, 4),
            "is_churn_risk": bool(prob >= self.optimal_threshold),
            "optimal_threshold_applied": self.optimal_threshold,
            "recommended_action": action
        }
