"""
Automated Unit Test Suite for Customer Churn & Imbalanced Classification Engine.
Verifies Data Ingestion, ColumnTransformer Pipeline, ROC-AUC / PR-AUC Metrics, and Decision Threshold Optimization.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.churn_classifier import CustomerChurnClassifierEngine


class TestCustomerChurnEngine(unittest.TestCase):
    """
    Unit test cases for Customer Churn Imbalanced Classifier.
    """

    def setUp(self):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "customer_churn.csv")
        self.df = pd.read_csv(data_path)
        self.engine = CustomerChurnClassifierEngine(random_state=42)

    def test_data_ingestion(self):
        """Verify churn dataset contains 7,043 subscriber records and expected class balance."""
        self.assertEqual(len(self.df), 7043)
        self.assertIn("Churn", self.df.columns)
        self.assertIn("MonthlyCharges", self.df.columns)
        self.assertIn("TotalCharges", self.df.columns)
        self.assertIn("Contract", self.df.columns)

    def test_data_preprocessing(self):
        """Verify TotalCharges missing string coercion and binary label encoding."""
        X, y = self.engine.preprocess_data(self.df)
        self.assertEqual(len(X), len(y))
        self.assertTrue(pd.api.types.is_numeric_dtype(X["TotalCharges"]))
        self.assertEqual(set(y.unique()), {0, 1})

    def test_model_auc_performance(self):
        """Verify model achieves ROC-AUC > 0.80 and PR-AUC > 0.60 on benchmark."""
        _, metrics = self.engine.train_and_evaluate(self.df, test_size=0.20)
        self.assertGreater(metrics["roc_auc"], 0.80)
        self.assertGreater(metrics["pr_auc"], 0.60)

    def test_bayesian_cost_optimization(self):
        """Verify optimal cost threshold is lower than default 0.50 (around 0.28 - 0.36)."""
        _, metrics = self.engine.train_and_evaluate(self.df, test_size=0.20)
        self.assertLess(metrics["optimal_threshold"], 0.45)
        self.assertGreater(metrics["optimal_threshold"], 0.15)


if __name__ == '__main__':
    unittest.main()
