import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ChurnClassifier:
    """
    Standard Random Forest Wrapper for Churn Prediction.
    Handles column alignment automatically to prevent crashes on test data.
    """

    def __init__(self, params=None):
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        self.model = RandomForestClassifier(**params)
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)

    def predict(self, X):
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict(X_aligned)

    def predict_proba(self, X):
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict_proba(X_aligned)[:, 1]

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class TLearner:
    """
    Uplift Model (Two-Model Approach).
    Trains two separate models:
    1. Control Model (Users who were IGNORED)
    2. Treated Model (Users who were CONTACTED)
    """

    def __init__(self, base_params=None):
        if base_params is None:
            base_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        # Two independent models
        self.model_control = RandomForestClassifier(**base_params)
        self.model_treated = RandomForestClassifier(**base_params)
        self.feature_names = []

    def fit(self, df_train, feature_cols, target_col='churn', treatment_col='outreach'):
        self.feature_names = [c for c in feature_cols if c != treatment_col]

        # Split Data
        df_control = df_train[df_train[treatment_col] == 0]
        df_treated = df_train[df_train[treatment_col] == 1]

        self.model_control.fit(df_control[self.feature_names], df_control[target_col])

        self.model_treated.fit(df_treated[self.feature_names], df_treated[target_col])

    def predict_uplift(self, df_test):
        """
        Returns a DataFrame with Uplift Scores.
        Uplift = P(Churn|Ignored) - P(Churn|Called)
        """
        # Align features
        X = df_test.reindex(columns=self.feature_names, fill_value=0)

        # Predict Probabilities for both scenarios
        prob_if_ignored = self.model_control.predict_proba(X)[:, 1]
        prob_if_called = self.model_treated.predict_proba(X)[:, 1]

        # Calculate Benefit (Positive Uplift = Calling helps)
        uplift_scores = prob_if_ignored - prob_if_called

        results = pd.DataFrame({
            'member_id': df_test['member_id'],
            'prob_if_ignored': prob_if_ignored,
            'prob_if_called': prob_if_called,
            'uplift_score': uplift_scores
        })

        return results