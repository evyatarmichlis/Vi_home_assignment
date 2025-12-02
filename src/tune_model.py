import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from data_loader import DataLoader
from main import FeatureEngineer


# 1. Objective Function (Uses CV for Hyperparameters)
def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**param)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()


# 2. Main Optimization Routine
def run_cv_optimization(X_train, y_train, X_test, y_test, n_trials=30):
    # --- A. Optuna Search ---
    print(f" Starting Optuna CV Optimization ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    print(f"Best CV AUC: {study.best_value:.4f}")

    # --- B. Threshold Tuning via CV Predict ---
    print("Calculating Robust Threshold via Cross-Validation...")
    model_for_cv = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

    # Get OOF probabilities for the entire training set
    cv_probs = cross_val_predict(
        model_for_cv, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, cv_probs)
    f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    best_thresh = thresholds[np.argmax(f1_scores)]

    print(f" Robust Threshold: {best_thresh:.4f}")

    # --- C. Final Training ---
    print("\nRetraining Final Model on 100% of Data...")
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)

    # --- D. Evaluation ---
    print("\n Evaluation on UNSEEN TEST DATA...")
    probs_test = final_model.predict_proba(X_test)[:, 1]
    final_preds = (probs_test >= best_thresh).astype(int)

    print("\n--- FINAL CROSS-VALIDATED RESULTS ---")
    print(classification_report(y_test, final_preds))
    print(f"Final Test AUC: {roc_auc_score(y_test, probs_test):.4f}")

    # Return params so we can print them at the end
    return final_model, best_thresh, best_params


# --- Main Execution ---
if __name__ == "__main__":
    loader = DataLoader()
    engineer = FeatureEngineer(observation_end_date='2025-07-15')

    # Load & Process
    raw_train = loader.get_train_data()
    raw_test = loader.get_test_data()

    df_train, feature_cols = engineer.process(raw_train)
    df_test, _ = engineer.process(raw_test)

    X_train = df_train[feature_cols]
    y_train = df_train['churn']
    X_test = df_test.reindex(columns=feature_cols, fill_value=0)

    if 'churn' in df_test.columns:
        y_test = df_test['churn']

        # Run Pipeline
        final_model, thresh, winning_params = run_cv_optimization(X_train, y_train, X_test, y_test, n_trials=40)

        # --- 🖨️ PRINT FINAL PARAMS HERE ---
        print("\n" + "=" * 50)
        print(" BEST HYPERPARAMETERS (Copy to Config)")
        print("=" * 50)
        for k, v in winning_params.items():
            print(f"  '{k}': {v},")
        print(f"  'threshold': {thresh:.4f}")
        print("=" * 50 + "\n")


#   'n_estimators': 497,
#   'max_depth': 12,
#   'min_samples_leaf': 10,
#   'max_features': sqrt,
#   'class_weight': balanced,
#   'threshold': 0.4096
