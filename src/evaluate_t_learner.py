import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_t_learner(learner, df_test, feature_cols):
    """
    Audits the T-Learner by evaluating its two internal models separately.
    It treats the Control Model and Treated Model as standard classifiers
    on their respective populations.
    """
    print("\nAuditing T-Learner Performance (Classification Task)...")

    # 1. Align Features (Critical for safety)
    # We use the learner's internal feature list to be safe
    if hasattr(learner, 'feature_names'):
        cols = learner.feature_names
    else:
        cols = [c for c in feature_cols if c != 'outreach']

    X_test = df_test.reindex(columns=cols, fill_value=0)


    mask_treated = df_test['outreach'] == 1
    mask_control = df_test['outreach'] == 0

    X_treated = X_test[mask_treated]
    y_treated = df_test[mask_treated]['churn']

    X_control = X_test[mask_control]
    y_control = df_test[mask_control]['churn']

    # --- 3. Evaluate CONTROL Model (Natural Churn) ---
    print("\n📉 --- Control Model Report (Performance on Ignored Users) ---")
    if len(X_control) > 0:
        # Predict on Control Population
        preds_ctrl = learner.model_control.predict(X_control)
        probs_ctrl = learner.model_control.predict_proba(X_control)[:, 1]

        print(classification_report(y_control, preds_ctrl))
        auc_ctrl = roc_auc_score(y_control, probs_ctrl)
        print(f"   AUC Score (Control): {auc_ctrl:.4f}")
    else:
        print("   ⚠No Control users found in Test Set.")

    print("\n📈 --- Treated Model Report (Performance on Contacted Users) ---")
    if len(X_treated) > 0:
        preds_trt = learner.model_treated.predict(X_treated)
        probs_trt = learner.model_treated.predict_proba(X_treated)[:, 1]

        print(classification_report(y_treated, preds_trt))
        auc_trt = roc_auc_score(y_treated, probs_trt)
        print(f"   AUC Score (Treated): {auc_trt:.4f}")
    else:
        print("    No Treated users found in Test Set.")

    print("\nVisualizing Uplift Scores...")
    uplift_res = learner.predict_uplift(df_test)

    plt.figure(figsize=(10, 6))
    sns.histplot(uplift_res['uplift_score'], bins=50, kde=True, color='purple')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Distribution of Predicted Uplift Scores', fontsize=14)
    plt.xlabel('Uplift Score (Probability of Saving Customer)')
    plt.ylabel('Count of Users')

    # Calculate stats
    n_persuadable = (uplift_res['uplift_score'] > 0.05).sum()  # >5% gain
    n_sleeping_dogs = (uplift_res['uplift_score'] < -0.05).sum()  # >5% harm

    stats_text = (
        f"Total Users: {len(uplift_res)}\n"
        f"Persuadables (>5%): {n_persuadable}\n"
        f"Sleeping Dogs (<-5%): {n_sleeping_dogs}"
    )
    plt.text(0.02, plt.ylim()[1] * 0.8, stats_text, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig('data/processed/plots/uplift_distribution.png')
    print("   Distribution plot saved to data/processed/plots/uplift_distribution.png")


if __name__ == "__main__":
    # Integration test
    from src.data_loader import DataLoader
    from src.feature_eng import FeatureEngineer
    from src.model import TLearner

    loader = DataLoader()
    raw_train = loader.get_train_data()
    raw_test = loader.get_test_data()

    eng = FeatureEngineer()
    df_train, cols = eng.process(raw_train)
    df_test, _ = eng.process(raw_test)

    learner = TLearner()
    learner.fit(df_train, cols)

    evaluate_t_learner(learner, df_test, cols)