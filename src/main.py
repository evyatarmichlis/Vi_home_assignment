import pandas as pd

import os
from sklearn.metrics import classification_report, roc_auc_score

from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.model import ChurnClassifier, TLearner
from src.optimization_outreach import optimize_outreach, plot_roi_curve, run_sensitivity_analysis

# --- Main Pipeline ---
if __name__ == "__main__":
    # 1. Load Data
    print("--- 1. Data Loading ---")
    loader = DataLoader()
    raw_train = loader.get_train_data()
    raw_test = loader.get_test_data()

    # 2. Feature Engineering
    print("\n--- 2. Feature Engineering ---")
    engineer = FeatureEngineer(observation_end_date='2025-07-15')

    df_train, feature_cols = engineer.process(raw_train)
    df_test, _ = engineer.process(raw_test)

    print(f"   Features Generated: {len(feature_cols)}")

    print("\n--- 3. Standard Model Evaluation (Diagnostic) ---")
    classifier = ChurnClassifier()
    classifier.fit(df_train[feature_cols], df_train['churn'])

    if 'churn' in df_test.columns:
        probs = classifier.predict_proba(df_test)
        auc = roc_auc_score(df_test['churn'], probs)
        print(f"   Standard AUC Score: {auc:.4f}")


    print("\n--- 4. Uplift Modeling (T-Learner) ---")
    learner = TLearner()
    learner.fit(df_train, feature_cols)



    uplift_results = learner.predict_uplift(df_test)
    run_sensitivity_analysis(uplift_results, customer_value=100, output_dir='data/processed/plots')

    optimized_df, best_n = optimize_outreach(uplift_results, cost_per_call=5, customer_value=100)

    plot_roi_curve(optimized_df, best_n)

    print("\n--- 5. Saving Deliverables ---")
    os.makedirs('deliverables', exist_ok=True)

    final_list = optimized_df.head(best_n)[['member_id', 'uplift_score', 'rank']]
    final_list.to_csv('deliverables/prioritized_outreach_list.csv', index=False)


    print("\nDONE.")