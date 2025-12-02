
import os
from sklearn.metrics import classification_report, roc_auc_score
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.model import ChurnClassifier, TLearner
from src.optimization_outreach import optimize_outreach, plot_roi_curve, run_sensitivity_analysis
from src.vis import Visualizer


if __name__ == "__main__":
    # Toggle to True if you want to generate EDA plots (takes time)
    VIS_FLAG = False

    # 1. Load Data
    print("--- 1. Data Loading ---")
    loader = DataLoader()
    raw_train = loader.get_train_data()
    raw_test = loader.get_test_data()

    # 2. Feature Engineering
    print("\n--- 2. Feature Engineering ---")
    # We use the date from instructions as the "Present"
    engineer = FeatureEngineer(observation_end_date='2025-07-15')

    df_train, feature_cols = engineer.process(raw_train)
    df_test, _ = engineer.process(raw_test)  # Ignore test cols, we use train cols for consistency

    print(f"   🧠 Features Generated: {len(feature_cols)}")

    # (Optional) Visualization
    if VIS_FLAG:
        print("\n   🎨 Generating Visualizations...")
        vis = Visualizer()
        vis.plot_correlation_heatmap(df_train)
        vis.plot_tsne(df_train)
        vis.plot_feature_distributions(df_train)

    # 3. Standard Model Evaluation (Diagnostic)
    # This proves we can predict churn before we try to predict uplift
    print("\n--- 3. Standard Model Evaluation (Diagnostic) ---")
    classifier = ChurnClassifier()
    classifier.fit(df_train[feature_cols], df_train['churn'])

    if 'churn' in df_test.columns:
        probs = classifier.predict_proba(df_test)
        auc = roc_auc_score(df_test['churn'], probs)
        print(f" Standard AUC Score: {auc:.4f}")

    # 4. Uplift Modeling (The Goal)
    print("\n--- 4. Uplift Modeling (T-Learner) ---")
    learner = TLearner()
    learner.fit(df_train, feature_cols)

    # Predict Uplift on Target Population (Test Set)
    uplift_results = learner.predict_uplift(df_test)

    # 5. Financial Optimization
    # Step A: Run Sensitivity Analysis to see the effect of unknown costs ($0 $1, $5, $10, $20)
    # This generates the 'sensitivity_analysis.png' plot
    run_sensitivity_analysis(uplift_results, customer_value=100)

    # Step B: Select the "Robust" middle scenario ($5) for the final list
    print("\n--- 5. Final List Generation (Cost Scenario: $5) ---")
    optimized_df, best_n = optimize_outreach(uplift_results, cost_per_call=5, customer_value=100)

    # Generate the specific ROI curve for this choice
    plot_roi_curve(optimized_df, best_n)

    # 6. Save Deliverables
    os.makedirs('deliverables', exist_ok=True)  # Save to root folder, not src/

    final_list = optimized_df.head(best_n)[['member_id', 'uplift_score', 'rank']]
    final_list.to_csv('deliverables/prioritized_outreach_list.csv', index=False)

    print(f"\n✅ DONE. Final list of top {best_n} members saved to 'deliverables/prioritized_outreach_list.csv'.")