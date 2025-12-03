import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def optimize_outreach(uplift_df, cost_per_call=10, customer_value=100):
    """
    Determines optimal 'n' by simulating P&L based on Uplift Scores.

    Args:
        uplift_df (pd.DataFrame): Must contain 'uplift_score' column.
        cost_per_call (float): Marginal cost of intervention.
        customer_value (float): Value of a saved customer.

    Returns:
        tuple: (DataFrame with financial cols, int best_n)
    """
    print("\nOptimizing Outreach Size (n)...")

    # 1. Sort by Uplift (Best candidates first)
    df = uplift_df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)

    # 2. Calculate Economics
    # Revenue = Probability Delta * Customer Value
    df['expected_revenue'] = df['uplift_score'] * customer_value

    df['cumulative_cost'] = (df.index + 1) * cost_per_call

    df['cumulative_revenue'] = df['expected_revenue'].cumsum()

    # Profit = Total Revenue - Total Cost
    df['roi_profit'] = df['cumulative_revenue'] - df['cumulative_cost']
    df['rank'] = df.index + 1

    # 3. Find Peak Profit
    best_idx = df['roi_profit'].idxmax()
    best_n = df.loc[best_idx, 'rank']
    max_profit = df.loc[best_idx, 'roi_profit']
    if max_profit <= 0:
        # If the maximum profit found is zero or negative, the optimal choice is to call no one.
        best_n = 0
        max_profit = 0
    print(f"   Optimal n: {best_n}")
    print(f"   Projected Profit: ${max_profit:,.2f}")
    return df, best_n


def plot_roi_curve(df, best_n, output_dir='data/processed/plots'):
    """
    Visualizes the Profit Curve to justify 'n' selection.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df['rank'], df['roi_profit'], label='Projected Profit', color='green', linewidth=2)
    plt.axvline(best_n, color='red', linestyle='--', label=f'Optimal n={best_n}')

    plt.title('Outreach Optimization Curve (Profit vs Volume)', fontsize=14)
    plt.xlabel('Number of Users Contacted (n)')
    plt.ylabel('Net Profit ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'optimization_curve.png')
    plt.savefig(save_path)
    plt.close()

    print(f"   ROI Curve saved to {save_path}")


def run_sensitivity_analysis(uplift_df, customer_value=100, output_dir='data/processed/plots'):
    """
     Tests multiple cost scenarios to find  'n'.
    Since cost is 'marginal', we test ranges like 1%, 5%, 10% of LTV.
    """
    print("\nRunning Sensitivity Analysis on Costs...")
    os.makedirs(output_dir, exist_ok=True)

    scenarios = [
        {'cost': 0, 'label': 'Free Cost (0% of costumer value)'},
        {'cost': 1, 'label': 'Very Low Cost (1% of costumer value)'},
        {'cost': 5, 'label': 'Low Cost (5% of costumer value)'},
        {'cost': 10, 'label': 'Moderate Cost (10% of costumer value)'},
        {'cost': 20, 'label': 'high Cost (20% of costumer value)'}

    ]

    plt.figure(figsize=(12, 7))

    best_ns = []

    for s in scenarios:
        cost = s['cost']
        label = s['label']


        df_opt, best_n = optimize_outreach(uplift_df, cost_per_call=cost, customer_value=customer_value)
        best_ns.append(best_n)

        # Plot Curve
        plt.plot(df_opt['rank'], df_opt['roi_profit'], label=f"{label} -> n={best_n}")
        if best_n > 0:
            profit_at_n = float(df_opt.loc[df_opt['rank'] == best_n, 'roi_profit'].iloc[0])
        else:
            profit_at_n = 0

        plt.scatter([best_n], [profit_at_n], s=100)

    plt.title('Sensitivity Analysis: Optimal "n" at different Marginal Costs', fontsize=14)
    plt.xlabel('Number of Users Contacted')
    plt.ylabel('Projected Profit ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(save_path)
    plt.close()



def validate_uplift_effectiveness(ranking_path, labels_path, n):
    """
    Validates the T-Learner model by calculating the 'Realized Uplift'
    on the selected Top N candidates.
    """
    print(f"\nValidating Uplift Effectiveness for Top {n} Candidates...")

    try:
        ranking_df = pd.read_csv(ranking_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError:
        print("Cannot verify: Data files missing.")
        return

    top_n_ids = ranking_df.sort_values('rank').head(n)['member_id']
    target_group = labels_df[labels_df['member_id'].isin(top_n_ids)].copy()
    real_treated = target_group[target_group['outreach'] == 1]
    real_control = target_group[target_group['outreach'] == 0]
    if len(real_treated) == 0 or len(real_control) == 0:
        print("Not enough history in test set to compare Treated vs Control.")
        return
    churn_rate_with_call = real_treated['churn'].mean()
    churn_rate_no_call = real_control['churn'].mean()
    realized_uplift = churn_rate_no_call - churn_rate_with_call
    print(f"   ------------------------------------------------")
    print(f"   Analysis of Selected Group (n={len(target_group)}):")
    print(f"   ACTUAL Churn Rate (No Call):   {churn_rate_no_call:.2%} (Control)")
    print(f"   ACTUAL Churn Rate (With Call): {churn_rate_with_call:.2%} (Treated)")
    print(f"   ------------------------------------------------")
    print(f"    REALIZED UPLIFT:            {realized_uplift:.2%}")
    print(f"   ------------------------------------------------")

    if realized_uplift > 0:
        print(f" SUCCESS: Calling these specific people reduces churn by {realized_uplift:.1%}.")
    else:
        print(f" WARNING: Calling this group didn't help (or made it worse).")

