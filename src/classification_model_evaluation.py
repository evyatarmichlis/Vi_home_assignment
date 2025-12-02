import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# 1. Setup Data (Hardcoded from your results)
data = {
    'Model': [
        'Baseline (Random)',
        'Simple Features',
        'Complex (Old NLP)',
        'Short-Window',
        'LLM Anchors (Base)',
        'Short-Window + LLM',
        'SW + LLM + PCA (Deep)',
        'XGBoost',
        'Final Optuna RF'
    ],
    'AUC Score': [0.4891, 0.5732, 0.5886, 0.6144, 0.6471, 0.6568, 0.6573, 0.6504, 0.6611],
    'Churn Recall': [0.48, 0.20, 0.14, 0.48, 0.59, 0.57, 0.58, 0.52, 0.63],
    'Churn Precision': [0.20, 0.29, 0.33, 0.28, 0.28, 0.29, 0.29, 0.30, 0.29]
}

df = pd.DataFrame(data)

# Sort by AUC to show progression
df = df.sort_values(by='AUC Score', ascending=True)

# 2. Setup Plot Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})


def plot_metric(metric_col, title, filename, color_palette="viridis"):
    plt.figure(figsize=(12, 6))

    # Create Bar Plot
    ax = sns.barplot(
        x='Model',
        y=metric_col,
        data=df,
        palette=color_palette,
        edgecolor=".2"
    )

    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontweight='bold')

    # Highlight the Baseline line
    baseline_val = df[df['Model'] == 'Baseline (Random)'][metric_col].values[0]
    plt.axhline(baseline_val, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_val:.2f})')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(metric_col)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    # Save
    os.makedirs('data/processed/plots', exist_ok=True)
    plt.savefig(f'data/processed/plots/{filename}')
    print(f"Saved {filename}")
    plt.close()


# 3. Generate 3 Key Plots
if __name__ == "__main__":
    plot_metric('AUC Score', 'AUC Score Progression', 'comparison_auc.png', 'Blues_d')

    plot_metric('Churn Recall', 'Recall Progression', 'comparison_recall.png', 'Greens_d')

    plot_metric('Churn Precision', 'Precision Progression', 'comparison_precision.png', 'Reds_d')