import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    def __init__(self, output_dir='data/processed/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_correlation_heatmap(self, df, target='churn'):

        print("Generating Correlation Heatmap")

        ignore_cols = ['member_id', 'uplift_score', 'rank']
        cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

        corr = df[cols].corr()

        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300)
        print(f"save plot to {self.output_dir / 'correlation_matrix.png'}")

        plt.close()

        plt.figure(figsize=(10, 8))
        target_corr = corr[target].drop(target).sort_values(ascending=False)
        sns.barplot(x=target_corr.values, y=target_corr.index, palette='coolwarm')
        plt.title(f'Feature Correlation with Target ({target})', fontsize=16)
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_correlation.png', dpi=300)
        print(f"save plot to {self.output_dir / 'target_correlation.png'}")

        plt.close()

    def plot_tsne(self, df, target='churn', n_samples=3000):

        print(" Generating t-SNE Plot (Latent Space)...")

        ignore_cols = ['member_id', 'churn', 'uplift_score', 'rank']
        features = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
        else:
            df_sample = df

        X = df_sample[features].fillna(0)
        y = df_sample[target]

        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=y, palette={0: 'skyblue', 1: 'crimson'},
            alpha=0.6, s=50
        )
        plt.title('t-SNE Projection of Churn Data', fontsize=16)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(title='Churn Status', loc='upper right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tsne_plot.png', dpi=300)
        print(f"save plot to {self.output_dir / 'tsne_plot.png'}")
        plt.close()

    def plot_feature_distributions(self, df, target='churn', top_n=12):


        ignore_cols = ['member_id', 'churn', 'uplift_score', 'rank']
        numeric_cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

        correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        top_features = correlations.head(top_n).index.tolist()

        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            ax = axes[i]

            sns.kdeplot(data=df, x=feature, hue=target, fill=True, common_norm=False,
                        palette={0: 'skyblue', 1: 'crimson'}, alpha=0.3, ax=ax)

            mean_0 = df[df[target] == 0][feature].mean()
            mean_1 = df[df[target] == 1][feature].mean()
            ax.axvline(mean_0, color='skyblue', linestyle='--', label=f'Stay Avg: {mean_0:.2f}')
            ax.axvline(mean_1, color='crimson', linestyle='--', label=f'Churn Avg: {mean_1:.2f}')

            ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            ax.legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300)
        print(f"save plot to {self.output_dir / 'feature_distributions.png'}")

        plt.close()


if __name__ == "__main__":


    loader = DataLoader()
    data = loader.get_all_data()

    eng = FeatureEngineer()
    df_processed,_ = eng.process(data)

    vis = Visualizer()
    vis.plot_correlation_heatmap(df_processed)
    vis.plot_tsne(df_processed)
    vis.plot_feature_distributions(df_processed)

    print("\n✅ All plots generated in data/processed/plots/")