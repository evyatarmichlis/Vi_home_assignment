import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class FeatureEngineer:
    def __init__(self, observation_end_date='2025-07-15'):
        """
        Initializes the Feature Engineering pipeline.

        Sets up the cutoff date, loads the NLP model (if available), and defines
        configurations for Claims Severity mapping and Semantic Anchors.
        """
        self.cutoff = pd.to_datetime(observation_end_date)
        self.active_threshold_days = 3

        # --- 1. Health Config (Severity) ---
        self.severity_map = {
            'C': 3, 'E': 3, 'I': 3, 'M': 2, 'K': 2, 'J': 1, 'H': 1, 'Z': 0
        }
        # --- Deep Feature Config ---
        self.pca = None
        self.pca_components = 3
        # --- 2. NLP Config (Anchors) ---
        self.use_nlp = False
        if SentenceTransformer:
            print("Loading NLP Model (MiniLM) for Semantic Scoring...")
            self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.anchors = {
                'risk': "cancel subscription refund close account support problem complaint",
                'medical': "symptoms diagnosis doctor hospital pain medicine treatment condition",
                'fitness': "fitness gym workout diet nutrition running yoga healthy",
                'lifestyle': "sports football travel news weather music movies celebrities"
            }
            # Pre-compute anchor embeddings
            self.anchor_names = list(self.anchors.keys())
            self.anchor_embeddings = self.nlp_model.encode(list(self.anchors.values()))
            self.use_nlp = True
        else:
            print("sentence-transformers not found. Skipping NLP features.")

    def _analyze_behavior(self, df, date_col, prefix):
        """
        Calculates behavioral pulse metrics for short observation windows.

        Metrics:
        - Recency (days_since): Days since the last event.
        - Intensity: Total events per unique active day.
        - Active Late Window: Binary flag indicating activity in the last 3 days.
        """
        if df.empty: return None
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[df[date_col] < self.cutoff]

        stats = df.groupby('member_id')[date_col].max().reset_index()
        stats[f'{prefix}_days_since'] = (self.cutoff - stats[date_col]).dt.days

        activity = df.groupby('member_id').agg(
            total_events=(date_col, 'count'),
            unique_days=(date_col, lambda x: x.dt.date.nunique())
        ).reset_index()

        stats = stats.merge(activity, on='member_id')
        stats[f'{prefix}_intensity'] = stats['total_events'] / stats['unique_days']

        stats[f'{prefix}_is_active_late'] = (stats[f'{prefix}_days_since'] <= self.active_threshold_days).astype(int)

        return stats.drop(columns=[date_col])

    def _process_claims(self, df):
        """
        Aggregates medical claims into severity scores and category flags.

        - Maps ICD codes to severity levels (e.g., Cancer=3, Checkup=0).
        - Creates One-Hot Encoded flags for specific condition categories (E, J, I, etc.).
        """
        if df.empty: return None
        print("   -> Processing Claims Severity...")
        df = df.copy()
        df['icd_category'] = df['icd_code'].astype(str).str[0].str.upper()
        df['severity_score'] = df['icd_category'].map(self.severity_map).fillna(1)

        # Aggregations
        stats = df.groupby('member_id').agg(
            total_claims=('icd_code', 'count'),
            max_severity=('severity_score', 'max'),
            unique_conditions=('icd_category', 'nunique')
        ).reset_index()

        # Binary Flags (Ensure integer 0/1)
        dummies = pd.get_dummies(df['icd_category'], prefix='claim_cat', dtype=int)
        dummies['member_id'] = df['member_id']
        flags = dummies.groupby('member_id').max().reset_index()

        return stats.merge(flags, on='member_id', how='left')

    def _process_nlp(self, df):
        """
        Performs advanced NLP feature extraction on web visit logs.

        Part A: Semantic Anchor Scoring
        - Calculates Cosine Similarity between page titles and predefined anchors
          (Risk, Medical, Fitness, Lifestyle).

        Part B: Deep Latent Personas
        - Identifies the user's most visited page.
        - Extracts the LLM model embedding for that page.
        - Applies PCA to reduce the embedding (384 dim) to 3 latent dimensions (Deep Features).
        """
        if not self.use_nlp or df.empty: return None
        print("   -> Running NLP: Anchors (Explicit) + Deep Features (Implicit)...")

        # 1. Unique Pages & Embeddings (Shared Resource)
        pages = df[['title', 'description']].drop_duplicates().copy()

        # Create text for embedding
        pages['text'] = (pages['title'].fillna('') + " " + pages['description'].fillna('')).astype(str)

        # Calculate Embeddings ONCE (Expensive step)
        # page_embeddings shape: [N_pages, 384]
        page_embeddings = self.nlp_model.encode(pages['text'].tolist(), show_progress_bar=False)

        sim_matrix = cosine_similarity(page_embeddings, self.anchor_embeddings)

        # Assign Anchor Scores to Pages
        for i, name in enumerate(self.anchor_names):
            pages[f'score_{name}'] = sim_matrix[:, i]

        # Merge back to history and aggregate
        df_scored = df.merge(pages, on=['title', 'description'], how='left')
        agg_funcs = {f'score_{name}': ['mean', 'max'] for name in self.anchor_names}

        result_anchors = df_scored.groupby('member_id').agg(agg_funcs)
        result_anchors.columns = [f"{c[0]}_{c[1]}" for c in result_anchors.columns]
        result_anchors = result_anchors.reset_index()

        print("      Extracting Deep Features (PCA on most visited page)...")

        visit_counts = df.groupby(['member_id', 'title', 'description']).size().reset_index(name='visit_count')

        top_pages = visit_counts.sort_values(['member_id', 'visit_count'], ascending=[True, False]) \
            .groupby('member_id').head(1)

        pages['embedding_vector'] = list(page_embeddings)

        top_pages_with_emb = top_pages.merge(
            pages[['title', 'description', 'embedding_vector']],
            on=['title', 'description'],
            how='left'
        )

        # Stack the vectors into a matrix: [N_users, 384]
        valid_embs = top_pages_with_emb.dropna(subset=['embedding_vector'])

        if len(valid_embs) > 0:
            X_emb = np.stack(valid_embs['embedding_vector'].values)

            if self.pca is None:
                # Train Mode: Fit
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                deep_features = self.pca.fit_transform(X_emb)
            else:
                # Test Mode: Transform only
                deep_features = self.pca.transform(X_emb)

            cols = [f'deep_persona_{i}' for i in range(self.pca_components)]
            df_deep = pd.DataFrame(deep_features, columns=cols)
            df_deep['member_id'] = valid_embs['member_id'].values

            final_result = result_anchors.merge(df_deep, on='member_id', how='left')
        else:
            final_result = result_anchors

        return final_result

    def process(self, data_dict):
        """
        Main pipeline orchestrator.

        Merges App Usage, Web Visits, and Claims data into a single
        flat feature table per member. Handles missing values and column selection.
        """
        print(" Running Unified Feature Engineering...")
        base = data_dict['churn_labels'].copy()

        # 1. Tenure
        if 'signup_date' in base.columns:
            base['signup_date'] = pd.to_datetime(base['signup_date'])
            base['tenure_days'] = (self.cutoff - base['signup_date']).dt.days
            base.drop(columns=['signup_date'], inplace=True)

        # 2. App Behavior
        if 'app_usage' in data_dict:
            feats = self._analyze_behavior(data_dict['app_usage'], 'timestamp', 'app')
            if feats is not None: base = base.merge(feats, on='member_id', how='left')

        # 3. Web Behavior + NLP
        if 'web_visits' in data_dict:
            # Behavior
            web_feats = self._analyze_behavior(data_dict['web_visits'], 'timestamp', 'web')
            if web_feats is not None: base = base.merge(web_feats, on='member_id', how='left')

            # NLP
            nlp_feats = self._process_nlp(data_dict['web_visits'])
            if nlp_feats is not None: base = base.merge(nlp_feats, on='member_id', how='left')

        # 4. Claims Severity
        if 'claims' in data_dict:
            claim_feats = self._process_claims(data_dict['claims'])
            if claim_feats is not None: base = base.merge(claim_feats, on='member_id', how='left')

        # 5. Cleanup
        base = base.fillna(0)

        # Logic: If intensity is 0, days_since should be 999 (not 0)
        for c in base.columns:
            if 'days_since' in c:
                prefix = c.split('_')[0]
                int_col = f'{prefix}_intensity'
                if int_col in base.columns:
                    base.loc[base[int_col] == 0, c] = 999

        # 6. Extract Feature Columns
        ignore = ['member_id', 'churn', 'outreach', 'icd_code', 'rank', 'url', 'title']
        features = [c for c in base.columns if c not in ignore and np.issubdtype(base[c].dtype, np.number)]

        return base, features