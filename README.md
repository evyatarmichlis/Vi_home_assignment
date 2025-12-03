# Vi Home Assignment

## Overview

This project delivers a robust, data-driven solution to reduce member churn for WellCo.

Unlike traditional churn prediction models that simply identify who is likely to leave (Risk Prediction), this solution uses **Uplift Modeling** (T-Learner approach) to identify who can be saved (Persuadability).

By targeting members with the highest **Uplift Score** (difference in churn probability between 'outreach' and 'No outreach' scenarios)

## Key Features

- **Unified Feature Engineering**: Combines behavioral pulse checks (recency/frequency), health severity context (ICD codes), and Deep Semantic Features (NLP on web logs) into a single pipeline.

- **T-Learner Uplift Model**: Trains two independent classifiers (Control vs. Treatment) to mathematically isolate the causal impact of outreach.

- **Financial Optimization**: Uses Sensitivity Analysis to determine the optimal outreach list size (n) that maximizes net profit across multiple cost scenarios.

- **Robustness**: Optimized using Optuna with Cross-Validation to prevent overfitting on the small, noisy dataset (2-week window).

## Repository Structure

```
├── data/
│   ├── raw/                  # Place original CSV files here
│   └── processed/            # Generated plots and lists saved here
├── deliverables/
│   └── prioritized_outreach_list.csv  # FINAL OUTPUT: Top 'n' users to call
├── src/
│   ├── data_loader.py        # Robust CSV loading with schema validation
│   ├── feature_eng.py        # Pipeline: Behavior + Health + NLP + Deep Features
│   ├── model.py              # ChurnClassifier and TLearner classes
│   ├── optimization_outreach.py       # ROI Calculation and Sensitivity Analysis
│   ├── tune model.py       # use optuna to find best hyperparams
│   ├── vis.py                # Exploratory visualizations: correlations, distributions, t-SNE
│   └── evaluate_uplift.py    # Audit script for model performance
├── main.py                   # Main execution script (End-to-End)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

##  Setup & Installation

### Prerequisites
- Python 3.9+

### Clone the Repository

```bash
git clone ://github.com/evyatarmichlis/Vi_home_assignment
cd Vi_home_assignment
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Setup
Ensure the provided raw data files (e.g., `web_visits.csv`, `churn_labels.csv`) are placed inside the `data/raw/train/` and `data/raw/test/` directories.

##  How to Run

### 1. Run the End-to-End Pipeline

This single command loads data, trains the Uplift Model, optimizes the outreach strategy, and generates the final CSV.

```bash
python main.py
```

**Expected Output:**
- Console logs showing training progress and AUC scores.
- 
- `data/processed/plots/optimization_curve.png`: ROI curve showing the optimal n.
- `data/processed/plots/sensitivity_analysis.png`: diffrents optimal n for diffrent marinals costs
- `deliverables/prioritized_outreach_list.csv`: The ranked list of members to contact.


**Optinal Output:**
if you turn the vis flag into true in the main.py you can recreate varios EDA plots.


### 2. (Optional) Run Optimization Search

To re-tune hyperparameters using Optuna (takes ~5-10 mins):

```bash
python -m src.tune model
```

##  Methodological Approach

### 1. Feature Engineering (The "Unified" Engine)

We engineered three layers of features to capture the short-term nature of the data:

- **Pulse & Behavior**: Since we only have 14 days of history, we focused on Recency ("Days since last active") and Consistency ("Unique active days") rather than long-term trends.

- **Health Severity**: Claims were mapped to severity scores (e.g., Diabetes > Flu). High-severity members showed higher retention.

- **Deep Latent Personas**:Deep Latent Personas & Intent (The Core Engine): We moved beyond simple frequency counts by leveraging Natural Language Processing (NLP). We used a pre-trained Transformer Model (all-MiniLM-L6-v2)
-    to generate dense semantic embeddings for every web page visited. By applying Principal Component Analysis (PCA) to these embeddings, we compressed thousands of browsing signals into compact vector ready to use in our model
-    . We further augmented this with Semantic Anchor Scoring, which calculates the cosine similarity between a user's history and specific high-value concepts (like "Cancellation" or "Health"), allowing the model to detect subtle churn intent that traditional metrics miss.
### 2. Modeling Strategy (T-Learner)

We solved the "S-Learner Problem" (where models ignore weak intervention signals) by using a T-Learner:
Traditional churn models often fail to detect the subtle impact of outreach because stronger signals (like tenure or inactivity) dominate the prediction. To accurately measure the true causal effect of our intervention, we implemented a **T-Learner** framework, which trains two independent classifiers:
- **Model Control**: Learns natural churn risk (trained on outreach=0).
- **Model Treated**: Learns churn risk after intervention (trained on outreach=1).
- **Uplift Score**: P(Churn|Control) - P(Churn|Treated).

### 3. Determining 'n' (Sensitivity Analysis)

## Finding the Optimal Outreach List Size

The optimal list size ($n$) is the rank that maximizes Net Profit, calculated as the difference between the total expected revenue from saved customers and the total cost of contacting those customers.

This objective is achieved by maximizing the following function:

$$
\text{Optimal } n = \arg \max_{n} \left( \sum_{i=1}^{n} \left[ \text{Uplift Score}_i \times \text{Customer Value} \right] - \left[ n \times \text{Cost per Call} \right] \right)
$$

**Where:**
- $\text{Uplift Score}_i$ is the probability of saving member $i$
- $n$ is the outreach list size
- $\text{Customer Value}$ is the Lifetime Value (LTV) of a saved customer
- $n \times \text{Cost per Call}$ is the cumulative expense of the outreach



Since the outreach cost is "unknown and marginal," determining a single fixed list size is risky. Instead, we implemented a dynamic Sensitivity Analysis framework. This allows stakeholders to configure different cost assumptions
(e.g.,0%, 1%, 5%, 10% or 20% of Customer Value), and the model automatically selects the optimal outreach size (n) that maximizes profitability for that specific scenario.
we can see the results in sensitivity_analysis.png in plot file
##  Results Summary

| Metric | Score | Insight |
|--------|-------|---------|
| Baseline AUC | 0.50 | Random guessing (prior state). |
| Final AUC | 0.66 | Strong predictive power given short history. |
| Churn Recall | 63% | We successfully identify ~2/3 of all at-risk members. |
| Uplift | Positive | The model successfully separates "Persuadables" from "Lost Causes." |

### 4. Model Evaluation History (Deep Dive)

The Results Summary reflects the final outcome of multiple experiments designed to find the optimal balance between high Recall and AUC 

To see the full progression of model improvements, feature effectiveness, and why the final model was chosen:

Review Visual History: The complete trade-offs between performance metrics are captured in the comparison plots saved in the data/processed/plots/ directory:


comparison_auc.png


comparison_recall.png


comparison_precision.png

Reproduce the Comparison: You can re-run the dedicated model evaluation script to regenerate the historical comparison charts and verify the final model selection:

Bash
```bash

python src/classification_model_evaluation.py c

```
For questions regarding this analysis, please contact evyatarmich@gmail.com.
