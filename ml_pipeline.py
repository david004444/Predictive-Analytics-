"""
=============================================================================
6003CMD Dissertation Artefact
Predictive Analytics in Customer Behaviour: Anticipating Needs and
Optimising Strategies
Student: Nosakhare David Ugiagbe | ID: 12185872
=============================================================================
Full Machine Learning Pipeline:
  1. Synthetic Dataset Generation
  2. Exploratory Data Analysis (EDA)
  3. Data Preprocessing
  4. Model Training (Logistic Regression, Random Forest, XGBoost)
  5. Hyperparameter Tuning
  6. Evaluation & Comparative Analysis
  7. Feature Importance Analysis
  8. Visualisations
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

FIGURES_DIR = "/home/ubuntu/dissertation_artefact/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_telecom_dataset(n_customers=10000, random_state=42):
    """
    Generates a statistically realistic synthetic telecommunications
    customer dataset with approximately 22% churn rate.
    The dataset is designed to reflect real-world noise, non-linear
    relationships, and class imbalance typical of the industry.
    """
    rng = np.random.default_rng(random_state)
    n = n_customers

    # Demographic & Account Features
    tenure          = rng.integers(1, 73, size=n).astype(float)
    contract_type   = rng.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                  size=n, p=[0.55, 0.25, 0.20])
    payment_method  = rng.choice(
        ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
        size=n, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless_billing = rng.choice([0, 1], size=n, p=[0.41, 0.59])
    senior_citizen    = rng.choice([0, 1], size=n, p=[0.84, 0.16])

    # Service Features
    monthly_charges = (
        20 + 5 * (contract_type == 'Month-to-Month').astype(int)
        + rng.normal(0, 15, size=n)
    ).clip(18, 120)

    total_charges = (
        monthly_charges * tenure
        + rng.normal(0, 50, size=n)
    ).clip(0)

    num_products       = rng.integers(1, 7, size=n)
    tech_support       = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    online_security    = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    streaming_tv       = rng.choice([0, 1], size=n, p=[0.44, 0.56])
    internet_service   = rng.choice(['DSL', 'Fiber Optic', 'No'],
                                     size=n, p=[0.34, 0.44, 0.22])

    # Behavioural Features
    customer_service_calls = rng.integers(0, 10, size=n)
    avg_monthly_gb_download = rng.exponential(scale=15, size=n).clip(0, 100)

    # Introduce 2% missing values in monthly_charges (as described in dissertation)
    missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    monthly_charges_with_missing = monthly_charges.copy().astype(object)
    monthly_charges_with_missing[missing_idx] = np.nan

    # ── Churn Label Generation (non-linear, realistic) ──────────────────────
    # Build a latent churn score from multiple factors
    churn_score = np.zeros(n)

    # Month-to-month customers churn far more
    churn_score += 1.8 * (contract_type == 'Month-to-Month').astype(float)
    churn_score += 0.6 * (contract_type == 'One Year').astype(float)

    # Tenure: longer tenure → less likely to churn
    churn_score -= 0.04 * tenure

    # High service calls → frustration → churn
    churn_score += 0.35 * customer_service_calls
    # Non-linear threshold: >3 calls dramatically increases churn
    churn_score += 0.8 * (customer_service_calls > 3).astype(float)

    # High monthly charges relative to tenure
    charge_to_tenure_ratio = monthly_charges / (tenure + 1)
    churn_score += 0.025 * charge_to_tenure_ratio

    # No tech support or security → more likely to churn
    churn_score += 0.4 * (1 - tech_support)
    churn_score += 0.3 * (1 - online_security)

    # Electronic check payment → higher churn
    churn_score += 0.5 * (payment_method == 'Electronic Check').astype(float)

    # Senior citizens slightly more likely to churn
    churn_score += 0.3 * senior_citizen

    # Fiber optic users churn more (higher cost, more competition)
    churn_score += 0.4 * (internet_service == 'Fiber Optic').astype(float)

    # Add noise
    churn_score += rng.normal(0, 0.8, size=n)

    # Convert to probability via sigmoid (offset=4.5 tuned to yield ~22% churn)
    churn_prob = 1 / (1 + np.exp(-churn_score + 4.5))
    churn = (rng.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id':              [f'CUST{str(i).zfill(5)}' for i in range(n)],
        'tenure':                   tenure,
        'contract_type':            contract_type,
        'payment_method':           payment_method,
        'paperless_billing':        paperless_billing,
        'senior_citizen':           senior_citizen,
        'monthly_charges':          monthly_charges_with_missing,
        'total_charges':            total_charges,
        'num_products':             num_products,
        'tech_support':             tech_support,
        'online_security':          online_security,
        'streaming_tv':             streaming_tv,
        'internet_service':         internet_service,
        'customer_service_calls':   customer_service_calls,
        'avg_monthly_gb_download':  avg_monthly_gb_download,
        'churn':                    churn
    })

    return df


print("=" * 70)
print("6003CMD DISSERTATION ARTEFACT — ML PIPELINE")
print("Student: Nosakhare David Ugiagbe | ID: 12185872")
print("=" * 70)

print("\n[1/8] Generating synthetic telecommunications dataset...")
df = generate_telecom_dataset(n_customers=10000)
df.to_csv("/home/ubuntu/dissertation_artefact/telecom_dataset.csv", index=False)

print(f"      Dataset shape: {df.shape}")
print(f"      Churn rate:    {df['churn'].mean()*100:.1f}%")
print(f"      Missing values in monthly_charges: {df['monthly_charges'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/8] Generating EDA visualisations...")

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = {'Churn': '#E74C3C', 'No Churn': '#2ECC71'}

# ── Figure 1: Churn Distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
churn_counts = df['churn'].value_counts()
bars = ax.bar(['No Churn (0)', 'Churn (1)'],
              churn_counts.values,
              color=['#2ECC71', '#E74C3C'],
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{val:,}\n({val/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Figure 1: Class Distribution — Churn vs. No Churn', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Number of Customers', fontsize=11)
ax.set_ylim(0, churn_counts.max() * 1.18)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_churn_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Churn Rate by Contract Type ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
contract_churn = df.groupby('contract_type')['churn'].mean().sort_values(ascending=False) * 100
colors = ['#E74C3C', '#F39C12', '#2ECC71']
bars = ax.bar(contract_churn.index, contract_churn.values,
              color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, contract_churn.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Figure 2: Churn Rate by Contract Type', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_xlabel('Contract Type', fontsize=11)
ax.set_ylim(0, contract_churn.max() * 1.2)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_churn_by_contract.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Churn Rate by Customer Service Calls ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
calls_churn = df.groupby('customer_service_calls')['churn'].mean() * 100
ax.bar(calls_churn.index, calls_churn.values,
       color=['#E74C3C' if v > 30 else '#3498DB' for v in calls_churn.values],
       edgecolor='white', linewidth=1.2)
ax.axvline(x=3, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8, label='Threshold: >3 calls')
ax.set_title('Figure 3: Churn Rate by Number of Customer Service Calls\n(Non-linear threshold effect visible at >3 calls)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Number of Customer Service Calls', fontsize=11)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_churn_by_service_calls.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 4: Tenure Distribution by Churn ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
df[df['churn'] == 0]['tenure'].plot(kind='hist', bins=30, alpha=0.6,
                                     color='#2ECC71', label='No Churn', ax=ax, density=True)
df[df['churn'] == 1]['tenure'].plot(kind='hist', bins=30, alpha=0.6,
                                     color='#E74C3C', label='Churn', ax=ax, density=True)
ax.set_title('Figure 4: Tenure Distribution by Churn Status\n(Long-tenure customers are significantly less likely to churn)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Tenure (Months)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_tenure_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 5: Correlation Heatmap ────────────────────────────────────────────
df_numeric = df.copy()
df_numeric['monthly_charges'] = pd.to_numeric(df_numeric['monthly_charges'], errors='coerce')
# Encode categoricals for correlation
le = LabelEncoder()
for col in ['contract_type', 'payment_method', 'internet_service']:
    df_numeric[col + '_enc'] = le.fit_transform(df_numeric[col])

corr_cols = ['tenure', 'monthly_charges', 'total_charges', 'num_products',
             'tech_support', 'online_security', 'customer_service_calls',
             'avg_monthly_gb_download', 'senior_citizen', 'paperless_billing',
             'contract_type_enc', 'payment_method_enc', 'churn']
corr_labels = ['Tenure', 'Monthly\nCharges', 'Total\nCharges', 'Num\nProducts',
               'Tech\nSupport', 'Online\nSecurity', 'Service\nCalls',
               'Avg GB\nDownload', 'Senior\nCitizen', 'Paperless\nBilling',
               'Contract\nType', 'Payment\nMethod', 'Churn']

corr_matrix = df_numeric[corr_cols].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            xticklabels=corr_labels, yticklabels=corr_labels,
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Figure 5: Correlation Matrix — Feature Relationships and Churn',
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9, rotation=0)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print("      EDA figures saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/8] Preprocessing data...")

df_proc = df.copy()
df_proc['monthly_charges'] = pd.to_numeric(df_proc['monthly_charges'], errors='coerce')

# Median imputation by service tier (contract_type as proxy for service tier)
for tier in df_proc['contract_type'].unique():
    mask = (df_proc['contract_type'] == tier) & (df_proc['monthly_charges'].isna())
    median_val = df_proc.loc[df_proc['contract_type'] == tier, 'monthly_charges'].median()
    df_proc.loc[mask, 'monthly_charges'] = median_val

print(f"      Missing values after imputation: {df_proc['monthly_charges'].isna().sum()}")

# Feature Engineering
df_proc['charge_to_tenure_ratio'] = df_proc['monthly_charges'] / (df_proc['tenure'] + 1)
df_proc['high_service_calls']     = (df_proc['customer_service_calls'] > 3).astype(int)
df_proc['no_support_or_security'] = ((df_proc['tech_support'] == 0) & (df_proc['online_security'] == 0)).astype(int)

# One-hot encoding
df_proc = pd.get_dummies(df_proc,
                          columns=['contract_type', 'payment_method', 'internet_service'],
                          drop_first=False)

# Drop non-feature columns
drop_cols = ['customer_id', 'churn']
feature_cols = [c for c in df_proc.columns if c not in drop_cols]
X = df_proc[feature_cols]
y = df_proc['churn']

print(f"      Feature space: {X.shape[1]} features, {X.shape[0]} samples")
print(f"      Churn rate: {y.mean()*100:.1f}%")

# Train/Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MODEL TRAINING & HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/8] Training models with hyperparameter tuning...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Logistic Regression (L2 Ridge, C tuned via CV) ───────────────────────────
print("      Training Logistic Regression...")
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_search = RandomizedSearchCV(
    LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
    param_distributions=lr_param_grid,
    n_iter=6, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
lr_search.fit(X_train_scaled, y_train)
lr_model = lr_search.best_estimator_
print(f"      Best C: {lr_search.best_params_['C']} | CV AUC: {lr_search.best_score_:.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("      Training Random Forest...")
rf_param_dist = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [8, 10, 12, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':    ['sqrt', 'log2']
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=20, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
print(f"      Best params: {rf_search.best_params_} | CV AUC: {rf_search.best_score_:.4f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("      Training XGBoost...")
xgb_param_dist = {
    'n_estimators':  [100, 200, 300, 400],
    'max_depth':     [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample':     [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha':     [0, 0.1, 0.5, 1],
    'reg_lambda':    [1, 1.5, 2]
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_dist,
    n_iter=30, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
print(f"      Best params: {xgb_search.best_params_} | CV AUC: {xgb_search.best_score_:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/8] Evaluating models on holdout test set...")

models = {
    'Logistic Regression': (lr_model,  X_test_scaled),
    'Random Forest':       (rf_model,  X_test),
    'XGBoost':             (xgb_model, X_test),
}

results = {}
for name, (model, X_eval) in models.items():
    y_pred  = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_proba),
        'y_pred':    y_pred,
        'y_proba':   y_proba,
    }

results_df = pd.DataFrame({
    name: {k: round(v, 4) for k, v in vals.items() if k not in ['y_pred', 'y_proba']}
    for name, vals in results.items()
}).T

print("\n      ── Model Performance on Test Set ──")
print(results_df.to_string())
results_df.to_csv("/home/ubuntu/dissertation_artefact/model_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: VISUALISATIONS — EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/8] Generating evaluation visualisations...")

# ── Figure 6: ROC Curves (all three models) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = {'Logistic Regression': '#3498DB', 'Random Forest': '#F39C12', 'XGBoost': '#E74C3C'}
for name, (model, X_eval) in models.items():
    y_proba = results[name]['y_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]['AUC-ROC']
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})',
            color=colors_roc[name], linewidth=2.5)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random Classifier (AUC = 0.500)')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
ax.set_title('Figure 6: ROC Curves — Comparative Model Performance', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 7: Confusion Matrices (3-panel) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 7: Confusion Matrices — All Three Models (Test Set, n=2,000)',
             fontsize=13, fontweight='bold', y=1.02)
for ax, (name, (model, X_eval)) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    # Annotate percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.texts[i*2+j].set_text(f'{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)')
            ax.texts[i*2+j].set_fontsize(11)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 8: Model Comparison Bar Chart ─────────────────────────────────────
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metrics_to_plot))
width = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width,     [results['Logistic Regression'][m] for m in metrics_to_plot], width, label='Logistic Regression', color='#3498DB', edgecolor='white')
bars2 = ax.bar(x,             [results['Random Forest'][m]       for m in metrics_to_plot], width, label='Random Forest',       color='#F39C12', edgecolor='white')
bars3 = ax.bar(x + width,     [results['XGBoost'][m]             for m in metrics_to_plot], width, label='XGBoost',             color='#E74C3C', edgecolor='white')
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xlabel('Evaluation Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 8: Comparative Model Performance Across All Metrics', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("      Evaluation figures saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7/8] Generating feature importance visualisations...")

feature_names = list(X.columns)

# ── Figure 9: XGBoost Feature Importance (Top 15) ────────────────────────────
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
top15_xgb = xgb_importances.head(15)

fig, ax = plt.subplots(figsize=(9, 7))
colors_fi = ['#E74C3C' if i < 3 else '#3498DB' for i in range(15)]
bars = ax.barh(range(15), top15_xgb.values[::-1], color=colors_fi[::-1], edgecolor='white')
ax.set_yticks(range(15))
ax.set_yticklabels([n.replace('_', ' ').title() for n in top15_xgb.index[::-1]], fontsize=10)
ax.set_xlabel('Feature Importance Score (Gain)', fontsize=11)
ax.set_title('Figure 9: XGBoost — Top 15 Feature Importances\n(Red = Top 3 most important predictors)',
             fontsize=12, fontweight='bold', pad=15)
ax.spines[['top', 'right']].set_visible(False)
# Annotate values
for bar, val in zip(bars, top15_xgb.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_xgb_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 10: Random Forest Feature Importance (Top 15) ─────────────────────
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
top15_rf = rf_importances.head(15)

fig, ax = plt.subplots(figsize=(9, 7))
colors_rf = ['#E74C3C' if i < 3 else '#F39C12' for i in range(15)]
bars = ax.barh(range(15), top15_rf.values[::-1], color=colors_rf[::-1], edgecolor='white')
ax.set_yticks(range(15))
ax.set_yticklabels([n.replace('_', ' ').title() for n in top15_rf.index[::-1]], fontsize=10)
ax.set_xlabel('Feature Importance Score (Mean Decrease in Impurity)', fontsize=11)
ax.set_title('Figure 10: Random Forest — Top 15 Feature Importances',
             fontsize=12, fontweight='bold', pad=15)
ax.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, top15_rf.values[::-1]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_rf_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 11: Cross-Validation AUC Scores (Boxplot) ─────────────────────────
cv_scores_lr  = cross_val_score(lr_model,  X_train_scaled, y_train, cv=cv, scoring='roc_auc')
cv_scores_rf  = cross_val_score(rf_model,  X_train,        y_train, cv=cv, scoring='roc_auc')
cv_scores_xgb = cross_val_score(xgb_model, X_train,        y_train, cv=cv, scoring='roc_auc')

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([cv_scores_lr, cv_scores_rf, cv_scores_xgb],
                labels=['Logistic\nRegression', 'Random\nForest', 'XGBoost'],
                patch_artist=True, notch=False, widths=0.4)
colors_bp = ['#3498DB', '#F39C12', '#E74C3C']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_ylabel('AUC-ROC Score', fontsize=12)
ax.set_title('Figure 11: 5-Fold Cross-Validation AUC-ROC Distribution\n(Training Set)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_ylim(0.5, 1.0)
ax.spines[['top', 'right']].set_visible(False)
# Annotate means
for i, scores in enumerate([cv_scores_lr, cv_scores_rf, cv_scores_xgb], 1):
    ax.text(i, scores.min() - 0.015, f'μ={scores.mean():.3f}',
            ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig11_cv_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()

print("      Feature importance figures saved.")
print(f"\n      Top 3 XGBoost features: {list(top15_xgb.index[:3])}")
print(f"      Top 3 Random Forest features: {list(top15_rf.index[:3])}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8/8] Pipeline complete. Summary:")
print("=" * 70)
print(results_df.to_string())
print("=" * 70)
print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Dataset saved to: /home/ubuntu/dissertation_artefact/telecom_dataset.csv")
print("Results saved to: /home/ubuntu/dissertation_artefact/model_results.csv")
print("\nPipeline completed successfully.")
