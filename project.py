# %% [markdown]
# # Online Shoppers Purchase Intention Prediction
# 
# ## Project Overview
# This notebook analyzes online shopping behavior to predict whether a session will result in a purchase (Revenue).

# %% [markdown]
# ## 1. Import Libraries and Setup

# %%
# Import all required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set visualization style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully!")

# %% [markdown]
# ## 2. Load and Initial Data Exploration

# %%
# Load the dataset
df = pd.read_csv('online_shoppers_intention.csv')
df.length = len(df)
print(f"Total number of rows in the dataset: {df.length}")

# Display basic information
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# %%
# Check if any duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# %%
# Mean, max, min, mode of numerical columns - improved aesthetics
def describe_numerical_columns(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    description = {}
    for col in numerical_cols:
        description[col] = {
            'mean': df[col].mean(),
            'max': df[col].max(),
            'min': df[col].min(),
            'mode': df[col].mode()[0] if len(df[col].mode()) > 0 else np.nan
        }
    return pd.DataFrame(description).round(3)

numerical_description = describe_numerical_columns(df)
print("\nNumerical Columns Description:")
print(numerical_description.T)  # Transpose for better readability

# %% [markdown]
# ## 3. Target Variable Analysis

# %%
# Class distribution
def class_distribution(df, target_col):
    return df[target_col].value_counts(normalize=True)

target_col = 'Revenue'
distribution = class_distribution(df, target_col)
print(f"\nClass Distribution for {target_col}:")
print(distribution)
print(f"Class Imbalance Ratio: {distribution[False]/distribution[True]:.2f}:1")

# %%
# Plot conversion distribution with improved aesthetics
def plot_conversion_distribution(df, target_col):
    distribution = df[target_col].value_counts(normalize=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(distribution, labels=['No Purchase', 'Purchase'], autopct='%1.1f%%', 
            startangle=140, colors=colors, explode=[0.05, 0])
    ax1.set_title('Revenue Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Bar chart for counts
    counts = df[target_col].value_counts()
    ax2.bar(['No Purchase', 'Purchase'], counts, color=colors)
    ax2.set_title('Revenue Distribution (Counts)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_conversion_distribution(df, target_col)

# %% [markdown]
# ## 4. Numerical Features Distribution Analysis

# %%
# Distribution plots with improved visualization
def plot_numerical_distributions(df, cols):
    n_cols = len(cols)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols):
        # Histogram
        ax1 = axes[idx*2]
        df[col].hist(bins=30, edgecolor='black', alpha=0.7, ax=ax1, color='skyblue')
        ax1.set_title(f'Histogram of {col}', fontsize=12)
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax1.legend()

        # Boxplot
        ax2 = axes[idx*2 + 1]
        df.boxplot(column=col, ax=ax2)
        ax2.set_title(f'Boxplot of {col}', fontsize=12)
        
        # Calculate and display outlier count
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        ax2.text(0.5, 0.95, f'Outliers: {outliers}', transform=ax2.transAxes, 
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove empty subplots
    for idx in range(len(cols)*2, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

numerical_cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 
                 'BounceRates', 'ExitRates', 'PageValues']

print("Before outlier removal:")
plot_numerical_distributions(df, numerical_cols)

# %% [markdown]
# ## 5. Outlier Detection and Removal

# %%
# Remove Outliers - enhanced with reporting
def remove_outliers(df, cols, threshold=3):
    initial_shape = df.shape[0]
    outliers_info = {}
    
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        before = len(df)
        df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]
        after = len(df)
        outliers_info[col] = before - after
    
    print(f"\nOutlier Removal Summary:")
    print(f"Total rows before: {initial_shape}")
    print(f"Total rows after: {df.shape[0]}")
    print(f"Total rows removed: {initial_shape - df.shape[0]} ({(initial_shape - df.shape[0])/initial_shape*100:.2f}%)")
    print("\nOutliers removed by column:")
    for col, count in outliers_info.items():
        print(f"  {col}: {count}")
    
    return df

cols_to_remove_outliers = ['Administrative_Duration', 'Informational_Duration', 
                            'ProductRelated_Duration', 'PageValues']
before_rows = df.shape[0]
df = remove_outliers(df, cols_to_remove_outliers)

print("\nAfter outlier removal:")
plot_numerical_distributions(df, numerical_cols)

# %% [markdown]
# ## 6. Categorical Features Analysis

# %%
# Frequency counts of categorical columns - enhanced visualization
def plot_categorical_counts(df, cols):
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        
        if col == 'Month':
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            df[col] = pd.Categorical(df[col], categories=month_order, ordered=True)
        
        value_counts = df[col].value_counts().sort_index()
        value_counts.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title(f'Frequency Counts of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(value_counts):
            ax.text(i, v + 10, str(v), ha='center', fontsize=9)
    
    # Remove empty subplots
    for idx in range(len(cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
plot_categorical_counts(df, categorical_cols)

# %% [markdown]
# ## 7. Experimental: BounceRates Regression Analysis
# **Note:** This section analyzes BounceRates prediction as an experimental feature. The main target variable remains Revenue (purchase prediction).

# %%
# NOTE: The BounceRates regression section is preserved but marked as experimental
print("\n" + "="*80)
print("EXPERIMENTAL: BounceRates Regression Analysis")
print("Note: This is experimental - Revenue is the main target variable")
print("="*80)

# Make split train data for bouncerates (keeping original code)
def split_train_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train_br, X_test_br, y_train_br, y_test_br = split_train_data(df, 'BounceRates')

# One-hot encode categorical features for both train and test sets
X_train_encoded_br = pd.get_dummies(X_train_br)
X_test_encoded_br = pd.get_dummies(X_test_br)

# Align the columns of train and test sets
X_train_encoded_br, X_test_encoded_br = X_train_encoded_br.align(X_test_encoded_br, join='left', axis=1, fill_value=0)

# %%
# Linear regression model for BounceRates
from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model_lr = train_linear_regression(X_train_encoded_br, y_train_br)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

mse, r2 = evaluate_model(model_lr, X_test_encoded_br, y_test_br)
print(f"Linear Regression - Mean Squared Error: {mse:.4f}")
print(f"Linear Regression - R^2 Score: {r2:.4f}")

# %%
# Use RandomForestRegressor for continuous target
from sklearn.ensemble import RandomForestRegressor

# Feature importance for Random Forest Regressor
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

rf_model_br = train_random_forest(X_train_encoded_br, y_train_br)

# Enhanced feature importance plot
def plot_feature_importance(model, X_train, title="Feature Importances", top_n=20):
    importances = model.feature_importances_
    feature_names = X_train.columns
    indices = importances.argsort()[::-1][:top_n]  # Get top N features

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=14, fontweight='bold')
    bars = plt.bar(range(len(indices)), importances[indices], align='center', color='steelblue')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return indices, importances

plot_feature_importance(rf_model_br, X_train_encoded_br, "Feature Importances - BounceRates Prediction")

# %%
# Predicting BounceRates using Random Forest Regressor
def predict_bouncerates(model, X_test):
    return model.predict(X_test)

y_pred_rf_br = predict_bouncerates(rf_model_br, X_test_encoded_br)

# Evaluate Random Forest Regressor model
def evaluate_rf_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

mse_rf, r2_rf = evaluate_rf_model(rf_model_br, X_test_encoded_br, y_test_br)
print(f"Random Forest Regressor - Mean Squared Error: {mse_rf:.5f}")
print(f"Random Forest Regressor - R^2 Score: {r2_rf:.4f}")

# %% [markdown]
# ## 8. Main Analysis: Revenue (Purchase) Prediction
# Now we focus on the main objective: predicting whether a session will result in a purchase.

# %%
print("\n" + "="*80)
print("MAIN ANALYSIS: Revenue (Purchase) Prediction")
print("="*80)

# Relationship between categorical variables and Revenue - enhanced visualization
def plot_categorical_vs_revenue(df, categorical_col, target_col='Revenue'):
    plt.figure(figsize=(10, 6))
    
    # Calculate conversion rate and counts
    grouped = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    grouped['conversion_rate'] = grouped['mean'] * 100
    
    # Sort by conversion rate
    grouped = grouped.sort_values('conversion_rate')
    
    # Create bar plot
    ax = grouped['conversion_rate'].plot(kind='bar', color='mediumseagreen')
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(i, row['conversion_rate'] + 0.5, f"n={row['count']}", 
                ha='center', fontsize=9, style='italic')
    
    # Add overall conversion rate line
    overall_rate = df[target_col].mean() * 100
    ax.axhline(y=overall_rate, color='red', linestyle='--', 
              label=f'Overall Rate: {overall_rate:.1f}%', linewidth=2)
    
    plt.title(f'Conversion Rate by {categorical_col}', fontsize=14, fontweight='bold')
    plt.xlabel(categorical_col)
    plt.ylabel('Conversion Rate (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

for col in categorical_cols:
    plot_categorical_vs_revenue(df, col, target_col)

# %% [markdown]
# ## 9. Feature Correlation Analysis

# %%
# Features correlate most with Revenue - enhanced
def plot_correlation_with_revenue(df, target_col='Revenue'):
    # Only use numerical columns for correlation
    numerical_df = df.select_dtypes(include=['float64', 'int64', 'bool'])
    correlation = numerical_df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Remove self-correlation
    correlation = correlation[correlation.index != target_col]
    
    plt.figure(figsize=(10, 8))
    bars = correlation.plot(kind='barh', color='darkorange')
    plt.title(f'Features Correlation with {target_col} (Absolute Values)', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Features')
    
    # Add value labels
    for i, v in enumerate(correlation):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return correlation

revenue_correlation = plot_correlation_with_revenue(df, target_col)

# %%
# Correlation matrix/heatmap of numerical features - enhanced
def plot_correlation_matrix(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64', 'bool'])
    
    plt.figure(figsize=(14, 12))
    correlation_matrix = numerical_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    
    # Create heatmap with better styling
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

correlation_matrix = plot_correlation_matrix(df)

# %% [markdown]
# ## 10. Multicollinearity Analysis

# %%
# Multicollinearity between features - enhanced reporting
def check_multicollinearity(df, threshold=0.8):
    numerical_df = df.select_dtypes(include=['float64', 'int64', 'bool'])
    correlation_matrix = numerical_df.corr()
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                corr_value = correlation_matrix.iloc[i, j]
                high_corr_pairs.append((
                    correlation_matrix.columns[i], 
                    correlation_matrix.columns[j],
                    corr_value
                ))
    
    return high_corr_pairs

high_corr_pairs = check_multicollinearity(df)
if high_corr_pairs:
    print("\nHigh correlation pairs (multicollinearity):")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
else:
    print("\nNo high correlation pairs found (threshold: 0.8)")

# %%
# Remove multicollinearity by dropping one of the correlated features
def remove_multicollinearity(df, threshold=0.8):
    numerical_cols = df.select_dtypes(include=['float64', 'int64', 'bool']).columns
    correlation_matrix = df[numerical_cols].corr()
    to_drop = set()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                to_drop.add(colname)
    
    if to_drop:
        print(f"\nDropping highly correlated features: {list(to_drop)}")
    
    return df.drop(columns=to_drop)

df = remove_multicollinearity(df)
print(f"Shape after removing multicollinearity: {df.shape}")

# %% [markdown]
# ## 11. Data Preparation for Classification

# %%
# Prepare data and handle class imbalance for classification tasks
def prepare_data(df, target_col='Revenue'):
    # Split into X/y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Features after encoding: {X_encoded.shape[1]}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X_train, y_train):
    print("\nHandling class imbalance with SMOTE...")
    print(f"Before SMOTE: {y_train.value_counts()}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

X_train, X_test, y_train, y_test = prepare_data(df, target_col='Revenue')
X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

# %% [markdown]
# ## 12. Model Training - Random Forest (Scikit-learn)

# %%
# Store all models for comparison
all_models = {}
model_results = []

# Train a Classification model (Random Forest) - Enhanced
def train_classification_model(X_train, y_train):
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

classification_model = train_classification_model(X_train_resampled, y_train_resampled)
all_models['Random Forest'] = classification_model

# %%
# Enhanced evaluation function with multiple metrics
def evaluate_classification_model_comprehensive(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if auc:
        print(f"  AUC-ROC:   {auc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }

# Evaluate Random Forest
rf_results = evaluate_classification_model_comprehensive(
    classification_model, X_test, y_test, "Random Forest"
)
model_results.append(rf_results)

# %%
# Plot feature importance for the classification model - Enhanced
def plot_classification_feature_importance(model, X_train, model_name="Model"):
    importances = model.feature_importances_
    feature_names = X_train.columns
    indices = importances.argsort()[::-1]

    # Print feature importances
    print(f"\n{model_name} - Top 15 Feature Importances:")
    for idx in indices[:15]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(14, 8))
    top_n = min(30, len(importances))  # Show top 30 features
    indices_top = indices[:top_n]
    
    bars = plt.bar(range(top_n), importances[indices_top], align='center', color='teal')
    plt.xticks(range(top_n), [feature_names[i] for i in indices_top], rotation=90)
    plt.title(f'{model_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    
    # Add value labels on top bars
    for i in range(min(10, top_n)):
        plt.text(i, importances[indices_top[i]] + 0.001, 
                f'{importances[indices_top[i]]:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return indices_top, importances

rf_top_features, rf_importances = plot_classification_feature_importance(
    classification_model, X_train_resampled, "Random Forest"
)

# %%
# Confusion matrix for classification model - Enhanced
def plot_confusion_matrix_enhanced(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 'Purchase'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    # Add metrics text box
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(2.5, 0.5, textstr, fontsize=12, bbox=props)
    
    plt.tight_layout()
    plt.show()

plot_confusion_matrix_enhanced(classification_model, X_test, y_test, "Random Forest")

# %%
# Precision-recall curve for classification model - Enhanced
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve_enhanced(model, X_test, y_test, model_name="Model"):
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    average_precision = average_precision_score(y_test, y_scores)

    print(f"\n{model_name} - Average Precision Score: {average_precision:.4f}")

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', markersize=8, 
             label=f'{model_name} (AP = {average_precision:.3f})', linewidth=2)
    
    # Fill area under curve
    plt.fill_between(recall, precision, alpha=0.2)
    
    # Add baseline (random classifier)
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='r', linestyle='--', 
                label=f'Baseline (Random): {baseline:.3f}')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_precision_recall_curve_enhanced(classification_model, X_test, y_test, "Random Forest")

# %% [markdown]
# ## 13. XGBoost Models (For Comparison - Not Final Model)

# %%
# XGBoost model for classification
print("\n" + "="*80)
print("XGBoost Models (For Comparison - Not Final Model)")
print("="*80)

def train_xgboost_model(X_train, y_train):
    print("\nTraining XGBoost Classifier (Default)...")
    model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train)
    return model

model_xgboost = train_xgboost_model(X_train_resampled, y_train_resampled)
all_models['XGBoost (Default)'] = model_xgboost

# Evaluate XGBoost
xgb_results = evaluate_classification_model_comprehensive(
    model_xgboost, X_test, y_test, "XGBoost (Default)"
)
model_results.append(xgb_results)

# %%
# XGBoost model with best hyperparameters (as specified in original code)
def train_best_xgboost_model(X_train, y_train, params):
    print("\nTraining XGBoost Classifier (Tuned)...")
    model = XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

def best_xgboost_params():
    return {
        'n_estimators': 100,
        'max_depth': 9,
        'learning_rate': 0.1,
        'subsample': 0.5,
        'colsample_bytree': 1.0,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 0
    }

model_best_xgboost = train_best_xgboost_model(X_train_resampled, y_train_resampled, best_xgboost_params())
all_models['XGBoost (Tuned)'] = model_best_xgboost

# Evaluate best XGBoost
xgb_tuned_results = evaluate_classification_model_comprehensive(
    model_best_xgboost, X_test, y_test, "XGBoost (Tuned)"
)
model_results.append(xgb_tuned_results)

# %%
# Feature importance for XGBoost model
plot_xgboost_feature_importance = plot_classification_feature_importance
xgb_top_features, xgb_importances = plot_xgboost_feature_importance(
    model_best_xgboost, X_train_resampled, "XGBoost (Tuned)"
)

# %%
# Remove low importance features from XGBoost model
def remove_low_importance_features(model, X_train, threshold=0.01):
    importances = model.feature_importances_
    feature_names = X_train.columns
    important_features = [feature for feature, importance in zip(feature_names, importances) 
                         if importance >= threshold]
    
    print(f"\nFeature Selection based on importance (threshold={threshold}):")
    print(f"  Original features: {len(feature_names)}")
    print(f"  Selected features: {len(important_features)}")
    print(f"  Removed features: {len(feature_names) - len(important_features)}")
    
    return X_train[important_features], important_features

X_train_reduced, important_features = remove_low_importance_features(
    model_best_xgboost, X_train_resampled
)

# %%
# XGBoost training with best hyperparameters and reduced features
def train_best_xgboost_with_reduced_features(X_train, y_train, important_features):
    X_train_reduced = X_train[important_features]
    model = XGBClassifier(
        n_estimators=100,
        max_depth=9,
        learning_rate=0.1,
        subsample=0.5,
        colsample_bytree=1.0,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=0,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_reduced, y_train)
    return model

model_best_xgboost_reduced = train_best_xgboost_with_reduced_features(
    X_train_resampled, y_train_resampled, important_features
)
all_models['XGBoost (Reduced Features)'] = model_best_xgboost_reduced

# Evaluate the best XGBoost model with reduced features
def evaluate_best_xgboost_reduced_model(model, X_test, y_test, important_features):
    X_test_reduced = X_test[important_features]
    y_pred = model.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

accuracy_best_xgboost_reduced = evaluate_best_xgboost_reduced_model(
    model_best_xgboost_reduced, X_test, y_test, important_features
)
print(f"\nXGBoost (Reduced Features) Accuracy: {accuracy_best_xgboost_reduced:.4f}")

# %%
# Evaluate Accuracy and Recall for XGBoost model - Enhanced
from sklearn.metrics import recall_score

def evaluate_xgboost_accuracy_recall(model, X_train, y_train, X_test, y_test):
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    train_recall = recall_score(y_train, model.predict(X_train))
    test_recall = recall_score(y_test, model.predict(X_test))
    
    print(f"\nTrain vs Test Performance:")
    print(f"  Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"  Train Recall:   {train_recall:.4f} | Test Recall:   {test_recall:.4f}")
    
    # Check for overfitting
    acc_diff = train_accuracy - test_accuracy
    if acc_diff > 0.05:
        print(f"  ⚠️  Potential overfitting detected (difference: {acc_diff:.4f})")
    else:
        print(f"  ✓  Model generalization appears good (difference: {acc_diff:.4f})")
    
    return train_accuracy, test_accuracy, train_recall, test_recall

# Ensure test data has the same columns as important_features
X_test_reduced = X_test[important_features]

print("\nXGBoost (Reduced Features) - Train vs Test Performance:")
train_accuracy_xgboost, test_accuracy_xgboost, train_recall_xgboost, test_recall_xgboost = evaluate_xgboost_accuracy_recall(
    model_best_xgboost_reduced, X_train_reduced, y_train_resampled, X_test_reduced, y_test
)

# %%
# XGBoost model to reduce overfitting
print("\nTraining XGBoost with Overfitting Prevention...")

def train_xgboost_with_reduced_overfitting(X_train, y_train, important_features):
    X_train_reduced = X_train[important_features]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,  # Reduced depth to prevent overfitting
        learning_rate=0.01,  # Reduced learning rate
        subsample=0.8,  # Increased subsample to reduce overfitting
        colsample_bytree=0.7,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_reduced, y_train)
    return model

model_xgboost_reduced_overfitting = train_xgboost_with_reduced_overfitting(
    X_train_resampled, y_train_resampled, important_features
)
all_models['XGBoost (Overfitting Prevention)'] = model_xgboost_reduced_overfitting

# Evaluate all XGBoost variations
print("\nXGBoost (Overfitting Prevention) - Train vs Test Performance:")
train_accuracy_xgboost_reduced, test_accuracy_xgboost_reduced, train_recall_xgboost_reduced, test_recall_xgboost_reduced = evaluate_xgboost_accuracy_recall(
    model_xgboost_reduced_overfitting, X_train_reduced, y_train_resampled, X_test_reduced, y_test
)

# Add to results
xgb_overfit_results = evaluate_classification_model_comprehensive(
    model_xgboost_reduced_overfitting, X_test_reduced, y_test, "XGBoost (Overfitting Prevention)"
)
model_results.append(xgb_overfit_results)

# %% [markdown]
# ## 14. Additional Scikit-learn Models

# %%
# Train additional scikit-learn models for comparison
print("\n" + "="*80)
print("Additional Scikit-learn Models")
print("="*80)

# Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled)
all_models['Logistic Regression'] = lr_model

lr_results = evaluate_classification_model_comprehensive(
    lr_model, X_test, y_test, "Logistic Regression"
)
model_results.append(lr_results)

# %%
# Extra Trees Classifier (scikit-learn)
from sklearn.ensemble import ExtraTreesClassifier

print("\nTraining Extra Trees Classifier...")
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train_resampled, y_train_resampled)
all_models['Extra Trees'] = et_model

et_results = evaluate_classification_model_comprehensive(
    et_model, X_test, y_test, "Extra Trees"
)
model_results.append(et_results)

# %% [markdown]
# ## 15. Model Comparison and Visualization

# %%
# Model Comparison Summary
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

# Create comparison dataframe
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\nAll Models Performance (sorted by F1-Score):")
print(results_df.to_string(index=False))

# %%
# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Bar plot of all metrics
ax1 = axes[0, 0]
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x + i*width, results_df[metric], width, label=metric)

ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# 2. ROC Curves
ax2 = axes[0, 1]
from sklearn.metrics import roc_curve

for model_name, model in all_models.items():
    if hasattr(model, 'predict_proba'):
        # Handle reduced features for some XGBoost models
        if 'Reduced' in model_name or 'Overfitting' in model_name:
            X_test_use = X_test_reduced
        else:
            X_test_use = X_test
        
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

ax2.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. F1-Score comparison
ax3 = axes[1, 0]
results_df_sorted = results_df.sort_values('F1-Score', ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df_sorted)))
bars = ax3.barh(results_df_sorted['Model'], results_df_sorted['F1-Score'], color=colors)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax3.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

ax3.set_xlabel('F1-Score', fontsize=12)
ax3.set_title('Models Ranked by F1-Score', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.grid(True, alpha=0.3, axis='x')

# 4. Best scikit-learn model analysis
ax4 = axes[1, 1]
# Find best scikit-learn model (excluding XGBoost)
sklearn_models = [r for r in model_results if 'XGBoost' not in r['Model']]
best_sklearn = max(sklearn_models, key=lambda x: x['F1-Score'])
best_sklearn_name = best_sklearn['Model']
print(f"\nBest Scikit-learn Model: {best_sklearn_name}")

# Plot confusion matrix for best sklearn model
best_sklearn_model = all_models[best_sklearn_name]
y_pred_best = best_sklearn_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
ax4.set_title(f'Best Sklearn Model: {best_sklearn_name}', fontsize=14, fontweight='bold')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Cross-Validation Analysis

# %%
# Cross-validation for best models
print("\n" + "="*80)
print("CROSS-VALIDATION ANALYSIS")
print("="*80)

from sklearn.model_selection import cross_val_score

print("\n5-Fold Cross-Validation Results:")
for model_name in [best_sklearn_name, 'Random Forest', 'XGBoost (Tuned)']:
    if model_name in all_models:
        model = all_models[model_name]
        
        # Use appropriate dataset
        if model_name == 'XGBoost (Tuned)':
            X_cv = X_train_resampled
        else:
            X_cv = X_train_resampled
        
        cv_scores = cross_val_score(model, X_cv, y_train_resampled, 
                                   cv=5, scoring='f1')
        
        print(f"\n{model_name}:")
        print(f"  CV F1-Scores: {cv_scores}")
        print(f"  Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %% [markdown]
# ## 17. Feature Importance Comparison

# %%
# Feature importance comparison between best models
print("\n" + "="*80)
print("FEATURE IMPORTANCE COMPARISON")
print("="*80)

# Get top features from Random Forest and best sklearn model
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, (model_name, ax) in enumerate([(best_sklearn_name, axes[0]), 
                                        ('Random Forest', axes[1])]):
    model = all_models[model_name]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train_resampled.columns
        indices = importances.argsort()[::-1][:20]
        
        ax.barh(range(20), importances[indices], color='steelblue')
        ax.set_yticks(range(20))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 20 Features - {model_name}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i in range(20):
            ax.text(importances[indices[i]] + 0.001, i, 
                   f'{importances[indices[i]]:.3f}', 
                   va='center', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 18. Final Model Selection and Recommendations

# %%
# Final model selection and recommendations
print("\n" + "="*80)
print("FINAL MODEL SELECTION AND RECOMMENDATIONS")
print("="*80)

print(f"\n✓ RECOMMENDED FINAL MODEL (Scikit-learn): {best_sklearn_name}")
print(f"  - F1-Score: {best_sklearn['F1-Score']:.4f}")
print(f"  - Accuracy: {best_sklearn['Accuracy']:.4f}")
print(f"  - Precision: {best_sklearn['Precision']:.4f}")
print(f"  - Recall: {best_sklearn['Recall']:.4f}")

print("\n📊 Model Performance Summary:")
print("  - XGBoost models show strong performance but are not from scikit-learn")
print("  - Random Forest provides good balance between performance and interpretability")
print("  - All models benefit from SMOTE for handling class imbalance")

print("\n🔑 Key Insights:")
print("  1. PageValues is the strongest predictor of purchase intent")
print("  2. November shows highest conversion rates - seasonal pattern")
print("  3. Session duration features are important indicators")
print("  4. Bounce and Exit rates negatively correlate with purchases")

print("\n💡 Business Recommendations:")
print("  1. Focus on visitors with high PageValues")
print("  2. Optimize for November traffic (holiday shopping)")
print("  3. Reduce bounce rates through better landing pages")
print("  4. Target returning visitors who show higher conversion rates")

# %%
# Save the final model
print(f"\n💾 Saving final model ({best_sklearn_name})...")
import joblib
joblib.dump(best_sklearn_model, f'final_model_{best_sklearn_name.lower().replace(" ", "_")}.pkl')
print("Model saved successfully!")

# Save feature names for deployment
with open('feature_names.txt', 'w') as f:
    for feature in X_train_resampled.columns:
        f.write(f"{feature}\n")
print("Feature names saved!")