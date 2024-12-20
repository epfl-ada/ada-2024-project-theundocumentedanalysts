import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint


# -----------------------
# Data Loading & Cleaning
# -----------------------
def load_tsv_data(filepath: str, col_names: list, sep: str='\t') -> pd.DataFrame:
    """Load a TSV file into a DataFrame with given column names."""
    return pd.read_csv(filepath, sep=sep, header=None, names=col_names)

def load_excel_data(filepath: str) -> pd.DataFrame:
    """Load Excel data into a DataFrame."""
    return pd.read_excel(filepath)

def filter_by_date(df: pd.DataFrame, date_col: str, cutoff_date: str) -> pd.DataFrame:
    """Filter rows by a cutoff date."""
    if date_col not in df.columns:
        raise ValueError(f"Column {date_col} not found in DataFrame.")
    return df[df[date_col] <= cutoff_date]

def filter_by_value(df: pd.DataFrame, col: str, min_value: float) -> pd.DataFrame:
    """Filter rows where a column's value is >= min_value."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame.")
    return df[df[col] >= min_value]

def drop_missing_threshold(df: pd.DataFrame, threshold_ratio: float=0.6) -> pd.DataFrame:
    """Drop columns with more than 40% missing values."""
    return df.dropna(thresh=threshold_ratio*len(df), axis=1)

def fill_missing_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median."""
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df


# -----------------------
# Feature Engineering
# -----------------------
def clean_column_strings(column_str: str) -> list:
    """Extract categories (like genres) using a regex pattern."""
    if pd.isna(column_str):
        return []
    pattern = r':\s*["\']([^"\']+)["\']'
    return re.findall(pattern, column_str)

def one_hot_encode_multilabel(df: pd.DataFrame, source_col: str, min_count: int=20) -> pd.DataFrame:
    """
    One-hot encode a MultiLabel column using MultiLabelBinarizer.
    Keep only columns with at least min_count occurrences.
    """
    if source_col not in df.columns:
        raise ValueError(f"Column {source_col} not found in DataFrame.")
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df[source_col])
    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_, index=df.index)
    encoded_df = encoded_df.loc[:, encoded_df.sum() >= min_count]
    return encoded_df

def convert_to_binary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a boolean column into binary (1/0)."""
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if x else 0)
    return df

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns."""
    return df.select_dtypes(include=[np.number])

def remove_zero_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance."""
    return df.loc[:, df.var() != 0]

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with only one unique value."""
    return df.loc[:, df.nunique() > 1]

def remove_outliers_iqr(df: pd.DataFrame, target_col: str, threshold: float=1.5) -> pd.DataFrame:
    """Remove outliers using IQR method for the target column."""
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col} not found in DataFrame.")
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

def scale_features(X):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler


# -----------------------
# Modeling
# -----------------------
def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, random_state: int=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_random_search(X_train: pd.DataFrame, y_train: pd.Series, param_distributions: dict, n_iter: int=50, cv: int=5, random_state: int=42, n_jobs: int=-1):
    """Run RandomizedSearchCV on an XGBRegressor to find best hyperparameters."""
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=random_state, tree_method='hist', n_jobs=n_jobs)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=n_jobs
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate model performance using MAE and MAPE."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mae, mape, y_pred

def cross_validate_model(model: XGBRegressor, X_train: pd.DataFrame, y_train: pd.Series, cv: int=5, random_state: int=42):
    """Cross-validate the model and return mean and std of MAE."""
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    mean_cv_mae = -np.mean(cv_scores)
    std_cv_mae = np.std(cv_scores)
    return mean_cv_mae, std_cv_mae, -cv_scores

def save_model_and_scaler(model: XGBRegressor, feature_columns: list, scaler: StandardScaler, model_path: str='best_xgb_model.pkl', columns_path: str='feature_columns.json', scaler_path: str='scaler.pkl'):
    """Save model, feature columns, and scaler to disk."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(columns_path, 'w') as f:
        json.dump(feature_columns, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


# -----------------------
# Plotting
# -----------------------
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray, title: str='Predicted vs Actual'):
    """Plot predicted vs. actual values with reference line y = x."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title(title)
    plt.axline((0, 0), slope=1, color='r', linestyle='--')
    plt.show()

def plot_residuals(y_pred: np.ndarray, residuals: np.ndarray, title: str='Residual Plot'):
    """Plot residuals against predicted values with a reference line at 0."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()

def plot_residual_distribution(residuals: np.ndarray, title: str='Distribution of Residuals', bins: int=30):
    """Plot a histogram and KDE of residuals."""
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=bins)
    plt.xlabel('Residuals')
    plt.title(title)
    plt.show()

def plot_qq(residuals: np.ndarray, title: str='Normal Q-Q Plot for Residuals'):
    """Plot Q-Q plot for residual normality check."""
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)
    plt.show()

def remove_invalid_chars(df: pd.DataFrame, invalid_chars: list):
    """Remove invalid characters from column names."""
    for char in invalid_chars:
        df.columns = df.columns.str.replace(char, '', regex=False)
    return df



