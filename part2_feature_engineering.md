# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 2: Time Series Features & Tree-Based Models

**Objective:** Extract basic time-series features from heart rate data, train Random Forest and XGBoost models, and compare their performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Load the dataset.

```python
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data with timestamp parsed as datetime
    """
    # YOUR CODE HERE
    # Load the CSV file using pandas
    # Make sure to parse the timestamp column as datetime
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    return df  # Replace with actual implementation
```

## 3. Feature Engineering

Implement `extract_rolling_features` to calculate rolling mean and standard deviation for the `heart_rate`.

```python
def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
    # YOUR CODE HERE
    # 1. Sort data by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # 2. Set timestamp as index (this allows time-based operations)
    df_indexed = df_sorted.set_index('timestamp')

    # Sort by patient_id and timestamp
    df = df.sort_values(['patient_id', 'timestamp'])

    # Define a function to apply rolling stats within each patient
    def rolling_stats(group):
        group = group.set_index('timestamp')
        rolling = group['heart_rate'].rolling(f'{window_size_seconds}s')
        group['hr_rolling_mean'] = rolling.mean()
        group['hr_rolling_std'] = rolling.std()
        group['rolling_count'] = rolling.count()
        return group.reset_index()

    # Apply the rolling stats per patient
    df_rolled = df.groupby('patient_id', group_keys=False).apply(rolling_stats)

    # Optional: fill missing values created by rolling (at beginning of each group)
    df_rolled[['hr_rolling_mean', 'hr_rolling_std']] = df_rolled[
        ['hr_rolling_mean', 'hr_rolling_std']
    ].fillna(method='bfill')  # or method='ffill', or df.dropna()

    return df_rolled
```

## 4. Data Preparation

Implement `prepare_data_part2` using the newly engineered features.

```python
def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Select relevant features including the rolling features
    # 2. Select target variable (disease_outcome)
    # Define features and outcome
    # In theory hr, hr_rolling_mean, and hr_rolling_std have a significant amount of interaction between them... but just including anyway
    feature_cols = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi', 'heart_rate', 'hr_rolling_mean', 'hr_rolling_std']
    target_col = 'disease_outcome'

    X = df_with_features[feature_cols]
    y = df_with_features[target_col]
    
    # 3. Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 4. Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train = pd.DataFrame(X_train_imputed, columns=feature_cols, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=feature_cols, index=X_test.index)
    
    # Placeholder return - replace with your implementation
    return X_train, X_test, y_train, y_test
```

## 5. Random Forest Model

Implement `train_random_forest`.

```python
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Fit the model to training data
    rf_model.fit(X_train, y_train)

    return rf_model
```

## 6. XGBoost Model

Implement `train_xgboost`.

```python
def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    # YOUR CODE HERE
    # Initialize and train an XGBClassifier
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state, 
        eval_metric='logloss'    
    )
    
    # Fit the model
    xgb_model.fit(X_train, y_train)

    return xgb_model
```

## 7. Model Comparison

Calculate and compare AUC scores for both models.

```python
# YOUR CODE HERE
def compare_auc(rf_model, xgb_model):
    # 1. Generate probability predictions for both models
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    # 2. Calculate AUC scores
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    # 3. Compare the performance
    print(f"Random forest: {rf_auc}. XGBoost: {xgb_auc}.")
    return
```

## 8. Save Results

Save the AUC scores to a text file.

```python
def output_results2(rf_auc, xgb_auc):
    os.makedirs('results', exist_ok=True)
    with open('results/results_part2.txt', 'w') as f:
        f.write(f"Random forest: {rf_auc}. XGBoost: {xgb_auc}.")
    return
```

## 9. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Calculate AUC scores
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # 6. Save results
    output_results2(rf_auc, xgb_auc)