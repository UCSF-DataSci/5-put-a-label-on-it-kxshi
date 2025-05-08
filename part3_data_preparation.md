# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 3: Practical Data Preparation

**Objective:** Handle categorical features using One-Hot Encoding and address class imbalance using SMOTE.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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
        DataFrame containing the data
    """
    # YOUR CODE HERE
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    return df 
```

## 3. Categorical Feature Encoding

Implement `encode_categorical_features` using `OneHotEncoder`.

```python
def encode_categorical_features(df, column_to_encode='smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.
    
    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode
        
    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    # YOUR CODE HERE
    # 1. Extract the categorical column
    col_interest = df[[column_to_encode]]
    # 2. Apply OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(df[[column_to_encode]])
    # 3. Create new column names
    encoded_cols = encoder.get_feature_names_out([column_to_encode])
    # 4. Replace the original categorical column with the encoded columns
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df_encoded = df.drop(columns=[column_to_encode])
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded
```

## 4. Data Preparation

Implement `prepare_data_part3` to handle the train/test split correctly.

```python
def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. One-hot encode categorical columns (you can adjust the list as needed)
    df_encoded = encode_categorical_features(df, column_to_encode='smoker_status')

    # 2. Select features and target
    feature_cols = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi'] + \
                   [col for col in df_encoded.columns if col.startswith('smoker_status_')]
    target_col = 'disease_outcome'
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    # 3. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # 4. Impute missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test
```

## 5. Handling Imbalanced Data

Implement `apply_smote` to oversample the minority class.

```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
```

## 6. Model Training and Evaluation

Train a model on the SMOTE-resampled data and evaluate it.

```python
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # YOUR CODE HERE
    # Initialize a model
    model = LogisticRegression(max_iter = 1000) 

    # Train
    model.fit(X_train, y_train)
    
    return model

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    # 3. Create confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    # 4. Return metrics in a dictionary
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'AUC': auc, 'confusion matrix': con_mat}
    
    return metrics
```

## 7. Save Results

Save the evaluation metrics to a text file.

```python
# YOUR CODE HERE
# 1. Create 'results' directory if it doesn't exist
# 2. Format metrics as strings
# 3. Write metrics to 'results/results_part3.txt'
def output_results3(metrics):
    os.makedirs('results', exist_ok=True)
    with open('results/results_part3.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{str(key)}: {str(value)}\n")
    return
```

## 8. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion matrix':
            print(f"{metric}: {value:.4f}")
    
    # 7. Save results
    output_results3(metrics)
    
    # 8. Load Part 1 results for comparison
    import json
    import re

    part1_metrics = {}

    try:
        with open('results/results_part1.txt', 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("confusion matrix:"):
                matrix = []
                i += 1  # Move to the next line after 'confusion matrix:'
                while i < len(lines) and lines[i].strip().startswith('['):
                    # Extract numbers from the line using regex
                    nums = list(map(int, re.findall(r'\d+', lines[i])))
                    matrix.append(nums)
                    i += 1
                part1_metrics["confusion matrix"] = matrix
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = float(value.strip())
                part1_metrics[key] = value

            i += 1

    except FileNotFoundError:
        print("File not found.")

    # 9. Compare models
    comparison = compare_models(part1_metrics, metrics)
    print("\nModel Comparison (improvement percentages):")
    for metric, improvement in comparison.items():
        print(f"{metric}: {improvement:.2f}%")
```

## 9. Compare Results

Implement a function to compare model performance between balanced and imbalanced data.

```python
def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'AUC']
    improvements = {}
    
    # 1. Calculate percentage improvement for each metric
    for metric in metrics_to_compare:
        if metric in part1_metrics and metric in part3_metrics:
            part1_value = part1_metrics[metric]
            part3_value = part3_metrics[metric]

            change = part3_value - part1_value
            # calculate percentage improvement
            pct_improvement = (change / part1_value) * 100

            improvements[metric] = pct_improvement
    return improvements