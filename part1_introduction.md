# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 1: Introduction to Classification & Evaluation

**Objective:** Load the synthetic health data, train a Logistic Regression model, and evaluate its performance.

## 1. Setup

Import necessary libraries.

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
```

## 2. Data Loading

Implement the `load_data` function to read the dataset.

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
    df = pd.read_csv(file_path)
    
    return df 
```

## 3. Data Preparation

Implement `prepare_data_part1` to select features, split data, and handle missing values.

```python
def prepare_data_part1(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: select features, split into train/test sets, handle missing values.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Define features and outcome
    feature_cols = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    target_col = 'disease_outcome'

    # Select only the columns we want
    X = df[feature_cols]
    y = df[target_col]

    # print(f"Missing feature values:\n{X.isna().sum()}\n")
    # print(f"Missing target values: {y.isna().sum()}")

    # Split using the sklearn function
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # transform into DataFrames (per documentation .fit_transform() and .transform() return ndarrays
    X_train = pd.DataFrame(X_train_imputed, columns=feature_cols, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=feature_cols, index=X_test.index)

    # print("\nAfter imputation...\n")
    # print(f"Missing feature values (train):\n{X_train.isna().sum()}\n")
    # print(f"Missing feature values (test):\n{X_test.isna().sum()}\n")
    # print(f"Missing target values (train): {y_train.isna().sum()}")
    # print(f"Missing target values (test): {y_test.isna().sum()}")

    # Placeholder return - replace with your implementation
    return X_train, X_test, y_train, y_test
```

## 4. Model Training

Implement `train_logistic_regression`.

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
    model = LogisticRegression() 

    # Train
    model.fit(X_train, y_train)
    
    return model
```

## 5. Model Evaluation

Implement `calculate_evaluation_metrics` to assess the model's performance.

```python
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

## 6. Save Results

Save the calculated metrics to a text file.

```python
# Create results directory and save metrics
def output_results(metrics):
    os.makedirs('results', exist_ok=True)
    with open('results/results_part1.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{str(key)}: {str(value)}\n")
    return
```

## 7. Main Execution

Run the complete workflow.

```python
# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    
    # 3. Train model
    model = train_logistic_regression(X_train, y_train)
    
    # 4. Evaluate model
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 5. Print metrics
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    # 6. Save results
    output_results(metrics)
    
    # 7. Interpret results
    interpretation = interpret_results(metrics)
    print("\nResults Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
```

## 8. Interpret Results

Implement a function to analyze the model performance on imbalanced data.

```python
def interpret_results(metrics):
    """
    Analyze model performance on imbalanced data.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Dictionary with keys:
        - 'best_metric': Name of the metric that performed best
        - 'worst_metric': Name of the metric that performed worst
        - 'imbalance_impact_score': A score from 0-1 indicating how much
          the class imbalance affected results (0=no impact, 1=severe impact)
    """
    # YOUR CODE HERE
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'AUC']
    metric_values = {k: metrics[k] for k in metric_keys}
    # 1. Determine which metric performed best and worst
    # I'm not sure what best and worst means because all the metrics describe different concepts??
    best_metric = max(metric_values, key=metric_values.get)
    worst_metric = min(metric_values, key=metric_values.get)
    
    # 2. Calculate an imbalance impact score based on the difference
    #    between accuracy and more imbalance-sensitive metrics like F1 or recall
    # Not sure there is a standard definition for this - could try to resample the data to change the class balance
    # and then remeasure metrics - but that doesn't seem to be in the spirit of this question
    # Will define my own score I guess??
    # Per instructions, will use accuracy, F1, recall, etc., and 0 means little impact while 1 means severe impact
    # So if accuracy is high and F1 or recall is low, severe impact - can conceptualize this as a simple difference
    imbalance_impact_score = metrics['accuracy'] - min(metrics['f1'], metrics['recall']) # yields the largest impact score
    
    # 3. Return the results as a dictionary
    
    # Placeholder return - replace with your implementation
    return {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': imbalance_impact_score
    }