# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. 

The accuracy metric performed the best (91.6%). This is classic for imbalanced datasets. Since the majority class outweighs the minority, a model that naively favors the majority class will appear "accurate" even just because it is labelling lots of things in the majority class. For example, if a spam filter for a junk email address labels everything as spam, it may have a 99% accuracy.

My recall metric was the worst (30.1%). In medicine we refer to this as sensitivity and this reflects the proportion of the positive class that was detected. This was low, which is concerning, suggesting that many people with the disease outcome were not detected by this model. Again this could be due to the class imbalance and not having enough information/examples of the minority class.

Here I used an informal imbalance score and got that it was around 0.6, which is moderately high (max 1), suggesting that class imbalance may be affecting our model. The best way would be to reanalyze the results in a balanced case. In part 3 we resample the dataset in a balanced way and reanalyze the results.

My confusion matrix reflects the results, where the majority of positive cases actually were labeled as negative and redemonstrates the concepts we get from reviewing the metrics (the confusion matrix has essentially the same information as the metrics). Given the importance of disease outcome, this may be unacceptable. 

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

XGBoost outperformed the random forest algorithm in terms for AUC, though the difference was small ~1%. XGBoost is generally the preferred algorithm, that while more computationally intensive, has better discriminatory abilities when properly tuned due to its sequential and iterative methods.

I didn't explicitly run the analyses with and without time-series data, but looking at the contributions of features to the models, time-series data was the top (or amongst the top) contributing feature for both models.

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
- Which metrics showed the least improvement?
- Why might some metrics improve more than others?
- What does this tell you about the importance of addressing class imbalance?

Recall and f1 improved. Accuracy and precision got worse. Recall and precision changed the most. This is a classic precision-recall trade off. I will refer to precision as positive predictive value and recall as sensitivity. Positive predictive value is ratio of the positive cases divided by those predicted to be positive. In the imbalanced case, this was higher than in the balanced case. This might suggest that in the imbalanced case, where the positive cases were the minority, the amount of positive predictions were low, and the ratio of true cases amongst the positive predictions was high, potentially because these were "clearly" positive cases. The positive label may have been reserved for these clearly positive cases, and less clear cases were not detected, hence low sensitivity. In the balanced case, more cases were detected (sensitivity increased), likely because the model could see more patterns of positive cases, but the positive predictive value decreased because of more mixed, marginal, or less "clearly" positive cases. In any case, this demonstrates the importance of class imbalance/balance.

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

*Your conclusions here...*

There was a lot of imbalance in the data set. Only around 10% of the dataset had the positive disease outcome, which impacted many metrics. Perhaps this could have been overcome by increased n (hopeful), or methods like cross-validation. Using more complex modeling methods (i.e. XGBoost vs. LogisticRegression) seemed to help, as well as resampling the data. In the future, perhaps cross-validation may help create even more accurate models. An understanding of the clinical relevance of the data may also be helpful (i.e. what is the "trade off" between a false positive vs. a false negative, etc.).