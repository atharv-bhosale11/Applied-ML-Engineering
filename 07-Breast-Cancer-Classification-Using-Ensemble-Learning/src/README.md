# 🩺 Breast Cancer Classification Using Ensemble Learning

## 📌 Overview

This project implements a machine learning pipeline to classify breast cancer cases into malignant and benign categories using ensemble learning algorithms. It compares multiple classifiers using cross-validation and evaluates their performance on unseen test data.

The implementation includes data preparation, preprocessing, model training, performance evaluation, model comparison, feature importance analysis, and saving the selected model.

## 🎯 Objectives

- Load and analyze the breast cancer dataset.
- Prepare features and identify the target variable.
- Handle missing numerical and categorical values.
- Split the dataset into training and testing sets.
- Implement and compare five ensemble classification models.
- Evaluate models using stratified 5-fold cross-validation.
- Select the best model using malignant recall, malignant F1-score, and accuracy.
- Evaluate classification performance using multiple metrics.
- Generate model comparison, confusion matrix, and feature importance visualizations.
- Save the selected model for future use.
- Display predictions for individual test records.

## 🤖 Algorithms Implemented

| No. | Algorithm | Description |
|:---:|---|---|
| 1 | Random Forest | Combines multiple decision trees to improve classification performance. |
| 2 | AdaBoost | Combines weak learners sequentially, giving more attention to previously misclassified samples. |
| 3 | Gradient Boosting | Builds an ensemble of models sequentially to improve predictive performance. |
| 4 | Hard Voting Classifier | Predicts the class receiving the majority of votes from the individual classifiers. |
| 5 | Soft Voting Classifier | Combines predicted class probabilities to determine the final prediction. |

## 🛠️ Technologies Used

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Pickle

## 📂 Source File

| File | Description |
|---|---|
| `breastcancer_classification.py` | Implements the complete breast cancer classification workflow, including preprocessing, model training, cross-validation, evaluation, visualization, and model saving. |

## ⚙️ Implementation Workflow

1. **Dataset Loading:** Load the CSV dataset and examine its structure, missing values, duplicate records, and statistical summary.
2. **Data Preparation:** Identify the target column, remove duplicate records, separate features and target labels, and exclude identifier columns.
3. **Train-Test Splitting:** Divide the dataset into training and testing sets using an 80:20 ratio with stratification.
4. **Data Preprocessing:** Impute missing numerical values using the median and categorical values using the most frequent value. Apply one-hot encoding to categorical features.
5. **Model Creation:** Initialize Random Forest, AdaBoost, and Gradient Boosting classifiers.
6. **Voting Classifiers:** Combine the individual classifiers using hard and soft voting.
7. **Pipeline Construction:** Integrate preprocessing and classification into Scikit-learn pipelines.
8. **Cross-Validation:** Compare all five models using stratified 5-fold cross-validation.
9. **Model Selection:** Select the model using malignant recall, malignant F1-score, and cross-validation accuracy, in that priority order.
10. **Test-Set Evaluation:** Evaluate the models on the held-out test set without using test results to select the best model.
11. **Model Comparison:** Generate a graph comparing classification metrics across the five models.
12. **Confusion Matrix:** Visualize the predictions of the selected model.
13. **Feature Importance:** Display the most important features when supported by the selected model.
14. **Individual Predictions:** Compare predictions with actual labels for selected test records.
15. **Model Saving:** Save the selected trained model using Pickle.

## 📊 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified records. |
| Precision | Measures the proportion of predicted positive cases that are actually positive. |
| Recall | Measures the proportion of actual positive cases correctly identified. |
| F1-Score | Combines precision and recall into a single metric. |
| Malignant Recall | Measures the proportion of actual malignant cases correctly identified. |
| Malignant F1-Score | Balances precision and recall for the malignant class. |
| Confusion Matrix | Summarizes correct and incorrect classifications for each class. |

**Note:** Malignant recall is particularly important in this project because failing to identify a malignant case can have serious consequences. This project is an educational machine learning implementation and is not intended for clinical diagnosis.

## 📁 Generated Outputs

The script generates the following files when the corresponding workflow steps execute successfully:
 
| File | Description |
|---|---|
| `Model_Comparison.csv` | Stores test-set evaluation metrics for all five models. |
| `Algorithm_Comparison.png` | Visualizes the performance metrics of the models. |
| `Confusion_Matrix.png` | Displays the selected model's confusion matrix. |
| `Feature_Importance.png` | Visualizes the top ten features when feature importance is supported. |
| `BestBreastCancerModel.pkl` | Stores the selected trained model. |

## ▶️ Execution

Run the script from the project directory containing the dataset:

```bash
python src/breastcancer_classification.py
