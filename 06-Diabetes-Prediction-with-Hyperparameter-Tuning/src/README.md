# Source Code — Diabetes Prediction with Hyperparameter Tuning

This directory contains the Python implementation of the **Diabetes Prediction with Hyperparameter Tuning** project. It includes data preprocessing, model training, hyperparameter optimization, algorithm comparison, model evaluation, and model saving.

## Source File

| File | Description |
|---|---|
| `DiabetesPrediction.py` | Main Python script for data preprocessing, model training, hyperparameter tuning, evaluation, visualization, and model saving. |

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Handle invalid zero values in selected medical measurement columns.
3. Impute missing values and apply feature scaling where required.
4. Split the dataset into training and testing sets.
5. Build machine learning pipelines.
6. Perform hyperparameter tuning using GridSearchCV and stratified 5-fold cross-validation.
7. Compare algorithms using mean cross-validation accuracy.
8. Select the best-performing algorithm.
9. Evaluate the selected model on the held-out test set.
10. Save the trained model and evaluation outputs.


```bash
pip install pandas numpy scikit-learn matplotlib joblib
