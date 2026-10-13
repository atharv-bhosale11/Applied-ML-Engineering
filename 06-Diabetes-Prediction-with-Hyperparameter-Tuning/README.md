# 🩺 Diabetes Prediction with Hyperparameter Tuning

A machine learning project that predicts diabetes using medical diagnostic features. This project implements multiple classification algorithms, performs hyperparameter optimization using GridSearchCV, compares model performance through cross-validation, and evaluates the selected model on a separate test dataset.

## 📌 Project Overview

The objective of this project is to develop and compare machine learning classification models for diabetes prediction.

Five classification algorithms are evaluated to identify the model with the highest mean cross-validation accuracy. The selected model is then evaluated using standard classification metrics and saved for future use.

## 🎯 Project Objectives

- Analyze and preprocess the diabetes dataset.
- Handle invalid zero values in selected medical measurement features.
- Apply missing-value imputation and feature scaling where required.
- Train multiple machine learning classification algorithms.
- Optimize model hyperparameters using GridSearchCV.
- Compare algorithms using stratified 5-fold cross-validation.
- Evaluate the selected model on a held-out test dataset.
- Visualize algorithm comparison and confusion matrix results.
- Save the trained model for future predictions.

## 📊 Dataset Information

| Property | Description |
|---|---|
| Dataset File | `diabetes.csv` |
| Total Records | 768 |
| Total Columns | 9 |
| Input Features | 8 |
| Target Variable | `Outcome` |
| Problem Type | Binary Classification |

### 🔍 Input Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skinfold thickness |
| `Insulin` | Insulin level |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function value |
| `Age` | Age of the individual |

### 🎯 Target Variable

The `Outcome` column represents the target class:

- `0` — No diabetes
- `1` — Diabetes

## 🤖 Machine Learning Algorithms

The following classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## 🛠️ Technologies and Libraries

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Model Serialization:** Joblib

## ⚙️ Machine Learning Workflow

1. Load and inspect the dataset.
2. Analyze missing values, duplicate records, and statistical information.
3. Handle invalid zero values in selected medical measurement columns.
4. Impute missing values and apply feature scaling where required.
5. Separate the input features and target variable.
6. Split the dataset into training and testing sets.
7. Build preprocessing and classification pipelines.
8. Perform hyperparameter tuning using GridSearchCV.
9. Compare algorithms using stratified 5-fold cross-validation.
10. Select the algorithm with the highest mean cross-validation accuracy.
11. Evaluate the selected model on the held-out test dataset.
12. Generate comparison visualizations and a confusion matrix.
13. Save the selected model for future predictions.

## 🎛️ Model Optimization

Hyperparameter tuning is performed using `GridSearchCV` with stratified 5-fold cross-validation.

The best-performing model is selected according to its mean cross-validation accuracy on the training data.

The held-out test dataset is used separately to evaluate the selected model's performance on unseen data.

## 📈 Evaluation Metrics

The selected model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

These metrics provide different perspectives on classification performance, including overall correctness and the identification of positive and negative cases.

## 📁 Project Structure

```text
06-Diabetes-Prediction-with-Hyperparameter-Tuning/
│
├── src/
│   ├── diabetesprediction.py
│   └── README.md
│
├── data/
│   └── README.md
│
├── models/
│   ├── BestDiabetesModel.pkl
│   └── README.md
│
├── screenshots/
│   ├── README.md
│   ├── output 1.png
│   ├── output 2.png
│   ├── output 3.png
│   ├── output 4.png
│   ├── Algorithm_Comparison.png
│   └── Confusion_Matrix.png
│
├── diabetes.csv
├── requirements.txt
└── README.md

## Author

**Atharv Tushar Bhosale**

