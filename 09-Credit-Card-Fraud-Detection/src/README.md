# Source Code

## 📌 Overview

This directory contains the Python source code for the Credit Card Fraud Detection project.

The project uses Machine Learning classification algorithms to identify potentially fraudulent credit card transactions.

## 📂 Files

### `creditcardfrauddetection.py`

This is the main Python source file responsible for executing the Credit Card Fraud Detection workflow.

The script includes the following functionalities:

- Dataset loading and validation
- Exploratory Data Analysis (EDA)
- Data preparation and train-test splitting
- Feature preprocessing
- Logistic Regression model training
- Multiple classification model comparison
- Hyperparameter tuning
- Cross-validation
- Classification threshold optimization
- Final model evaluation
- Prediction error analysis
- Feature importance analysis
- Class imbalance technique comparison
- Model serialization and saving results

## 🤖 Machine Learning Algorithms

The project uses the following classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Linear Support Vector Machine (Linear SVM)

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Average Precision
- ROC-AUC

## 📥 Dataset

The dataset is not included in this repository because of its large file size.

Download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place `creditcard.csv` in the directory expected by the Python script.

## 🚀 Execution

Run the source file using Python:

```bash
python creditcardfrauddetection.py
