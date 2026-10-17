# Trained Machine Learning Models

## 📌 Overview

This directory contains the trained Machine Learning model saved by the Credit Card Fraud Detection project.

The model is used to predict whether a credit card transaction is legitimate or fraudulent.

## 📂 Files Description

### `CreditCardFraudModel.pkl`

This file contains the trained Logistic Regression model selected after hyperparameter tuning.

The model is serialized using Joblib and saved with a `.pkl` file extension.

It can be loaded later to make predictions without retraining the model.

## 🎯 Model Purpose

The trained model predicts whether a credit card transaction belongs to one of two classes:

| Class | Description |
|---|---|
| `0` | Legitimate Transaction |
| `1` | Fraudulent Transaction |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
