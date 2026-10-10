# Saved Machine Learning Model

This folder contains the trained Machine Learning model used for the Titanic Survival Prediction project.

## Model Information

- **Model Name:** Titanic Logistic Regression Model
- **File Name:** `TitanicLogisticRegression.pkl`
- **Algorithm:** Logistic Regression
- **Problem Type:** Binary Classification
- **Target Variable:** `Survived`
- **Model Format:** Pickle (`.pkl`)

## Model Description

The trained Logistic Regression model is saved using Joblib and stored in this folder.

The saved model can be loaded later to generate predictions without training the model again.

## Model Usage

The saved model can be loaded using:

```python
import joblib

model = joblib.load("TitanicLogisticRegression.pkl")

