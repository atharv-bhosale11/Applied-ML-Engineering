# 📊 Dataset — Breast Cancer Classification

## 📌 Overview

This directory contains the dataset used for the **Breast Cancer Classification Using Ensemble Learning** project.

The dataset is used to train and evaluate machine learning models that classify breast cancer cases into malignant and benign categories.

## 📁 Dataset File

| File Name | Description |
|---|---|
| `breast_cancer.csv` | CSV dataset containing the features and target labels used for breast cancer classification. |

## 🎯 Purpose

The dataset supports the following tasks:

- Exploratory dataset analysis
- Data preparation and feature selection
- Missing-value handling
- Training and testing machine learning models
- Cross-validation and model comparison
- Evaluation of malignant and benign classifications

## 🧾 Dataset Requirements

The CSV file should contain:

- **Feature columns:** Input variables used by the classification models.
- **Target column:** A column named `target` or `diagnosis` representing the class labels.
- **Two classes:** Malignant and benign.

The target labels and their corresponding meanings must be verified before training the models.

## ⚙️ Usage

The dataset is loaded by the Python implementation:

`src/breastcancer_classification.py`

Ensure that the dataset path in the script points to the correct location of `breast_cancer.csv`.

## ⚠️ Important Notes

- Verify the target column and malignant class label before executing the program.
- The dataset should contain sufficient records from both classes for stratified 5-fold cross-validation.
- Missing feature values are handled by the preprocessing pipeline.
- The dataset is intended for educational and machine learning experimentation.
- Predictions from this project must not be used as a substitute for professional medical diagnosis.

## 👨‍💻 Author

**Atharv Tushar Bhosale**
