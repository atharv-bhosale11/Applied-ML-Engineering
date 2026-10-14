# 🤖 Trained Models — Breast Cancer Classification

## 📌 Overview

This directory stores the trained machine learning model selected by the **Breast Cancer Classification Using Ensemble Learning** project.

The model is selected based on stratified 5-fold cross-validation, prioritizing malignant recall, followed by malignant F1-score and accuracy.

## 📁 Model File

| File Name | Description |
|---|---|
| `BestBreastCancerModel.pkl` | Serialized Scikit-learn pipeline containing the selected trained classifier and preprocessing steps. |

## 🧠 Algorithms Evaluated

The project compares the following five ensemble classifiers:

1. Random Forest
2. AdaBoost
3. Gradient Boosting
4. Hard Voting Classifier
5. Soft Voting Classifier

The selected model is saved for potential reuse without repeating the complete training process.

## ⚙️ Model Selection

The model selection process prioritizes these cross-validation metrics:

1. **Malignant Recall:** Prioritizes identifying actual malignant cases.
2. **Malignant F1-Score:** Balances precision and recall for the malignant class.
3. **Cross-Validation Accuracy:** Measures overall classification performance.

The final model is selected using training data. The held-out test set is reserved for final performance evaluation.

## 💾 Model Saving

The trained model is serialized using Python's Pickle module.

The output filename is:

`BestBreastCancerModel.pkl`

The model includes the preprocessing pipeline and the selected classifier.

## 🔄 Loading the Saved Model

Use the following Python example to load the saved model:

```python
import pickle

with open("BestBreastCancerModel.pkl", "rb") as file:
    model = pickle.load(file)
