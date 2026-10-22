# Machine Learning Models

## 📌 Overview

This folder contains the trained Machine Learning model developed for the Olist E-Commerce Data Science Project.

The model is designed to predict late deliveries using order-related features from the Olist Brazilian E-Commerce Public Dataset.

## 📂 Model File

### `Best_late_delivery_model.pkl`

This file contains the saved, trained Gradient Boosting classification model selected during model comparison based on its validation Area Under the Precision-Recall Curve (AUPRC).

**Model Details:**

* **Algorithm:** Gradient Boosting Classifier
* **Task:** Binary Classification
* **Objective:** Predict whether an order will be delivered late
* **Target Variable:** `is_late_delivery`
* **Evaluation Metric:** Area Under the Precision-Recall Curve (AUPRC)

## 🎯 Model Purpose

The model aims to help identify orders that may experience delivery delays, supporting better delivery planning and operational decision-making.

## ⚙️ Loading the Model

The saved model can be loaded using Python's `joblib` library.

```python
import joblib

model = joblib.load("models/Best_late_delivery_model.pkl")

print("Late delivery prediction model loaded successfully.")
```

## ⚠️ Important Notes

* The model was trained using the Olist Brazilian E-Commerce Public Dataset.
* Initial test evaluation showed an AUPRC of approximately **0.4636**.
* The initial model had low recall, meaning it failed to identify many late deliveries.
* Use the same feature names, ordering, and preprocessing steps used during training when making predictions.
* Ensure that prediction features are available at the time the prediction is intended to be made.
* Only load `.pkl` files from trusted sources, because loading an untrusted serialized file can execute malicious code.

## 📜 Data Source

Olist Brazilian E-Commerce Public Dataset:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## 👨‍💻 Author

**Atharv Tushar Bhosale**
