# 🩺 Breast Cancer Classification Using Ensemble Learning

## 📌 Overview

This project implements a machine learning pipeline to classify breast cancer cases using ensemble learning techniques.

Five classification models are evaluated using stratified 5-fold cross-validation. The best model is selected by prioritizing malignant recall, followed by malignant F1-score and cross-validation accuracy.

The project includes data preparation, preprocessing, model training, evaluation, model comparison, feature importance analysis, individual predictions, and model serialization.

## 🎯 Objectives

- Analyze and prepare the breast cancer dataset.
- Handle missing numerical and categorical values.
- Split the dataset into training and testing sets.
- Implement five ensemble classification models.
- Compare models using stratified 5-fold cross-validation.
- Select the best model using malignant-class performance.
- Evaluate models using multiple classification metrics.
- Generate comparison graphs and a confusion matrix.
- Analyze feature importance when supported by the selected model.
- Save the selected trained model.
- Display predictions for individual test records.

## 🤖 Algorithms Implemented

| No. | Algorithm | Description |
|:---:|---|---|
| 1 | Random Forest | Combines multiple decision trees for classification. |
| 2 | AdaBoost | Combines weak learners sequentially to improve classification. |
| 3 | Gradient Boosting | Builds an ensemble of learners sequentially to improve predictions. |
| 4 | Hard Voting Classifier | Predicts the class receiving the majority of votes. |
| 5 | Soft Voting Classifier | Combines predicted class probabilities to determine the final class. |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Pickle

## 📂 Project Structure

```text
07-Breast-Cancer-Classification-Using-Ensemble-Learning/
│
├── data/
│   ├── breast_cancer.csv
│   └── README.md
│
├── src/
│   ├── breastcancer_classification.py
│   └── README.md
│
├── models/
│   ├── BestBreastCancerModel.pkl
│   └── README.md
│
├── results/
│   ├── Model_Comparison.csv
│   └── README.md
|── requirments.txt
|   
├── screenshots/
│   ├── output 1.png
│   ├── output 2.png
│   ├── output 3.png
│   ├── output 4.png
│   ├── output 5.png
│   └── README.md
│
└── README.md

## 👨‍💻 Author

**Atharv Tushar Bhosale**

GitHub: [@atharv-bhosale11](https://github.com/atharv-bhosale11)
