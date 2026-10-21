# Olist E-Commerce Data Science Project

## 📌 Project Overview
This project analyzes the Olist Brazilian E-Commerce Public Dataset using Python, Data Analysis, and Machine Learning techniques.

The project focuses on understanding customer purchasing behavior, analyzing delivery performance, predicting late deliveries, and generating business insights.

## 🎯 Objectives
- Load and analyze multiple e-commerce datasets.
- Clean and preprocess the data.
- Merge datasets into a master dataset.
- Perform Exploratory Data Analysis (EDA).
- Segment customers using RFM Analysis.
- Predict late deliveries using Machine Learning.
- Compare and evaluate classification models.
- Generate business insights and recommendations.
- Export summary datasets for Power BI.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Power BI

## 📊 Project Workflow

### 1. Data Loading
Loads nine CSV datasets related to customers, orders, products, payments, reviews, sellers, and geolocation.

### 2. Data Understanding
Examines dataset dimensions, column names, data types, and statistical summaries.

### 3. Data Quality Checks
Identifies missing values and duplicate records.

### 4. Data Cleaning
Converts date columns, processes product categories, and prepares geolocation data.

### 5. Data Integration
Merges datasets into a master dataset containing one row per order.

### 6. Feature Engineering
Creates delivery-related features, freight ratios, purchase date features, and product weight and volume features.

### 7. Exploratory Data Analysis
Analyzes monthly orders, sales, delivery delays, and customer review scores.

### 8. RFM Customer Segmentation
Segments customers based on:
- Recency
- Frequency
- Monetary Value

### 9. Late Delivery Prediction
Creates a binary target variable to classify delivered orders as late or on time/early.

### 10. Machine Learning Models
Trains and compares the following models:
- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### 11. Model Evaluation
Evaluates models using:
- Area Under the Precision-Recall Curve (AUPRC)
- Precision
- Recall
- F1-score
- Confusion Matrix

### 12. Business Insights
Generates insights about delivery performance, product category sales, customer satisfaction, and customer segmentation.

### 13. Power BI Integration
Exports summary datasets for business intelligence reporting.

## 📈 Model Results

Initial model evaluation produced the following results:

| Model | AUPRC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.4636 | 0.5281 | 0.0647 | 0.1153 |
| Baseline | 0.3996 | 0.3996 | 1.0000 | 0.5710 |

**Note:** These are the initial results. The low recall of Gradient Boosting indicates that further model improvement and classification threshold evaluation are needed.
