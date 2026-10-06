# Advertising Sales Prediction using Linear Regression

## Overview

Advertising plays a crucial role in increasing product sales. This project applies the Linear Regression Machine Learning algorithm to predict sales based on advertising budgets spent on TV, Radio, and Newspaper advertisements.

The project performs complete data analysis, preprocessing, model training, prediction, evaluation, and visualization to understand how advertising investments influence sales performance.

---

## Problem Statement

Predict product sales using advertising expenditure across different marketing channels.

### Input Features
- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

### Target Variable
- Sales

---

## Dataset Information

- Dataset Name: Advertising.csv
- Total Records: 200
- Features: 3
- Target Variable: Sales

The dataset contains advertising spending information and corresponding sales figures.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values by establishing a relationship between independent variables and a dependent variable.

In this project, the model learns the relationship between advertising budgets and sales performance.

---

## Project Workflow

### Step 1: Dataset Loading
- Load Advertising.csv dataset
- Display initial records
- Verify dataset dimensions

### Step 2: Data Cleaning
- Remove unnecessary columns
- Validate dataset structure

### Step 3: Exploratory Data Analysis
- Missing Value Analysis
- Statistical Summary
- Correlation Analysis

### Step 4: Feature Selection
- Independent Variables:
  - TV
  - Radio
  - Newspaper

- Dependent Variable:
  - Sales

### Step 5: Train-Test Split
- Training Data: 80%
- Testing Data: 20%

### Step 6: Model Training
- Train Linear Regression model using training data

### Step 7: Prediction
- Generate sales predictions using testing data

### Step 8: Model Evaluation
Evaluate model performance using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Step 9: Feature Importance Analysis
- Display coefficients of all advertising channels
- Display intercept value

### Step 10: Result Comparison
- Compare Actual Sales vs Predicted Sales

### Step 11: Visualization
- Correlation Matrix
- Actual vs Predicted Sales Scatter Plot

---

## Project Structure

```text
03-Advertising-Sales-Prediction-Linear-Regression
│
├── data
│   ├── Advertising.csv
│   └── README.md
│
├── src
│   ├── AdvertiseFinal.py
│   └── README.md
│
├── screenshots
│   ├── Output Screenshots
│   └── README.md
│
├── requirements.txt
└── README.md

## Author 

**Atharv Tushar Bhosale**

Python Developer | Machine Learning Enthusiast

