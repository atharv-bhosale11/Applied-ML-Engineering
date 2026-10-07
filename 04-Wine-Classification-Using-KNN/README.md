# 🍷 Wine Classification Using K-Nearest Neighbors (KNN)

## 📌 Project Overview

This project implements a Machine Learning Classification model using the **K-Nearest Neighbors (KNN)** algorithm to classify different types of wine based on their chemical properties.

The objective is to analyze wine characteristics and accurately predict the wine class using supervised machine learning techniques. The project follows a complete machine learning workflow including data preprocessing, feature scaling, hyperparameter tuning, model training, evaluation, and visualization.

---

## 🎯 Problem Statement

Given various chemical attributes of a wine sample, predict the correct wine category (Class 1, Class 2, or Class 3) using the K-Nearest Neighbors Classification algorithm.

---

## 📂 Dataset Information

- **Dataset Name:** Wine Dataset
- **Total Records:** 178
- **Total Features:** 13
- **Target Variable:** Class
- **Problem Type:** Multi-Class Classification

### Input Features

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

### Output Feature

- Wine Class (1, 2, 3)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm that classifies a data point based on the majority class among its nearest neighbors.

---

## 🚀 Project Workflow

### Step 1 : Display Project Information

### Step 2 : Load Dataset

### Step 3 : Clean Dataset

### Step 4 : Display Dataset Information

### Step 5 : Check Missing Values

### Step 6 : Display Statistical Summary

### Step 7 : Prepare Features and Target

### Step 8 : Split Dataset

### Step 9 : Perform Feature Scaling

### Step 10 : Explore Multiple K Values

### Step 11 : Plot K vs Accuracy Graph

### Step 12 : Find Best K Value

### Step 13 : Train Final KNN Model

### Step 14 : Generate Predictions

### Step 15 : Calculate Accuracy

### Step 16 : Display Confusion Matrix

### Step 17 : Display Classification Report

### Step 18 : Display Project Summary

---

## 📊 Model Performance

| Metric | Result |
|----------|----------|
| Best K Value | 7 |
| Accuracy | 100% |
| Classification Type | Multi-Class |

---

## 📈 Visualizations

### K Value vs Accuracy Graph

Used to determine the optimal value of K for the KNN model.

### Confusion Matrix Heatmap

Provides a graphical representation of model predictions versus actual classifications.

---

## 📁 Project Structure

```text
04-Wine-Classification-Using-KNN
│
├── data
│   └── WinePredictor.csv
│
├── src
│   └── WineClassificationUsingKNN.py
│
├── screenshots
│   ├── K_Value_vs_Accuracy.png
│   └── Confusion_Matrix.png
│
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd 04-Wine-Classification-Using-KNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python WineClassificationUsingKNN.py
```

---

## 📚 Learning Outcomes

- Data Preprocessing
- Feature Scaling
- KNN Classification
- Hyperparameter Tuning
- Model Evaluation
- Confusion Matrix Analysis
- Classification Report Interpretation 
- Machine Learning Workflow Design

---

## 👨‍💻 Author

**Atharv Tushar Bhosale**

Software Developer | Java Developer | Python Developer | Machine Learning Developer

---

⭐ If you found this project useful, consider giving it a star.
