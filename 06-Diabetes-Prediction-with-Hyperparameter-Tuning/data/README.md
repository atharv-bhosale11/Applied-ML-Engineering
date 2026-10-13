# Dataset — Diabetes Prediction

This directory contains the dataset used for the **Diabetes Prediction with Hyperparameter Tuning** project. The dataset is used to train, optimize, compare, and evaluate machine learning classification algorithms for diabetes prediction.

## Dataset Information

| Property | Description |
|---|---|
| Dataset File | `diabetes.csv` |
| Problem Type | Binary Classification |
| Target Variable | `Outcome` |
| Input Features | 8 |
| Total Columns | 9 |
| Total Records | 768 |

## Features Description

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skinfold thickness |
| `Insulin` | Insulin level |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function value |
| `Age` | Age of the individual |
| `Outcome` | Diabetes classification target |

## Target Variable

The `Outcome` column represents the target variable:

- `0` — No diabetes
- `1` — Diabetes

## Data Preprocessing

The dataset is processed before model training. The project handles selected zero values as invalid or missing measurements, imputes missing values, and applies feature scaling where required.

The columns in which zero values are treated as invalid measurements are:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

This treatment is based on the project's preprocessing assumption.

## Usage

The dataset is loaded by `diabetesprediction.py` and used for:

- Training machine learning models
- Hyperparameter tuning
- Cross-validation
- Algorithm comparison
- Model evaluation 

## Author

**Atharv Tushar Bhosale**
