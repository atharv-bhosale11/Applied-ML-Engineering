# 🏡 California Housing Price Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning regression algorithms to predict median house values in California based on housing characteristics, household income, population, and geographical information.

The objective is to train, evaluate, and optimize regression models to identify the best-performing model for predicting house prices.

## 🎯 Objectives

- Load and validate the California Housing dataset.
- Perform data exploration and preprocessing.
- Split the dataset into training and testing sets.
- Train multiple regression models.
- Evaluate models using cross-validation.
- Optimize the selected model using hyperparameter tuning.
- Evaluate the final model on test data.
- Visualize actual versus predicted values and residuals.
- Save the trained model and evaluation results.

## 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset File | `california_housing.csv` |
| Total Records | 20,640 |
| Input Features | 8 |
| Target Variable | `target` |
| Problem Type | Regression |

### Features

- `MedInc` — Median household income
- `HouseAge` — Median house age
- `AveRooms` — Average number of rooms
- `AveBedrms` — Average number of bedrooms
- `Population` — District population
- `AveOccup` — Average household occupancy
- `Latitude` — Geographic latitude
- `Longitude` — Geographic longitude

**Target:** `target` — Median house value in units of $100,000.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pickle

## ⚙️ Project Workflow

1. Load the dataset.
2. Validate the dataset.
3. Separate features and target.
4. Split data into training and testing sets.
5. Create regression models.
6. Perform cross-validation.
7. Compare model performance.
8. Select the best initial model.
9. Perform hyperparameter tuning.
10. Train and evaluate the final model.
11. Visualize predictions and residuals.
12. Save the trained model and evaluation metrics.

## 📈 Model Evaluation

The models can be evaluated using the following regression metrics:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| MSE | Measures average squared prediction error |
| RMSE | Measures prediction error in the target's units |
| R² Score | Measures how well the model explains target variation |

## 📉 Visualizations

The project generates the following visualizations:

- **Actual vs. Predicted Plot** — Compares actual house values with predicted values.
- **Residual Plot** — Helps analyze prediction errors.
- **Model Comparison Plot** — Compares regression model performance.

## 💾 Model and Results

The project saves the trained model and generated evaluation results.
 
Expected output files include:

- `BestHousingModel.pkl` — Saved trained model.
- `Actual_vs_Predicted.png` — Actual versus predicted visualization.
- `Residual_Plot.png` — Residual visualization.
- Model comparison plots and evaluation results.

*Output filenames and locations depend on the paths configured in the source code.*
