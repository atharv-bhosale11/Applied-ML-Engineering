# 🏡 California Housing Price Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning regression algorithms to predict median house values in California based on housing characteristics, household income, population, and geographical information.

The project includes data validation, model training, cross-validation, hyperparameter tuning, performance evaluation, and visualization to identify a suitable regression model.

## 🎯 Objectives

- Load and validate the California Housing dataset.
- Perform data exploration and preprocessing.
- Train multiple regression models.
- Compare model performance using cross-validation.
- Optimize the selected model using hyperparameter tuning.
- Evaluate predictions using regression metrics.
- Visualize actual versus predicted values and residuals.
- Save the trained model and evaluation results.

## 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset | California Housing Dataset |
| Dataset File | `california_housing.csv` |
| Total Records | 20,640 |
| Input Features | 8 |
| Target Variable | `target` |
| Problem Type | Regression |

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median household income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | District population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |
| `target` | Median house value in units of $100,000 |

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pickle

## ⚙️ Project Workflow

1. Load the dataset.
2. Validate the dataset and inspect its structure.
3. Separate input features and target variable.
4. Split the dataset into training and testing sets.
5. Create regression models.
6. Perform cross-validation.
7. Compare model performance.
8. Select the best initial model.
9. Perform hyperparameter tuning.
10. Train and evaluate the final model.
11. Generate prediction and residual plots.
12. Save the trained model and evaluation metrics.

## 🤖 Machine Learning Models

The project compares regression models using cross-validation and selects a candidate for further optimization.

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📈 Visualizations

The project generates visualizations to analyze model performance:

- **MAE Comparison:** Compares models using Mean Absolute Error.
- **RMSE Comparison:** Compares models using Root Mean Squared Error.
- **R² Comparison:** Compares models using R² scores.
- **Actual vs. Predicted Plot:** Compares actual house values with model predictions.
- **Residual Plot:** Helps analyze prediction errors.

Additional screenshots are available in the `screenshots` directory.

## 📂 Project Structure

```text
08-California-Housing-Price-Prediction-Using-Machine-Learning/
│
├── dataset/
│   ├── california_housing.csv
│   └── README.md
│
├── src/
│   ├── californiahousingpriceprediction.py
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── Final_Model_Metrics.csv
│   └── README.md
│
├── screenshots/
│   └── README.md
│
└── README.md
├── requirments.txt

```

*The directory structure above illustrates the intended organization. Actual filenames and locations may differ depending on your repository.*

## 💻 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/atharv-bhosale11/Applied-ML-Engineering.git
```

### 2. Navigate to the Project Directory

```bash
cd Applied-ML-Engineering/08-California-Housing-Price-Prediction-Using-Machine-Learning
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Run the Python Script

```bash
python src/californiahousingpriceprediction.py
```

Ensure that the dataset is available at the path configured in the source code before running the script.

## 📁 Model Availability

The trained model, `BestHousingModel.pkl`, is approximately 283,000 KB (about 276 MiB).

It has not been uploaded to this repository because its size exceeds GitHub's standard 25 MiB browser-upload limit. The model directory contains a README explaining this limitation.

To obtain the model, run the training script to generate it locally.

## 📊 Results

The project generates model evaluation metrics and visualizations to help assess prediction performance.

The final evaluation metrics are stored in:

`results/Final_Model_Metrics.csv`

Refer to the `screenshots` directory for visual evidence of dataset validation, cross-validation, model comparisons, and final evaluation.

## 🔮 Future Improvements

- Experiment with additional regression algorithms.
- Improve model performance through feature engineering.
- Analyze feature importance.
- Develop an interactive house price prediction interface.
- Deploy the trained model as a web application.

## 👨‍💻 Author

**Atharv Tushar Bhosale**


