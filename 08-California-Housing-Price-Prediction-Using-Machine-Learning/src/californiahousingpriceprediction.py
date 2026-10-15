"""
----------------------------------------------------------------------
Project Name        : California Housing Price Prediction Using Machine Learning
Dataset File        : california_housing.csv 
Target Variable     : target
Algorithms Used     : Linear Regression, Ridge Regression, Lasso Regression,
                      Support Vector Regression (SVR), Decision Tree Regressor,
                      Bagging Regressor, Random Forest Regressor,
                      AdaBoost Regressor, Gradient Boosting Regressor
Techniques Used     : Data Preparation, Train-Test Splitting,
                      Feature Scaling, 5-Fold Cross-Validation,
                      Hyperparameter Tuning, Model Comparison,
                      Model Selection, Regression Evaluation,
                      R2 Score, MAE, MSE, RMSE, Model Saving
Library             : Pandas, NumPy, Matplotlib, Scikit-Learn
Problem Type        : Supervised Learning - Regression
Author              : Atharv Tushar Bhosale
Date                : 15/10/2026
----------------------------------------------------------------------
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

"""
----------------------------------------------------------------------
Section Name : File Paths
Description  : Defines paths for the dataset, results, and saved model.
----------------------------------------------------------------------
"""

DATASET_PATH = "california_housing.csv"

RESULTS_PATH = "."

MODEL_PATH = "BestHousingModel.pkl"

"""
----------------------------------------------------------------------
Function Name : LoadDataset
Description   : Loads the California Housing dataset from a CSV file.
Input         : sFilePath - Path to the dataset CSV file.
Output        : DataFrame containing the loaded dataset.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def LoadDataset(sFilePath):

    df = pd.read_csv(sFilePath)

    return df

"""
----------------------------------------------------------------------
Function Name : DisplayDatasetInfo
Description   : Displays basic information about the loaded dataset.
Input         : df - California Housing dataset.
Output        : Displays dataset shape, columns, data types, and missing values.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def DisplayDatasetInfo(df):

    print("\n" + "-" * 70)
    print("STEP 1: DATASET INFORMATION")
    print("-" * 70)

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:", df.duplicated().sum())

"""
----------------------------------------------------------------------
Function Name : ValidateDataset
Description   : Validates the dataset and target column.
Input         : df - California Housing dataset.
Output        : True if validation succeeds.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def ValidateDataset(df):

    if df.empty or "target" not in df.columns:
        raise ValueError("Invalid dataset or missing target column.")

    if df["target"].isnull().any():
        raise ValueError("Target column contains missing values.")

    print("Dataset validation successful.")

    return True

"""
----------------------------------------------------------------------
Function Name : SeparateFeaturesTarget
Description   : Separates input features and the target variable.
Input         : df - California Housing dataset.
Output        : X - Input features, y - Target values.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def SeparateFeaturesTarget(df):

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y

"""
----------------------------------------------------------------------
Function Name : SplitDataset
Description   : Splits the dataset into training and testing sets.
Input         : X - Input features, y - Target values.
Output        : X_train, X_test, y_train, y_test.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def SplitDataset(X, y):

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

"""
----------------------------------------------------------------------
Function Name : CreateCrossValidation
Description   : Creates a reproducible 5-fold cross-validation strategy.
Input         : None.
Output        : KFold cross-validation object.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def CreateCrossValidation():

    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

"""
----------------------------------------------------------------------
Function Name : EvaluateModel
Description   : Calculates regression evaluation metrics.
Input         : yTest - Actual target values.
                yPred - Predicted target values.
Output        : Dictionary containing R2, MAE, MSE, and RMSE.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def EvaluateModel(yTest, yPred):

    fMSE = mean_squared_error(yTest, yPred)

    return {
        "R2": r2_score(yTest, yPred),
        "MAE": mean_absolute_error(yTest, yPred),
        "MSE": fMSE,
        "RMSE": np.sqrt(fMSE)
    }

"""
----------------------------------------------------------------------
Function Name : CreateModels
Description   : Creates nine regression models with suitable preprocessing.
Input         : None.
Output        : Dictionary containing nine regression models.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def CreateModels():

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=10000))
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Bagging": BaggingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

"""
----------------------------------------------------------------------
Function Name : CrossValidateModels
Description   : Evaluates regression models using 5-fold cross-validation.
Input         : models, X_train, y_train, cv.
Output        : DataFrame containing cross-validation results.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def CrossValidateModels(models, X_train, y_train, cv):

    scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error",
               "neg_root_mean_squared_error"]
    results = []

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=scoring, n_jobs=-1)

        results.append({
            "Algorithm": name,
            "R2": scores["test_r2"].mean(),
            "MAE": -scores["test_neg_mean_absolute_error"].mean(),
            "MSE": -scores["test_neg_mean_squared_error"].mean(),
            "RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
            "RMSE Std": scores["test_neg_root_mean_squared_error"].std()
        })

    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

"""
----------------------------------------------------------------------
Function Name : SaveComparisonResults
Description   : Saves cross-validation results to a CSV file.
Input         : dfResults - DataFrame containing model comparison results.
Output        : Saves Model_Comparison.csv in the results directory.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def SaveComparisonResults(dfResults):

    os.makedirs(RESULTS_PATH, exist_ok=True)
    dfResults.to_csv(os.path.join(RESULTS_PATH, "Model_Comparison.csv"), index=False)

    print("Model comparison results saved successfully.")

"""
----------------------------------------------------------------------
Function Name : PlotModelComparison
Description   : Plots RMSE, R2, and MAE comparisons for regression models.
Input         : dfResults - DataFrame containing cross-validation results.
Output        : Saves three model comparison plots in the results directory.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def PlotModelComparison(dfResults):

    os.makedirs(RESULTS_PATH, exist_ok=True)

    plots = {
        "RMSE": ("Mean Cross-Validation RMSE", "RMSE_Comparison.png"),
        "R2": ("Mean Cross-Validation R2 Score", "R2_Comparison.png"),
        "MAE": ("Mean Cross-Validation MAE", "MAE_Comparison.png")
    }

    for metric, (label, filename) in plots.items():
        results = dfResults.sort_values(metric, ascending=(metric != "R2"))

        plt.figure(figsize=(12, 6))
        plt.bar(results["Algorithm"], results[metric])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(label)
        plt.title(f"Regression Model Comparison - {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=300)
        plt.close()

    print("Model comparison plots saved successfully.")

"""
----------------------------------------------------------------------
Function Name : SelectBestModel
Description   : Selects the model with the lowest cross-validation RMSE.
Input         : models - Dictionary of regression models.
                dfResults - DataFrame containing cross-validation results.
Output        : Name and estimator of the best-performing model.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def SelectBestModel(models, dfResults):

    sBestModel = dfResults.iloc[0]["Algorithm"]

    print("Best Algorithm:", sBestModel)
    print("Cross-Validation RMSE:", dfResults.iloc[0]["RMSE"])

    return sBestModel, models[sBestModel]

"""
----------------------------------------------------------------------
Function Name : TuneBestModel
Description   : Tunes model hyperparameters using randomized search.
Input         : modelName, model, X_train, y_train, cv
Output        : Best tuned model and its parameters.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def TuneBestModel(modelName, model, X_train, y_train, cv):

    paramGrids = {
        "Ridge Regression": {"model__alpha": [0.1, 1, 10, 100]},
        "Lasso Regression": {"model__alpha": [0.001, 0.01, 0.1, 1]},
        "SVR": {"model__C": [0.1, 1, 10], "model__epsilon": [0.01, 0.1, 0.2]},
        "Decision Tree": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
        "Bagging": {"n_estimators": [10, 50, 100], "max_samples": [0.7, 1.0]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5]}
    }

    if modelName not in paramGrids:
        return model.fit(X_train, y_train), {}

    search = RandomizedSearchCV(
        model, paramGrids[modelName], n_iter=10,
        scoring="neg_root_mean_squared_error", cv=cv,
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)

    print("Best Parameters:", search.best_params_)
    print("Best Cross-Validation RMSE:", -search.best_score_)

    return search.best_estimator_, search.best_params_

"""
----------------------------------------------------------------------
Function Name : TrainAndEvaluateModel
Description   : Train the selected model and evaluate test performance.
Input         : model - Selected tuned estimator.
                X_train, y_train - Training data.
                X_test, y_test - Testing data.
Output        : Trained model and evaluation metrics.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def TrainAndEvaluateModel(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)
    yPred = model.predict(X_test)
    results = EvaluateModel(y_test, yPred)

    print("\nTest Set Performance")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    return model, results

def PlotActualVsPredicted(yTest, yPred):
    """Plot and save actual versus predicted house prices."""

    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(yTest, yPred, alpha=0.4)
    plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], "r--")
    plt.xlabel("Actual House Prices ($100,000s)")
    plt.ylabel("Predicted House Prices ($100,000s)")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()

    sPath = os.path.abspath(
        os.path.join(RESULTS_PATH, "Actual_vs_Predicted.png")
    )

    plt.savefig(sPath, dpi=300)
    print("Scatter plot saved at:", sPath)

    plt.show()
    plt.close()

def PlotResiduals(yTest, yPred):
    """Plot and save the residual plot."""

    os.makedirs(RESULTS_PATH, exist_ok=True)
    residuals = np.asarray(yTest) - np.asarray(yPred)

    plt.figure(figsize=(8, 6))
    plt.scatter(yPred, residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted House Prices ($100,000s)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()

    sPath = os.path.abspath(
        os.path.join(RESULTS_PATH, "Residual_Plot.png")
    )

    plt.savefig(sPath, dpi=300)
    print("Residual plot saved at:", sPath)

    plt.show()
    plt.close()

"""
----------------------------------------------------------------------
Function Name : SaveModelResults
Description   : Save the trained model and its evaluation metrics.
Input         : model - Final trained model.
                results - Dictionary containing evaluation metrics.
Output        : Saves the model and evaluation metrics.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def SaveModelResults(model, results):

    os.makedirs(RESULTS_PATH, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame([results]).to_csv(
        os.path.join(RESULTS_PATH, "Final_Model_Metrics.csv"),
        index=False
    )

    print("Final model and evaluation metrics saved successfully.")

"""
----------------------------------------------------------------------
Function Name : PredictHousePrice
Description   : Predict house prices using the saved trained model.
Input         : model - Trained regression model.
                features - New housing data.
Output        : Predicted house prices.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def PredictHousePrice(model, features):

    predictions = model.predict(features)

    return predictions

"""
----------------------------------------------------------------------
Function Name : LoadSavedModel
Description   : Load the previously saved regression model.
Input         : Model file path.
Output        : Loaded trained model.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def LoadSavedModel(sModelPath=MODEL_PATH):

    with open(sModelPath, "rb") as f:
        return pickle.load(f)

"""
----------------------------------------------------------------------
Function Name : main
Description   : Execute the complete California Housing Price Prediction
                workflow.
Input         : None.
Output        : Model comparison, evaluation metrics, plots, and saved model.
Author        : Atharv Tushar Bhosale
Date          : 15/10/2026
----------------------------------------------------------------------
"""

def main():
    """Control the complete California Housing Price Prediction workflow."""

    Border = "=" * 70

    print(Border)
    print("CALIFORNIA HOUSING PRICE PREDICTION USING MACHINE LEARNING")
    print(Border)

    print("\n" + Border)
    print("STEP 1: LOAD DATASET")
    print(Border)

    df = LoadDataset(DATASET_PATH)

    DisplayDatasetInfo(df)

    print("\n" + Border)
    print("STEP 2: VALIDATE DATASET")
    print(Border)

    ValidateDataset(df)

    print("\n" + Border)
    print("STEP 3: PREPARE DATA")
    print(Border)

    X, y = SeparateFeaturesTarget(df)

    X_train, X_test, y_train, y_test = SplitDataset(X, y)

    print("Features and target separated successfully.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\n" + Border)
    print("STEP 4: CREATE CROSS-VALIDATION STRATEGY")
    print(Border)

    cv = CreateCrossValidation()

    print("5-fold cross-validation created successfully.")

    print("\n" + Border)
    print("STEP 5: CREATE REGRESSION MODELS")
    print(Border)

    models = CreateModels()

    print("Total regression models:", len(models))

    for name in models:
        print("-", name)

    print("\n" + Border)
    print("STEP 6: CROSS-VALIDATION AND MODEL COMPARISON")
    print(Border)

    ResultsDF = CrossValidateModels(
        models,
        X_train,
        y_train,
        cv
    )

    print("\nModel Comparison Results:")
    print(ResultsDF.to_string(index=False))

    SaveComparisonResults(ResultsDF)

    print("\n" + Border)
    print("STEP 7: PLOT MODEL COMPARISON")
    print(Border)

    PlotModelComparison(ResultsDF)

    print("\n" + Border)
    print("STEP 8: SELECT BEST INITIAL MODEL")
    print(Border)

    BestAlgorithm, BestModel = SelectBestModel(
        models,
        ResultsDF
    )

    print("\nSelected Algorithm:", BestAlgorithm)

    print("\n" + Border)
    print("STEP 9: HYPERPARAMETER TUNING")
    print(Border)

    BestModel, BestParameters = TuneBestModel(
        BestAlgorithm,
        BestModel,
        X_train,
        y_train,
        cv
    )

    print("\nBest Parameters:", BestParameters)

    print("\n" + Border)
    print("STEP 10: TRAIN AND EVALUATE FINAL MODEL")
    print(Border)

    BestModel, Metrics = TrainAndEvaluateModel(
        BestModel,
        X_train,
        y_train,
        X_test,
        y_test
    )

    yPred = BestModel.predict(X_test)

    print("\n" + Border)
    print("STEP 11: PLOT ACTUAL VS PREDICTED VALUES")
    print(Border)

    PlotActualVsPredicted(y_test, yPred)

    print("\n" + Border)
    print("STEP 12: PLOT RESIDUALS")
    print(Border)

    PlotResiduals(y_test, yPred)

    print("\n" + Border)
    print("STEP 13: SAVE FINAL MODEL AND METRICS")
    print(Border)

    SaveModelResults(BestModel, Metrics)

    print("\n" + Border)
    print("STEP 14: FINAL PROJECT SUMMARY")
    print(Border)

    print("Best Algorithm:", BestAlgorithm)
    print("\nFinal Test Metrics:")

    for metric, score in Metrics.items():
        print(f"{metric}: {score:.4f}")

    print("\nResults Directory:", os.path.abspath(RESULTS_PATH))
    print("Saved Model:", os.path.abspath(MODEL_PATH))

    print("\n" + Border)
    print("CALIFORNIA HOUSING PRICE PREDICTION PROJECT")
    print("COMPLETED SUCCESSFULLY")
    print(Border)


if __name__ == "__main__":
    main()
