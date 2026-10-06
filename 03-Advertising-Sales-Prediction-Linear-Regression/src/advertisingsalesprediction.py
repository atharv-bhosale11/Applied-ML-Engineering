"""
------------------------------------------------------------
Project Name        : Iris Flower Classification

Dataset Information :

Dataset File        : data/iris.csv (150 Samples)

Species Encoding    : Setosa     = 0
                      Versicolor = 1
                      Virginica  = 2

Features            : Sepal Length (cm)
                      Sepal Width (cm)
                      Petal Length (cm)
                      Petal Width (cm)

Target              : Species of the Iris Flower

Machine Learning Information :

Algorithms Used     : K-Nearest Neighbors (KNN)
                      Decision Tree Classifier

Library             : Scikit-Learn

Problem Type        : Multi-Class Classification
 
Evaluation Metrics  : Accuracy Score
                      Classification Report
                      Confusion Matrix

Visualization       : Confusion Matrix
                      Model Accuracy Comparison Graph

Author              : Atharv Tushar Bhosale

Date                : 05/10/2026
------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
"""
Project Name : Advertising Sales Prediction using Linear Regression
Description  : Predicts product sales based on TV, Radio, and Newspaper
               advertising budgets using Machine Learning.
Author       : Atharv Tushar Bhosale
Date         : 05/10/2026
"""

"""
Function Name : DisplayHeader
Description   : Displays project header.
Input         : None
Output        : Project title.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayHeader():

    print("-----------------------------------------------------------------")
    print("------- Advertising Sales Prediction Case Study -----------------")
    print("-----------------------------------------------------------------")

"""
Function Name : LoadDataset
Description   : Loads dataset from CSV file.
Input         : Dataset Path
Output        : DataFrame
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def LoadDataset(DataPath):

    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print(df.head())

    return df

"""
Function Name : RemoveUnwantedColumns
Description   : Removes unnecessary columns.
Input         : DataFrame
Output        : Cleaned DataFrame
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def RemoveUnwantedColumns(df):

    print("Shape Before Cleaning :", df.shape)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print("Shape After Cleaning :", df.shape)

    return df

"""
Function Name : CheckMissingValues
Description   : Displays missing values.
Input         : DataFrame
Output        : Missing values information.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def CheckMissingValues(df):

    print(df.isnull().sum())

"""
Function Name : DisplayStatisticalSummary
Description   : Displays statistical summary.
Input         : DataFrame
Output        : Statistical report.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayStatisticalSummary(df):

    print(df.describe())

"""
Function Name : DisplayCorrelationMatrix
Description   : Displays correlation matrix.
Input         : DataFrame
Output        : Correlation matrix.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayCorrelationMatrix(df):

    print("Correlation Matrix")
    print(df.corr())

"""
Function Name : PrepareData
Description   : Creates independent and dependent variables.
Input         : DataFrame
Output        : X and Y
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def PrepareData(df):

    X = df[['TV', 'radio', 'newspaper']]
    Y = df['sales']

    return X, Y

"""
Function Name : SplitData
Description   : Splits dataset into training and testing data.
Input         : X and Y
Output        : X_train, X_test, Y_train, Y_test
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def SplitData(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, Y_train, Y_test

"""
Function Name : TrainModel
Description   : Trains Linear Regression model.
Input         : X_train, Y_train
Output        : Trained model
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def TrainModel(X_train, Y_train):

    model = LinearRegression()

    model.fit(X_train, Y_train)

    return model

"""
Function Name : TestModel
Description   : Predicts sales values.
Input         : Model, X_test
Output        : Predicted values
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def TestModel(model, X_test):

    Y_pred = model.predict(X_test)

    return Y_pred

"""
Function Name : EvaluateModel
Description   : Evaluates model performance.
Input         : Y_test, Y_pred
Output        : MSE, RMSE, R2 Score
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def EvaluateModel(Y_test, Y_pred):

    MSE = mean_squared_error(Y_test, Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test, Y_pred)

    print("Mean Square Error :", MSE)
    print("Root Mean Square Error :", RMSE)
    print("R2 Score :", R2)
    print("Model Accuracy :", round(R2 * 100, 2), "%")

"""
Function Name : DisplayModelCoefficients
Description   : Displays coefficients and intercept.
Input         : Model, X
Output        : Feature importance.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayModelCoefficients(model, X):

    for column, value in zip(X.columns, model.coef_):
        print(column, ":", value)

    print("Intercept :", model.intercept_)

"""
Function Name : CompareActualVsPredicted
Description   : Compares actual and predicted sales.
Input         : Y_test, Y_pred
Output        : Comparison DataFrame.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def CompareActualVsPredicted(Y_test, Y_pred):

    Result = pd.DataFrame({
        "Actual Sales": Y_test.values,
        "Predicted Sales": Y_pred
    })

    print(Result.head())

"""
Function Name : PlotActualVsPredicted
Description   : Displays scatter plot.
Input         : Y_test, Y_pred
Output        : Graph
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def PlotActualVsPredicted(Y_test, Y_pred):

    plt.figure(figsize=(8, 5))

    plt.scatter(Y_test, Y_pred)

    plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    'r--'
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual Sales vs Predicted Sales")

    plt.grid(True)

    plt.show()

"""
Function Name : DisplayFooter
Description   : Displays completion message.
Input         : None
Output        : Footer message.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayFooter():

    print("-----------------------------------------------------------------")
    print("------ Advertising Sales Prediction Completed Successfully ------")
    print("-----------------------------------------------------------------")

"""
Function Name : main
Description   : Controls complete Advertising Sales Prediction workflow.
Input         : None
Output        : Displays model evaluation and graph.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def main():

    Border = "--" * 50

    DataPath = "Advertising.csv"

    DisplayHeader()
    print(Border)

    df = LoadDataset(DataPath)
    print(Border)

    df = RemoveUnwantedColumns(df)
    print(Border)

    CheckMissingValues(df)
    print(Border)

    DisplayStatisticalSummary(df)
    print(Border)

    DisplayCorrelationMatrix(df)
    print(Border)

    X, Y = PrepareData(df)
    print(Border)

    X_train, X_test, Y_train, Y_test = SplitData(X, Y)
    print(Border)

    model = TrainModel(X_train, Y_train)
    print(Border)

    Y_pred = TestModel(model, X_test)
    print(Border)

    EvaluateModel(Y_test, Y_pred)
    print(Border)

    DisplayModelCoefficients(model, X)
    print(Border) 

    CompareActualVsPredicted(Y_test, Y_pred)
    print(Border)

    PlotActualVsPredicted(Y_test, Y_pred)
    print(Border)

    DisplayFooter()

if __name__ == "__main__":
    main()
