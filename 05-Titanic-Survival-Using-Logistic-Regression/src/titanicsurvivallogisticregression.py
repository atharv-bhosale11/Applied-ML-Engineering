"""
------------------------------------------------------------
Project Name        : Titanic Survival Prediction 

Dataset Information :

Dataset File        : Titanic.csv

Target Variable     : Survived

Features            : Age
                      Fare
                      Sex
                      sibsp
                      Parch
                      Pclass
                      Embarked

Machine Learning Information :

Algorithm Used      : Logistic Regression

Techniques Used     : Missing value handling
                      One-hot encoding of Embarked
                      Model saving and loading

Library             : Scikit-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Accuracy Score
                      Confusion Matrix

Author              : Atharv Tushar Bhosale

Date                : 10/10/2026
------------------------------------------------------------
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

import joblib

import matplotlib.pyplot as plt
import seaborn as sns


"""
Function Name : DisplayProjectInformation
Description   : Displays project information.
Input         : None
Output        : Project details.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayProjectInformation():

    print("------------------------------------------------------------")
    print("Project Name        : Titanic Survival Prediction")
    print("Dataset File        : Titanic.csv")
    print("Algorithm Used      : Logistic Regression")
    print("Problem Type        : Binary Classification")
    print("Author              : Atharv Tushar Bhosale")
    print("------------------------------------------------------------")


"""
Function Name : LoadDataset
Description   : Loads Titanic dataset from CSV file.
Input         : Dataset Path
Output        : DataFrame
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def LoadDataset(DataPath):

    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print("Total Records :",df.shape[0])
    print("Total Columns :",df.shape[1])

    return df


"""
Function Name : DisplayDatasetInformation
Description   : Displays basic information about the dataset.
Input         : DataFrame
Output        : Dataset information.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayDatasetInformation(df):

    print("\nFirst Five Rows of Dataset:")
    print(df.head())

    print("\nShape of Dataset:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    print(df.info())


"""
Function Name : CheckMissingValues
Description   : Displays missing values present in each column.
Input         : DataFrame
Output        : Missing value report.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def CheckMissingValues(df):

    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())


"""
Function Name : DisplayStatisticalSummary
Description   : Displays statistical summary of numerical columns.
Input         : DataFrame
Output        : Statistical summary.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayStatisticalSummary(df):

    print("\nStatistical Summary:")
    print(df.describe())


"""
Function Name : CleanDataset
Description   : Performs preprocessing on Titanic dataset.
                Removes unnecessary columns, handles missing
                values and encodes categorical data.
Input         : DataFrame
Output        : Cleaned DataFrame
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def CleanDataset(df):

    """
    Remove unnecessary columns
    """

    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\nColumns to be dropped:")
    print(existing_columns)

    df = df.drop(columns=existing_columns)


    """
    Handle Age Column
    """

    if "Age" in df.columns:

        print("\nAge Column before Processing:")
        print(df["Age"].head(10))

        df["Age"] = pd.to_numeric(df["Age"],errors="coerce")

        age_median = df["Age"].median()

        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge Column after Processing:")
        print(df["Age"].head(10))


    """
    Handle Fare Column
    """

    if "Fare" in df.columns:

        print("\nFare Column before Processing:")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"],errors="coerce")

        fare_median = df["Fare"].median()

        print("\nMedian of Fare Column:",fare_median)

        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare Column after Processing:")
        print(df["Fare"].head(10))


    """
    Handle Embarked Column
    """

    if "Embarked" in df.columns:

        print("\nEmbarked Column before Processing:")
        print(df["Embarked"].head(10))

        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        df["Embarked"] = df["Embarked"].replace(
            ['nan','None',''],
            np.nan
        )

        embarked_mode = df["Embarked"].mode()[0]

        print("\nMode of Embarked Column:",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\nEmbarked Column after Processing:")
        print(df["Embarked"].head(10))


    """
    Handle Sex Column
    """

    if "Sex" in df.columns:

        print("\nSex Column before Processing:")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(
            df["Sex"],
            errors="coerce"
        )

        print("\nSex Column after Processing:")
        print(df["Sex"].head(10))


    """
    Encode Embarked Column
    """

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True
    )


    """
    Convert Boolean Columns into Integers
    """

    for col in df.columns:

        if df[col].dtype == bool:

            df[col] = df[col].astype(int)


    print("\nData After Pre-Processing:")
    print(df.head())

    print("\nShape of Dataset:",df.shape)

    print("\nMissing Values After Pre-Processing:")
    print(df.isnull().sum())

    return df


"""
Function Name : PrepareData
Description   : Separates independent features and target variable.
Input         : DataFrame
Output        : X and Y
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def PrepareData(df):

    X = df.drop("Survived",axis=1)
    Y = df["Survived"]

    print("\nFeatures:")
    print(X.head())

    print("\nLabels:")
    print(Y.head())

    print("\nShape of X:",X.shape)
    print("Shape of Y:",Y.shape)

    return X,Y


"""
Function Name : SplitData
Description   : Splits dataset into training and testing data.
Input         : X and Y
Output        : X_train, X_test, Y_train, Y_test
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def SplitData(X,Y):

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("\nX_train Shape:",X_train.shape)
    print("X_test Shape :",X_test.shape)
    print("Y_train Shape:",Y_train.shape)
    print("Y_test Shape :",Y_test.shape)

    return X_train,X_test,Y_train,Y_test


"""
Function Name : TrainModel
Description   : Trains Logistic Regression model.
Input         : X_train, Y_train
Output        : Trained Model
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def TrainModel(X_train,Y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("\nModel Trained Successfully")

    return model


"""
Function Name : DisplayModelCoefficients
Description   : Displays model intercept and coefficients.
Input         : Model, Feature Names
Output        : Model coefficients.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayModelCoefficients(model,FeatureNames):

    print("\nIntercept of Model:")
    print(model.intercept_)

    print("\nCoefficient of Model:")

    for feature,coefficient in zip(
        FeatureNames,
        model.coef_[0]
    ):

        print(feature," : ",coefficient)


"""
Function Name : SaveModel
Description   : Saves trained machine learning model using joblib.
Input         : Model, Filename
Output        : Saved model file.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def SaveModel(model,filename):

    joblib.dump(model,filename)

    print(
        "\nModel Saved Successfully with Name:",
        filename
    )


"""
Function Name : LoadModel
Description   : Loads previously saved machine learning model.
Input         : Filename
Output        : Loaded trained model.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def LoadModel(filename):

    loaded_model = joblib.load(filename)

    print("\nModel Loaded Successfully")

    return loaded_model


"""
Function Name : GeneratePredictions
Description   : Generates predictions using trained model.
Input         : Model, X_test
Output        : Predicted values.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def GeneratePredictions(model,X_test):

    Y_pred = model.predict(X_test)

    print("\nPredictions Generated Successfully")

    return Y_pred


"""
Function Name : CalculateAccuracy
Description   : Calculates model accuracy.
Input         : Y_test, Y_pred
Output        : Accuracy.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def CalculateAccuracy(Y_test,Y_pred):

    Accuracy = accuracy_score(
        Y_test,
        Y_pred
    )

    print(
        f"\nModel Accuracy : {Accuracy * 100:.2f}%"
    )

    return Accuracy


"""
Function Name : DisplayConfusionMatrix
Description   : Displays graphical confusion matrix.
Input         : Y_test, Y_pred
Output        : Confusion Matrix Graph.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayConfusionMatrix(Y_test,Y_pred):

    cm = confusion_matrix(
        Y_test,
        Y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,5))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    classes = sorted(Y_test.unique())

    plt.xticks(
        range(len(classes)),
        classes
    )

    plt.yticks(
        range(len(classes)),
        classes
    )

    for i in range(len(cm)):

        for j in range(len(cm[0])):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


"""
Function Name : ShowGraphs
Description   : Displays graphical representation of the dataset
                including survival count and age distribution.
Input         : df - Titanic dataset
Output        : Graphs.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def ShowGraphs(df):

    """
    Survival Count Graph
    """

    plt.figure(figsize=(6,4))

    sns.countplot(
        x="Survived",
        data=df
    )

    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")

    plt.show()


    """
    Age Distribution
    """

    plt.figure(figsize=(6,4))

    sns.histplot(
        df["Age"],
        bins=20,
        kde=True
    )

    plt.title("Age Distribution of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Frequency")

    plt.show()


"""
Function Name : DisplayProjectSummary
Description   : Displays final project summary.
Input         : Accuracy
Output        : Project summary.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayProjectSummary(Accuracy):

    print("-" * 60)

    print("Project Summary")

    print("-" * 60)

    print(
        "Algorithm Used : Logistic Regression"
    )

    print(
        f"Accuracy       : {Accuracy * 100:.2f}%"
    )

    print(
        "Problem Type   : Binary Classification"
    )

    print("-" * 60)


"""
Function Name : DisplayFooter
Description   : Displays completion message.
Input         : None
Output        : Footer Message.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def DisplayFooter():

    print("-" * 100)

    print(
        "Titanic Survival Prediction Using Logistic Regression Completed Successfully"
    )

    print("-" * 100)


"""
Function Name : main
Description   : Controls complete Titanic Survival Prediction workflow.
Input         : None
Output        : Displays model evaluation and results.
Author        : Atharv Tushar Bhosale
Date          : 10/10/2026
"""

def main():

    Border = "-" * 100

    DataPath = "Titanic.csv"


    print(Border)
    print("Step 1 : Project Information")
    print(Border)

    DisplayProjectInformation()


    print(Border)
    print("Step 2 : Load Dataset")
    print(Border)

    df = LoadDataset(DataPath)


    print(Border)
    print("Step 3 : Display Dataset Information")
    print(Border)

    DisplayDatasetInformation(df)


    print(Border)
    print("Step 4 : Check Missing Values")
    print(Border)

    CheckMissingValues(df)


    print(Border)
    print("Step 5 : Display Statistical Summary")
    print(Border)

    DisplayStatisticalSummary(df)


    print(Border)
    print("Step 6 : Clean Dataset")
    print(Border)

    df = CleanDataset(df)


    print(Border)
    print("Step 7 : Prepare Features and Target")
    print(Border)

    X,Y = PrepareData(df)


    print(Border)
    print("Step 8 : Split Dataset")
    print(Border)

    X_train,X_test,Y_train,Y_test = SplitData(X,Y)


    print(Border)
    print("Step 9 : Train Logistic Regression Model")
    print(Border)

    model = TrainModel(X_train,Y_train)


    print(Border)
    print("Step 10 : Display Model Coefficients")
    print(Border)

    DisplayModelCoefficients(
        model,
        X.columns
    )


    print(Border)
    print("Step 11 : Save Model")
    print(Border)

    SaveModel(
        model,
        "TitanicLogisticRegression.pkl"
    )


    print(Border)
    print("Step 12 : Load Saved Model")
    print(Border)

    loaded_model = LoadModel(
        "TitanicLogisticRegression.pkl"
    )


    print(Border)
    print("Step 13 : Generate Predictions")
    print(Border)

    Y_pred = GeneratePredictions(
        loaded_model,
        X_test
    )


    print(Border)
    print("Step 14 : Calculate Accuracy")
    print(Border)

    Accuracy = CalculateAccuracy(
        Y_test,
        Y_pred
    )


    print(Border)
    print("Step 15 : Display Confusion Matrix")
    print(Border)

    DisplayConfusionMatrix(
        Y_test,
        Y_pred
    )

    print(Border)
    print("Step 16 : Display Dataset Graphs")
    print(Border)

    ShowGraphs(df)

    print(Border)
    print("Step 17 : Display Project Summary")
    print(Border)

    DisplayProjectSummary(Accuracy)

    DisplayFooter()

if __name__ == "__main__":

    main()
