"""
------------------------------------------------------------
Project Name        : Wine Classification Using KNN

Dataset Information :

Dataset File        : WinePredictor.csv

Total Samples       : 178

Total Features      : 13

Target Classes      : Class 1
                      Class 2
                      Class 3

Features            : Alcohol
                      Malic Acid
                      Ash 
                      Alcalinity of Ash
                      Magnesium
                      Total Phenols
                      Flavanoids
                      Nonflavanoid Phenols
                      Proanthocyanins
                      Color Intensity
                      Hue
                      OD280/OD315 of Diluted Wines
                      Proline

Target Variable     : Wine Class

Machine Learning Information :

Algorithm Used      : K-Nearest Neighbors (KNN)

Library             : Scikit-Learn

Problem Type        : Multi-Class Classification

Evaluation Metrics  : Accuracy Score
                      Confusion Matrix
                      Classification Report

Author              : Atharv Tushar Bhosale

Date                : 07/10/2026
------------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
    )

"""
Function Name : DisplayProjectInformation
Description   : Displays project information.
Input         : None
Output        : Project details.
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayProjectInformation():

    print("------------------------------------------------------------")
    print("Project Name        : Wine Classification Using KNN")
    print("Dataset File        : WinePredictor.csv")
    print("Algorithm Used      : K-Nearest Neighbors (KNN)")
    print("Problem Type        : Multi-Class Classification")
    print("Author              : Atharv Tushar Bhosale")
    print("------------------------------------------------------------")

"""
Function Name : LoadDataset
Description   : Loads dataset from CSV file.
Input         : Dataset Path
Output        : DataFrame
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def LoadDataset(DataPath):

    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print("Total Records :", df.shape[0])
    print("Total Columns :", df.shape[1])

    return df

"""
Function Name : CleanDataset
Description   : Removes empty rows from dataset.
Input         : DataFrame
Output        : Cleaned DataFrame
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def CleanDataset(df):

    df.dropna(inplace=True)

    print("Dataset Cleaned Successfully")
    print("Total Records :", df.shape[0])
    print("Total Columns :", df.shape[1])

    return df

"""
Function Name : DisplayDatasetInformation
Description   : Displays dataset information.
Input         : DataFrame
Output        : Dataset information.
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayDatasetInformation(df):

    print(df.info())

"""
Function Name : CheckMissingValues
Description   : Displays missing values.
Input         : DataFrame
Output        : Missing value report.
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def CheckMissingValues(df):

    print(df.isnull().sum())

"""
Function Name : DisplayStatisticalSummary
Description   : Displays statistical summary.
Input         : DataFrame
Output        : Statistical report.
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayStatisticalSummary(df):

    print(df.describe())

"""
Function Name : PrepareData
Description   : Separates features and target.
Input         : DataFrame
Output        : X and Y
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def PrepareData(df):

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X :", X.shape)
    print("Shape of Y :", Y.shape)

    return X, Y

"""
Function Name : SplitData
Description   : Splits dataset into training and testing data.
Input         : X and Y
Output        : X_train, X_test, Y_train, Y_test
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def SplitData(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    return X_train, X_test, Y_train, Y_test

"""
Function Name : PerformFeatureScaling
Description   : Performs feature scaling using StandardScaler.
Input         : X_train, X_test
Output        : X_train_scaled, X_test_scaled
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def PerformFeatureScaling(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)

    print("Feature Scaling Completed Successfully")

    return X_train_scaled, X_test_scaled

"""
Function Name : ExploreKValues
Description   : Evaluates K values from 1 to 20.
Input         : X_train_scaled, X_test_scaled,
                Y_train, Y_test
Output        : K_values, AccuracyScores
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def ExploreKValues(X_train_scaled,
                   X_test_scaled,
                   Y_train,
                   Y_test):

    AccuracyScores = []

    K_values = range(1, 21)

    for k in K_values:

        model = KNeighborsClassifier(
            n_neighbors = k
        )

        model.fit(X_train_scaled, Y_train)

        Y_pred = model.predict(X_test_scaled)

        Accuracy = accuracy_score(
            Y_test,
            Y_pred
        )

        AccuracyScores.append(Accuracy)

    print("Accuracy Report")

    for k, score in zip(K_values, AccuracyScores):

        print(f"K = {k} --> {score * 100:.2f}%")

    return K_values, AccuracyScores

"""
Function Name : PlotKVsAccuracy
Description   : Displays graph of K vs Accuracy.
Input         : K_values, AccuracyScores
Output        : Graph
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def PlotKVsAccuracy(K_values, AccuracyScores):

    plt.figure(figsize=(8,5))

    plt.plot(
        K_values,
        AccuracyScores,
        marker='o'
    )

    plt.title("K Value vs Accuracy")

    plt.xlabel("K Value")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.xticks(K_values)

    plt.show()

"""
Function Name : FindBestK
Description   : Finds best K value.
Input         : K_values, AccuracyScores
Output        : Best K
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def FindBestK(K_values, AccuracyScores):

    BestK = list(K_values)[
        AccuracyScores.index(
            max(AccuracyScores)
        )
    ]

    print("Best K Value :", BestK)

    return BestK

"""
Function Name : TrainFinalModel
Description   : Trains final KNN model.
Input         : X_train_scaled,
                Y_train,
                BestK
Output        : Final Model
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def TrainFinalModel(
    X_train_scaled,
    Y_train,
    BestK
):

    model = KNeighborsClassifier(
        n_neighbors = BestK
    )

    model.fit(
        X_train_scaled,
        Y_train
    )

    print("Final Model Trained Successfully")

    return model

"""
Function Name : GeneratePredictions
Description   : Generates predictions using trained model.
Input         : Model, X_test_scaled
Output        : Predicted Values
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def GeneratePredictions(model, X_test_scaled):

    Y_pred = model.predict(X_test_scaled)

    print("Predictions Generated Successfully")

    return Y_pred

"""
Function Name : CalculateAccuracy
Description   : Calculates model accuracy.
Input         : Y_test, Y_pred
Output        : Accuracy
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def CalculateAccuracy(Y_test, Y_pred):

    Accuracy = accuracy_score(
        Y_test,
        Y_pred
    )

    print(
        f"Model Accuracy : {Accuracy * 100:.2f}%"
    )

    return Accuracy

"""
Function Name : DisplayConfusionMatrix
Description   : Displays graphical confusion matrix.
Input         : Y_test, Y_pred
Output        : Heatmap Graph
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayConfusionMatrix(Y_test, Y_pred):

    cm = confusion_matrix(Y_test, Y_pred)

    plt.figure(figsize=(6, 5))

    plt.imshow(cm, cmap='Blues')

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    classes = sorted(Y_test.unique())

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(
                j,
                i,
                cm[i, j],
                ha='center',
                va='center'
            )

    plt.tight_layout()
    plt.show()

"""
Function Name : DisplayClassificationReport
Description   : Displays classification report.
Input         : Y_test, Y_pred
Output        : Classification Report
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayClassificationReport(
    Y_test,
    Y_pred
    ):

    print(
        classification_report(
            Y_test,
            Y_pred
        )
    )

"""
Function Name : DisplayProjectSummary
Description   : Displays final project summary.
Input         : Accuracy, BestK
Output        : Summary
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayProjectSummary(
    Accuracy,
    BestK
):

    print("-" * 60)

    print("Project Summary")

    print("-" * 60)

    print(
        "Algorithm Used : K-Nearest Neighbors"
    )

    print(
        "Best K Value   :",
        BestK
    )

    print(
        f"Accuracy       : {Accuracy * 100:.2f}%"
    )

    print(
        "Problem Type   : Multi-Class Classification"
    )

    print("-" * 60)

"""
Function Name : DisplayFooter
Description   : Displays completion message.
Input         : None
Output        : Footer Message
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def DisplayFooter():

    print("-" * 100)

    print(
        "Wine Classification Using KNN Completed Successfully"
    )

    print("-" * 100)

"""
Function Name : main
Description   : Controls complete Wine Classification workflow.
Input         : None
Output        : Displays model evaluation and results.
Author        : Atharv Tushar Bhosale
Date          : 07/10/2026
"""

def main():

    Border = "-" * 100

    DataPath = "WinePredictor.csv"

    print(Border)
    print("Step 1 : Project Information")
    print(Border)
    DisplayProjectInformation()

    print(Border)
    print("Step 2 : Load Dataset")
    print(Border)
    df = LoadDataset(DataPath)

    print(Border)
    print("Step 3 : Clean Dataset")
    print(Border)
    df = CleanDataset(df)

    print(Border)
    print("Step 4 : Display Dataset Information")
    print(Border)
    DisplayDatasetInformation(df)

    print(Border)
    print("Step 5 : Check Missing Values")
    print(Border)
    CheckMissingValues(df)

    print(Border)
    print("Step 6 : Display Statistical Summary")
    print(Border)
    DisplayStatisticalSummary(df)

    print(Border)
    print("Step 7 : Prepare Features and Target")
    print(Border)
    X, Y = PrepareData(df)

    print(Border)
    print("Step 8 : Split Dataset")
    print(Border)
    X_train, X_test, Y_train, Y_test = SplitData(X, Y)

    print(Border)
    print("Step 9 : Perform Feature Scaling")
    print(Border)
    X_train_scaled, X_test_scaled = PerformFeatureScaling(
        X_train,
        X_test
    )

    print(Border)
    print("Step 10 : Explore Multiple K Values")
    print(Border)
    K_values, AccuracyScores = ExploreKValues(
        X_train_scaled,
        X_test_scaled,
        Y_train,
        Y_test
    )

    print(Border)
    print("Step 11 : Plot K vs Accuracy Graph")
    print(Border)
    PlotKVsAccuracy(
        K_values,
        AccuracyScores
    )

    print(Border)
    print("Step 12 : Find Best K Value")
    print(Border)
    BestK = FindBestK(
        K_values,
        AccuracyScores
    )

    print(Border)
    print("Step 13 : Train Final KNN Model")
    print(Border)
    model = TrainFinalModel(
        X_train_scaled,
        Y_train,
        BestK
    )

    print(Border)
    print("Step 14 : Generate Predictions")
    print(Border)
    Y_pred = GeneratePredictions(
        model,
        X_test_scaled
    )

    print(Border)
    print("Step 15 : Calculate Accuracy")
    print(Border)
    Accuracy = CalculateAccuracy(
        Y_test,
        Y_pred
    )

    print(Border)
    print("Step 16 : Display Confusion Matrix")
    print(Border)
    DisplayConfusionMatrix(
        Y_test,
        Y_pred
    )

    print(Border)
    print("Step 17 : Display Classification Report")
    print(Border)
    DisplayClassificationReport(
        Y_test,
        Y_pred
    )

    print(Border)
    print("Step 18 : Display Project Summary")
    print(Border)
    DisplayProjectSummary(
        Accuracy,
        BestK
    )

    DisplayFooter()

if __name__ == "__main__":
    main()

