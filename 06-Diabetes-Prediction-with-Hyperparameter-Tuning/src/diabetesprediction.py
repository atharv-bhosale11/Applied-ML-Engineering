"""
----------------------------------------------------------------------
Project Name        : Diabetes Prediction with Hyperparameter Tuning

Dataset Information :

Dataset File        : diabetes.csv 

Target Variable     : Outcome

Features            : Pregnancies
                      Glucose
                      BloodPressure
                      SkinThickness
                      Insulin
                      BMI
                      DiabetesPedigreeFunction
                      Age

Machine Learning Information :

Algorithms Used     : Logistic Regression
                      Decision Tree
                      Random Forest
                      K-Nearest Neighbors
                      Support Vector Machine

Techniques Used     : Missing value handling
                      Feature scaling
                      Hyperparameter tuning
                      Cross-validation
                      Model comparison
                      Model saving and loading

Library             : Scikit-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Accuracy Score
                      Precision
                      Recall
                      F1-Score
                      Confusion Matrix

Author              : Atharv Tushar Bhosale

Date                : 13/10/2026
----------------------------------------------------------------------
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


DATA_PATH = "diabetes.csv"
MODEL_PATH = "BestDiabetesModel.pkl"
COMPARISON_PATH = "Algorithm_Comparison.png"
MATRIX_PATH = "Confusion_Matrix.png"
RESULTS_PATH = "Model_Comparison.csv"

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


"""
------------------------------------------------------------
Function Name : DisplayProjectInformation
Description   : Displays the diabetes project information.
Input         : None
Output        : Displays project details.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def DisplayProjectInformation(): 

    print("\nProject Name        : Diabetes Prediction")
    print("Dataset File        : diabetes.csv")
    print("Target Variable     : Outcome")
    print("Classification Type : Binary Classification")

    print("\nAlgorithms Used:")
    print("1. Logistic Regression")
    print("2. Decision Tree")
    print("3. Random Forest")
    print("4. K-Nearest Neighbors")
    print("5. Support Vector Machine")

    print("\nAuthor              : Atharv Tushar Bhosale")


"""
------------------------------------------------------------
Function Name : LoadDataset
Description   : Loads the diabetes dataset from a CSV file.
Input         : DataPath - Path of the dataset.
Output        : Returns the loaded Pandas DataFrame.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def LoadDataset(DataPath):

    if not os.path.exists(DataPath):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(DataPath)}"
        )

    df = pd.read_csv(DataPath)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if "Outcome" not in df.columns:
        raise ValueError(
            "The dataset must contain an Outcome column."
        )

    if df["Outcome"].isnull().any():
        raise ValueError(
            "The Outcome column contains missing values."
        )

    if not set(df["Outcome"].unique()).issubset({0, 1}):
        raise ValueError(
            "The Outcome column must contain only 0 and 1."
        )

    print("\nDataset Loaded Successfully")
    print("Total Records :", df.shape[0])
    print("Total Columns :", df.shape[1])

    return df


"""
------------------------------------------------------------
Function Name : DisplayDatasetInformation
Description   : Displays dataset structure and statistics.
Input         : df - Diabetes dataset DataFrame.
Output        : Displays dataset information and statistics.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def DisplayDatasetInformation(df):

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nTarget Distribution:")
    print(df["Outcome"].value_counts().sort_index())

    print("\nZero Values in Medical Measurements:")

    for Column in INVALID_ZERO_COLUMNS:
        if Column in df.columns:
            print(f"{Column}: {(df[Column] == 0).sum()}")


"""
------------------------------------------------------------
Function Name : ReplaceInvalidZeros
Description   : Replaces zero values in selected medical
                measurement columns with missing values.
Input         : X - Input feature DataFrame.
Output        : Returns the updated DataFrame.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def ReplaceInvalidZeros(X):

    X = X.copy()

    for Column in INVALID_ZERO_COLUMNS:
        if Column in X.columns:
            X.loc[X[Column] == 0, Column] = np.nan

    return X


"""
------------------------------------------------------------
Function Name : PrepareData
Description   : Separates the input features and target variable.
Input         : df - Diabetes dataset DataFrame.
Output        : Returns X containing features and Y containing
                the target variable.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def PrepareData(df):

    X = df.drop(columns=["Outcome"])
    Y = df["Outcome"]

    print("\nInput Features:")
    print(X.columns.tolist())

    print("\nTarget Variable: Outcome")
    print("0 = No Diabetes")
    print("1 = Diabetes")

    print("\nFeature Shape:", X.shape)
    print("Target Shape :", Y.shape)

    return X, Y


"""
------------------------------------------------------------
Function Name : SplitData
Description   : Splits the dataset into training and testing sets.
Input         : X - Input features.
                Y - Target variable.
Output        : Returns XTrain, XTest, YTrain, and YTest.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def SplitData(X, Y):

    XTrain, XTest, YTrain, YTest = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    print("\nTraining Records:", len(XTrain))
    print("Testing Records :", len(XTest))

    return XTrain, XTest, YTrain, YTest


"""
------------------------------------------------------------
Function Name : CreateModels
Description   : Defines five classification algorithms and
                their hyperparameter search spaces.
Input         : None.
Output        : Returns a dictionary containing classifiers,
                scaling requirements, and tuning parameters.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def CreateModels():
    """Define the five classifiers and their tuning parameters."""

    Models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=3000,
                random_state=42
            ),
            "scale": True,
            "parameters": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["liblinear"]
            }
        },

        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=42
            ),
            "scale": False,
            "parameters": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 4, 5, 6, 8, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=42
            ),
            "scale": False,
            "parameters": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2]
            }
        },

        "KNN": {
            "model": KNeighborsClassifier(),
            "scale": True,
            "parameters": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            }
        },

        "SVM": {
            "model": SVC(),
            "scale": True,
            "parameters": {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"]
            }
        }
    }

    return Models


"""
------------------------------------------------------------
Function Name : BuildPipeline
Description   : Creates a machine learning pipeline containing
                preprocessing and a classification algorithm.
Input         : Model - Classification algorithm.
                ScaleFeatures - Indicates whether scaling
                should be applied.
Output        : Returns the configured Scikit-Learn pipeline.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def BuildPipeline(Model, ScaleFeatures):

    Steps = [
        (
            "zero_handler",
            FunctionTransformer(ReplaceInvalidZeros)
        ),
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if ScaleFeatures:
        Steps.append(
            ("scaler", StandardScaler())
        )

    Steps.append(
        ("classifier", clone(Model))
    )

    return Pipeline(Steps)


"""
------------------------------------------------------------
Function Name : TuneAndCompareModels
Description   : Performs hyperparameter tuning using GridSearchCV
                and compares cross-validation accuracy.
Input         : Models - Dictionary of classifiers and parameters.
                XTrain - Training features.
                YTrain - Training target values.
Output        : Returns model comparison results and tuned pipelines.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def TuneAndCompareModels(Models, XTrain, YTrain):

    Results = []
    BestPipelines = {}

    CrossValidation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for ModelName, Configuration in Models.items():

        print("\n" + "-" * 70)
        print("Algorithm:", ModelName)
        print("-" * 70)

        PipelineModel = BuildPipeline(
            Configuration["model"],
            Configuration["scale"]
        )

        Grid = GridSearchCV(
            estimator=PipelineModel,
            param_grid=Configuration["parameters"],
            scoring="accuracy",
            cv=CrossValidation,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        Grid.fit(XTrain, YTrain)

        BestPipelines[ModelName] = Grid.best_estimator_

        Results.append({
            "Algorithm": ModelName,
            "Best CV Accuracy (%)": Grid.best_score_ * 100,
        })

        print(
            f"Best CV Accuracy: {Grid.best_score_ * 100:.2f}%"
        )

    ResultsDF = pd.DataFrame(Results)

    ResultsDF = ResultsDF.sort_values(
        by="Best CV Accuracy (%)",
        ascending=False
    ).reset_index(drop=True)

    print("\nCross-Validation Results:")

    print(
        ResultsDF[
            ["Algorithm", "Best CV Accuracy (%)"]
        ].to_string(
            index=False,
            formatters={
                "Best CV Accuracy (%)": lambda Value: f"{Value:.2f}%"
            }
        )
    )

    return ResultsDF, BestPipelines


"""
------------------------------------------------------------
Function Name : PlotModelComparison
Description   : Creates and saves a bar chart comparing the
                cross-validation accuracy of the algorithms.
Input         : ResultsDF - DataFrame containing model results.
Output        : Displays and saves the comparison chart.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def PlotModelComparison(ResultsDF):

    PlotDF = ResultsDF.sort_values(
        by="Best CV Accuracy (%)",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        PlotDF["Algorithm"],
        PlotDF["Best CV Accuracy (%)"]
    )

    plt.xlabel("Best Cross-Validation Accuracy (%)")
    plt.ylabel("Classification Algorithm")
    plt.title("Diabetes Prediction - Algorithm Comparison")

    plt.tight_layout()
    plt.savefig(
        COMPARISON_PATH,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


"""
------------------------------------------------------------
Function Name : EvaluateBestModel
Description   : Evaluates the selected classifier using accuracy,
                precision, recall, F1-score, and confusion matrix.
Input         : BestModel - Selected trained classification pipeline.
                XTest - Testing features.
                YTest - Actual testing target values.
Output        : Returns evaluation metrics and displays results.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def EvaluateBestModel(BestModel, XTest, YTest):

    YPred = BestModel.predict(XTest)

    Accuracy = accuracy_score(YTest, YPred)
    Precision = precision_score(
        YTest,
        YPred,
        zero_division=0
    )
    Recall = recall_score(
        YTest,
        YPred,
        zero_division=0
    )
    F1Score = f1_score(
        YTest,
        YPred,
        zero_division=0
    )

    print(f"Test Accuracy : {Accuracy * 100:.2f}%")
    print(f"Precision     : {Precision:.4f}")
    print(f"Recall        : {Recall:.4f}")
    print(f"F1-Score      : {F1Score:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            YTest,
            YPred,
            target_names=["No Diabetes", "Diabetes"],
            zero_division=0
        )
    )

    ConfusionMatrix = confusion_matrix(
        YTest,
        YPred
    )

    print("Confusion Matrix:")
    print(ConfusionMatrix)

    Display = ConfusionMatrixDisplay(
        confusion_matrix=ConfusionMatrix,
        display_labels=["No Diabetes", "Diabetes"]
    )

    Display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Best Model - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return {
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Recall": Recall,
        "F1-Score": F1Score
    }


"""
------------------------------------------------------------
Function Name : SaveBestModel
Description   : Saves the selected trained classification
                pipeline to a Pickle file.
Input         : BestModel - Trained classification pipeline.
Output        : Saves the model and displays its file path.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def SaveBestModel(BestModel):

    joblib.dump(
        BestModel,
        MODEL_PATH
    )

    print("\nBest Model Saved Successfully")
    print("Model Path:", os.path.abspath(MODEL_PATH))


"""
------------------------------------------------------------
Function Name : DisplayProjectSummary
Description   : Displays the selected algorithm and its
                cross-validation and test evaluation metrics.
Input         : BestAlgorithm - Name of the selected algorithm.
                CVAccuracy - Best cross-validation accuracy.
                Metrics - Dictionary containing test metrics.
Output        : Displays the final project summary.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def DisplayProjectSummary(BestAlgorithm, CVAccuracy, Metrics):

    print("\nProject Name       : Diabetes Prediction with Hyperparameter Tuning")
    print(f"Best Algorithm     : ", BestAlgorithm)
    print(f"Best Accuracy       : {CVAccuracy:.2f}%")
    print(f"Test Accuracy       : {Metrics['Accuracy'] * 100:.2f}%")
    print(f"Test Precision      : {Metrics['Precision']:.4f}")
    print(f"Test Recall         : {Metrics['Recall']:.4f}")
    print(f"Test F1-Score       : {Metrics['F1-Score']:.4f}")


"""
------------------------------------------------------------
Function Name : main
Description   : Controls the complete diabetes prediction
                workflow, from loading the dataset to
                evaluating and saving the best model.
Input         : None.
Output        : Executes the project workflow and displays
                the final results.
Author        : Atharv Tushar Bhosale
Date          : 13/10/2026
------------------------------------------------------------
"""
def main():
    """Control the complete diabetes prediction workflow."""

    Border = "=" * 70

    print(Border)
    print("STEP 1: PROJECT INFORMATION")
    print(Border)

    DisplayProjectInformation()

    print("\n" + Border)
    print("STEP 2: LOAD DATASET")
    print(Border)

    df = LoadDataset(DATA_PATH)

    DisplayDatasetInformation(df)

    print("\n" + Border)
    print("STEP 3: PREPARE DATA")
    print(Border)

    X, Y = PrepareData(df)

    XTrain, XTest, YTrain, YTest = SplitData(X, Y)

    print("\n" + Border)
    print("STEP 4: CROSS-VALIDATION AND MODEL COMPARISON")
    print(Border)

    Models = CreateModels()

    ResultsDF, BestPipelines = TuneAndCompareModels(
        Models,
        XTrain,
        YTrain
    )

    ResultsDF.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\nModel comparison saved to:", os.path.abspath(RESULTS_PATH))

    PlotModelComparison(ResultsDF)

    BestAlgorithm = ResultsDF.iloc[0]["Algorithm"]
    BestModel = BestPipelines[BestAlgorithm]

    BestCVAccuracy = ResultsDF.iloc[0]["Best CV Accuracy (%)"]

    print("\n" + Border)
    print("BEST MODEL SELECTED")
    print(Border)

    print("Algorithm:", BestAlgorithm)
    print(f"Best CV Accuracy: {BestCVAccuracy:.2f}%")

    print("\n" + Border)
    print("STEP 5: EVALUATE BEST MODEL")
    print(Border)

    Metrics = EvaluateBestModel(
        BestModel,
        XTest,
        YTest
    )

    print("\n" + Border)
    print("STEP 6: SAVE BEST MODEL")
    print(Border)

    SaveBestModel(BestModel)

    print("\n" + Border)
    print("STEP 7: FINAL PROJECT SUMMARY")
    print(Border)

    DisplayProjectSummary(
        BestAlgorithm,
        BestCVAccuracy,
        Metrics
    )

    print("\n" + Border)
    print("DIABETES PREDICTION PROJECT COMPLETED SUCCESSFULLY")
    print(Border)


if __name__ == "__main__":
    main()
