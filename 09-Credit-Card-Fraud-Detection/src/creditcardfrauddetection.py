"""
========================================================================
Project Name : Credit Card Fraud Detection
Description  : Detect fraudulent credit card transactions using
               Machine Learning classification algorithms.
Dataset      : Credit Card Fraud Detection Dataset
Algorithms   : Logistic Regression, Decision Tree, Random Forest,
               Gradient Boosting Classifier, Support Vector Machine
Evaluation   : Average Precision, Precision, Recall, F1-Score,
               Confusion Matrix, and ROC-AUC
Author       : Atharv Tushar Bhosale
Date         : 17/10/2026
========================================================================
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from sklearn.base import clone

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

"""
----------------------------------------------------------------------
Function Name : DisplayDatasetInfo
Description   : Displays dataset structure, data types, missing values,
                duplicate records, and statistical information.
Input         : pandas DataFrame
Output        : Displays dataset information
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def DisplayDatasetInfo(dfDataset):


    print("\nDataset Shape:", dfDataset.shape)

    print("\nColumn Names:")
    print(dfDataset.columns.tolist())

    print("\nDataset Information:")
    dfDataset.info()

    print("\nMissing Values:")
    print(dfDataset.isnull().sum())

    print("\nDuplicate Records:", dfDataset.duplicated().sum())

    print("\nStatistical Summary:")
    print(dfDataset.describe().T)

    print("\nClass Distribution:")
    print(dfDataset["Class"].value_counts())

    print("\nClass Distribution Percentage:")
    print(dfDataset["Class"].value_counts(normalize=True).mul(100).round(3))

"""
----------------------------------------------------------------------
Function Name : ValidateDataset
Description   : Validates and cleans the credit card transaction dataset.
Input         : pandas DataFrame
Output        : Cleaned pandas DataFrame
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def ValidateDataset(dfDataset):
  

    if dfDataset.empty:
        raise ValueError("Dataset is empty.")

    if "Class" not in dfDataset.columns:
        raise ValueError("Required target column 'Class' is missing.")

    if dfDataset["Class"].isnull().any() or not set(dfDataset["Class"].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain non-null binary labels (0 and 1).")

    if not dfDataset.drop(columns="Class").select_dtypes(include=np.number).shape[1] == len(dfDataset.columns) - 1:
        raise ValueError("All input features must be numeric.")

    print("Missing Values:", dfDataset.isnull().sum().sum())

    dfDataset = dfDataset.dropna().drop_duplicates().reset_index(drop=True)

    print("Dataset validation completed successfully.")
    print("Final Dataset Shape:", dfDataset.shape)
    print("\nUpdated Class Distribution:")
    print(dfDataset["Class"].value_counts())

    return dfDataset

"""
----------------------------------------------------------------------
Function Name : PerformEDA
Description   : Analyzes class imbalance and visualizes transaction
                amount and time distributions by fraud class.
Input         : pandas DataFrame
Output        : Displays class distribution and saves EDA plots
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def PerformEDA(dfDataset):


    os.makedirs("results", exist_ok=True)

    print("\nClass Distribution:")
    print(dfDataset["Class"].value_counts())

    print("\nClass Percentage:")
    print(dfDataset["Class"].value_counts(normalize=True).mul(100).round(3))

    fig, Axes = plt.subplots(1, 3, figsize=(18, 5))

    dfDataset["Class"].value_counts().sort_index().plot(
        kind="bar", ax=Axes[0], title="Transaction Class Distribution"
    )
    Axes[0].set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    Axes[0].set_ylabel("Number of Transactions")

    for iClass, sLabel in [(0, "Legitimate"), (1, "Fraud")]:
        dfDataset.loc[dfDataset["Class"] == iClass, "Amount"].plot(
            kind="hist", bins=50, alpha=0.6, ax=Axes[1], label=sLabel
        )
        dfDataset.loc[dfDataset["Class"] == iClass, "Time"].plot(
            kind="hist", bins=50, alpha=0.6, ax=Axes[2], label=sLabel
        )

    Axes[1].set_title("Transaction Amount Distribution")
    Axes[1].set_xlabel("Amount")
    Axes[1].legend()

    Axes[2].set_title("Transaction Time Distribution")
    Axes[2].set_xlabel("Time (seconds)")
    Axes[2].legend()

    plt.tight_layout()
    plt.savefig("results/EDA_Analysis.png", dpi=300)
    plt.show()
    plt.close()

    print("\nEDA plot saved: results/EDA_Analysis.png")

"""
----------------------------------------------------------------------
Function Name : PrepareData
Description   : Separates input features and target variable.
Input         : pandas DataFrame
Output        : Feature DataFrame (X) and target Series (y)
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def PrepareData(dfDataset):
 

    X = dfDataset.drop(columns="Class")
    y = dfDataset["Class"]

    print("Feature Shape:", X.shape)
    print("Target Shape:", y.shape)
    print("\nTarget Distribution:")
    print(y.value_counts())

    return X, y

"""
----------------------------------------------------------------------
Function Name : SplitData
Description   : Splits the dataset into stratified training and testing
                sets while preserving the fraud class distribution.
Input         : Feature DataFrame (X), Target Series (y)
Output        : X_train, X_test, y_train, y_test
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def SplitData(X, y):
    """Split features and target using stratified sampling."""


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training Features:", X_train.shape)
    print("Testing Features :", X_test.shape)

    print("\nTraining Class Distribution:")
    print(y_train.value_counts())

    print("\nTesting Class Distribution:")
    print(y_test.value_counts())

    print("\nTraining Fraud Percentage:", round(y_train.mean() * 100, 3), "%")
    print("Testing Fraud Percentage :", round(y_test.mean() * 100, 3), "%")

    return X_train, X_test, y_train, y_test

"""
----------------------------------------------------------------------
Function Name : CreatePreprocessingPipeline
Description   : Creates a preprocessing pipeline using RobustScaler
                for transaction amount and time features.
Input         : None
Output        : Configured ColumnTransformer
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def CreatePreprocessingPipeline():


    Preprocessor = ColumnTransformer(
        transformers=[
            ("Scaler", RobustScaler(), ["Amount", "Time"])
        ],
        remainder="passthrough"
    )


    return Preprocessor

"""
----------------------------------------------------------------------
Function Name : CreateLogisticRegressionModel
Description   : Creates a Logistic Regression pipeline for fraud
                detection with class imbalance handling.
Input         : Preprocessing pipeline
Output        : Configured Machine Learning pipeline
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def CreateLogisticRegressionModel(Preprocessor):


    Model = Pipeline([
        ("Preprocessor", Preprocessor),
        ("Classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Logistic Regression pipeline created successfully.")

    return Model

"""
----------------------------------------------------------------------
Function Name : TrainModel
Description   : Trains a Machine Learning pipeline and measures
                the time required for training.
Input         : Machine Learning pipeline, X_train, y_train
Output        : Trained Machine Learning pipeline and training time
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def TrainModel(Model, X_train, y_train):

    StartTime = time.perf_counter()

    Model.fit(X_train, y_train)

    TrainingTime = time.perf_counter() - StartTime

    print("Model training completed successfully.")
    print(f"Training Time: {TrainingTime:.2f} seconds")

    return Model, TrainingTime

"""
----------------------------------------------------------------------
Function Name : EvaluateModel
Description   : Evaluates the trained model using classification
                metrics and displays the confusion matrix.
Input         : Trained model, X_test, y_test
Output        : Dictionary containing evaluation metrics
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def EvaluateModel(Model, X_test, y_test):

    yPred = Model.predict(X_test)
    if hasattr(Model, "predict_proba"):
        yScore = Model.predict_proba(X_test)[:, 1]
    else:
        yScore = Model.decision_function(X_test)
    iPrecision = precision_score(y_test, yPred, zero_division=0)
    iRecall = recall_score(y_test, yPred, zero_division=0)
    iF1Score = f1_score(y_test, yPred, zero_division=0)
    fAveragePrecision = average_precision_score(y_test, yScore)
    fRocAuc = roc_auc_score(y_test, yScore)
    iAccuracy = accuracy_score(y_test, yPred)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"Precision        : {iPrecision:.4f}")
    print(f"Recall           : {iRecall:.4f}")
    print(f"F1-Score         : {iF1Score:.4f}")
    print(f"Average Precision: {fAveragePrecision:.4f}")
    print(f"ROC-AUC          : {fRocAuc:.4f}")
    print(f"Accuracy         : {iAccuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, yPred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        yPred,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues"
    )

    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/LogisticRegression_ConfusionMatrix.png", dpi=300)
    plt.show()
    plt.close()

    print("\nConfusion matrix saved to results/LogisticRegression_ConfusionMatrix.png")

    return {
        "Precision": iPrecision,
        "Recall": iRecall,
        "F1_Score": iF1Score,
        "Average_Precision": fAveragePrecision,
        "ROC_AUC": fRocAuc
    }

"""
----------------------------------------------------------------------
Function Name : CreateModels
Description   : Creates Decision Tree, Random Forest, Gradient
                Boosting, and Linear SVM classification models.
Input         : None
Output        : Dictionary containing classification models
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def CreateModels():
    """Create additional classification models."""

    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        )
    }

    print("Classification models created successfully.")

    for name in models:
        print(f"Model Created: {name}")

    return models

"""
----------------------------------------------------------------------
Function Name : TrainAndEvaluateModels
Description   : Trains and evaluates multiple classification models
                and records their performance metrics.
Input         : Models dictionary, X_train, y_train, X_test, y_test
Output        : Dictionary containing trained models and their metrics
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def TrainAndEvaluateModels(Models, X_train, y_train, X_test, y_test):

    TrainedModels = {}
    ModelResults = {}

    for Name, Classifier in Models.items():

        print("\n" + "=" * 70)
        print(f"TRAINING: {Name}")
        print("=" * 70)

        Model = Pipeline([
            ("Preprocessor", CreatePreprocessingPipeline()),
            ("Classifier", Classifier)
        ])

        Model, TrainingTime = TrainModel(Model, X_train, y_train)

        Metrics = EvaluateModel(Model, X_test, y_test)

        Metrics["Training_Time"] = TrainingTime

        TrainedModels[Name] = Model
        ModelResults[Name] = Metrics

    return TrainedModels, ModelResults

"""
----------------------------------------------------------------------
Function Name : CompareModels
Description   : Compares classification models using their evaluation
                metrics and training time.
Input         : Dictionary containing model evaluation results
Output        : Sorted pandas DataFrame of model performance
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def CompareModels(ModelResults):

    dfResults = pd.DataFrame.from_dict(ModelResults, orient="index")
    dfResults.index.name = "Model"

    dfResults = dfResults.sort_values(
        by="Average_Precision",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)

    print(dfResults.round(4))

    os.makedirs("results", exist_ok=True)
    dfResults.to_csv("results/Model_Comparison.csv")

    print("\nComparison saved to results/Model_Comparison.csv")

    return dfResults

"""
----------------------------------------------------------------------
Function Name : EvaluateTunedModelCV
Description   : Evaluates the tuned model using stratified
                cross-validation and Average Precision scoring.
Input         : Best tuned model, X_train, y_train
Output        : Dictionary containing cross-validation results
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def EvaluateTunedModelCV(BestModel, X_train, y_train):

    CrossValidator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    Scores = cross_validate(
        BestModel,
        X_train,
        y_train,
        cv=CrossValidator,
        scoring={
            "Average_Precision": "average_precision",
            "Precision": "precision",
            "Recall": "recall",
            "F1_Score": "f1"
        },
        n_jobs=-1
    )

    Results = {
        "Average_Precision_Mean": Scores["test_Average_Precision"].mean(),
        "Average_Precision_Std": Scores["test_Average_Precision"].std(),
        "Precision_Mean": Scores["test_Precision"].mean(),
        "Recall_Mean": Scores["test_Recall"].mean(),
        "F1_Score_Mean": Scores["test_F1_Score"].mean()
    }

    print("\n" + "=" * 70)
    print("TUNED MODEL: CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for Metric, Value in Results.items():
        print(f"{Metric}: {Value:.4f}")

    return Results

"""
----------------------------------------------------------------------
Function Name : TuneClassificationThreshold
Description   : Selects a classification threshold using precision,
                recall, and F1-score on validation predictions.
Input         : True validation labels and predicted fraud probabilities
Output        : Best threshold and threshold evaluation results
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def TuneClassificationThreshold(yValidation, yScore):


    Precision, Recall, Thresholds = precision_recall_curve(
        yValidation,
        yScore
    )

    F1Scores = (
        2 * Precision[:-1] * Recall[:-1]
        / (Precision[:-1] + Recall[:-1] + 1e-10)
    )

    BestIndex = np.argmax(F1Scores)
    BestThreshold = Thresholds[BestIndex]

    Results = pd.DataFrame({
        "Threshold": Thresholds,
        "Precision": Precision[:-1],
        "Recall": Recall[:-1],
        "F1_Score": F1Scores
    })

    print("\n" + "=" * 70)
    print("CLASSIFICATION THRESHOLD TUNING")
    print("=" * 70)

    print(f"Best Threshold: {BestThreshold:.4f}")
    print(f"Precision     : {Precision[BestIndex]:.4f}")
    print(f"Recall        : {Recall[BestIndex]:.4f}")
    print(f"F1-Score      : {F1Scores[BestIndex]:.4f}")

    os.makedirs("results", exist_ok=True)
    Results.to_csv("results/Threshold_Analysis.csv", index=False)

    print("\nThreshold analysis saved to results/Threshold_Analysis.csv")

    return BestThreshold, Results

"""
----------------------------------------------------------------------
Function Name : EvaluateFinalModel
Description   : Evaluates the tuned model on the unseen test set
                using the selected classification threshold.
Input         : Best tuned model, X_test, y_test, BestThreshold
Output        : Dictionary containing final evaluation metrics
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def EvaluateFinalModel(BestModel, X_test, y_test, BestThreshold):

    yScore = BestModel.predict_proba(X_test)[:, 1]
    yPred = (yScore >= BestThreshold).astype(int)

    Metrics = {
        "Accuracy": accuracy_score(y_test, yPred),
        "Precision": precision_score(y_test, yPred, zero_division=0),
        "Recall": recall_score(y_test, yPred, zero_division=0),
        "F1_Score": f1_score(y_test, yPred, zero_division=0),
        "Average_Precision": average_precision_score(y_test, yScore),
        "ROC_AUC": roc_auc_score(y_test, yScore)
    }

    print("\n" + "=" * 70)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 70)

    for Name, Value in Metrics.items():
        print(f"{Name}: {Value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        yPred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    os.makedirs("results", exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        yPred,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues"
    )

    plt.title("Final Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/Final_Confusion_Matrix.png", dpi=300)
    plt.show()
    plt.close()

    pd.DataFrame([Metrics]).to_csv(
        "results/Final_Model_Metrics.csv",
        index=False
    )

    print("\nConfusion matrix saved to results/Final_Confusion_Matrix.png")
    print("Metrics saved to results/Final_Model_Metrics.csv")

    return Metrics

"""
----------------------------------------------------------------------
Function Name : AnalyzePredictionErrors
Description   : Analyzes false positives and false negatives to
                understand the model's prediction errors.
Input         : Tuned model, X_test, y_test, classification threshold
Output        : DataFrame containing incorrectly classified transactions
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""

def AnalyzePredictionErrors(BestModel, X_test, y_test, BestThreshold):

    yScore = BestModel.predict_proba(X_test)[:, 1]
    yPred = (yScore >= BestThreshold).astype(int)

    dfErrors = X_test.copy()
    dfErrors["Actual_Class"] = y_test
    dfErrors["Predicted_Class"] = yPred
    dfErrors["Fraud_Probability"] = yScore

    dfErrors["Error_Type"] = np.select(
        [
            (y_test.to_numpy() == 0) & (yPred == 1),
            (y_test.to_numpy() == 1) & (yPred == 0)
        ],
        ["False Positive", "False Negative"],
        default="Correct Prediction"
    )

    dfErrors = dfErrors[dfErrors["Error_Type"] != "Correct Prediction"]

    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    print("\nError Counts:")
    print(dfErrors["Error_Type"].value_counts())
    print("\nFalse Positives:", (dfErrors["Error_Type"] == "False Positive").sum())
    print("False Negatives:", (dfErrors["Error_Type"] == "False Negative").sum())

    os.makedirs("results", exist_ok=True)
    dfErrors.to_csv("results/Prediction_Errors.csv", index=False)

    print("\nPrediction errors saved to results/Prediction_Errors.csv")

    return dfErrors

"""
----------------------------------------------------------------------
Function Name : AnalyzeAndVisualizeResults
Description   : Analyzes feature importance, compares class-imbalance
                techniques, and plots Precision-Recall curves.
Input         : Best model, X_train, y_train, X_test, y_test
Output        : DataFrames containing feature and imbalance analysis
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def AnalyzeAndVisualizeResults(BestModel, X_train, y_train, X_test, y_test):

    os.makedirs("results", exist_ok=True)

    Classifier = BestModel.named_steps["Classifier"]
    Preprocessor = BestModel.named_steps["Preprocessor"]

    Importance = (
        np.abs(Classifier.coef_[0])
        if hasattr(Classifier, "coef_")
        else Classifier.feature_importances_
        if hasattr(Classifier, "feature_importances_")
        else None
    )

    dfImportance = None

    if Importance is not None:
        dfImportance = pd.DataFrame({
            "Feature": Preprocessor.get_feature_names_out(),
            "Importance": Importance
        }).sort_values("Importance", ascending=False)

        dfImportance.to_csv("results/Feature_Importance.csv", index=False)

        dfImportance.head(15).sort_values("Importance").plot(
            x="Feature", y="Importance", kind="barh", legend=False
        )

        plt.title("Top 15 Important Features")
        plt.tight_layout()
        plt.savefig("results/Feature_Importance.png", dpi=300)
        plt.show()
        plt.close()

    Techniques = {
        "Class Weights": None,
        "SMOTE": SMOTE(random_state=42),
        "Undersampling": RandomUnderSampler(random_state=42)
    }

    ImbalanceResults = []

    for Name, Sampler in Techniques.items():
        Model = (
            clone(BestModel)
            if Sampler is None
            else ImbPipeline([
                ("Preprocessor", CreatePreprocessingPipeline()),
                ("Sampler", Sampler),
                ("Classifier", clone(Classifier))
            ])
        )

        Model.fit(X_train, y_train)

        ImbalanceResults.append({
            "Technique": Name,
            "Average_Precision": average_precision_score(
                y_test, Model.predict_proba(X_test)[:, 1]
            )
        })

    dfImbalance = pd.DataFrame(ImbalanceResults).sort_values(
        "Average_Precision", ascending=False
    )

    dfImbalance.to_csv("results/Imbalance_Comparison.csv", index=False)

    print("\nImbalance Technique Comparison:")
    print(dfImbalance.to_string(index=False))

    yScore = BestModel.predict_proba(X_test)[:, 1]
    Precision, Recall, _ = precision_recall_curve(y_test, yScore)
    AveragePrecision = average_precision_score(y_test, yScore)

    plt.figure(figsize=(8, 6))
    plt.plot(Recall, Precision, label=f"Tuned Model (AP={AveragePrecision:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/Precision_Recall_Curve.png", dpi=300)
    plt.show()
    plt.close()

    print("\nAnalysis completed. Results saved in the results folder.")

    return dfImportance, dfImbalance

"""
----------------------------------------------------------------------
Function Name : SaveModelAndResults
Description   : Saves the trained model and evaluation results,
                then reloads the model for verification.
Input         : BestModel, FinalMetrics, ModelResults, CVResults
Output        : Saved model and result files
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def SaveModelAndResults(BestModel, FinalMetrics, ModelResults, CVResults):

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    ModelPath = "models/CreditCardFraudModel.pkl"
    joblib.dump(BestModel, ModelPath)

    pd.DataFrame([FinalMetrics]).to_csv(
        "results/Final_Model_Metrics.csv", index=False
    )

    pd.DataFrame.from_dict(ModelResults, orient="index").to_csv(
        "results/Model_Comparison.csv"
    )

    pd.DataFrame([CVResults]).to_csv(
        "results/Tuned_Model_CrossValidation.csv", index=False
    )

    LoadedModel = joblib.load(ModelPath)

    print("\n" + "=" * 70)
    print("MODEL AND RESULTS SAVED SUCCESSFULLY")
    print("=" * 70)
    print(f"Model saved to: {ModelPath}")
    print("Final metrics saved to: results/Final_Model_Metrics.csv")
    print("Model comparison saved to: results/Model_Comparison.csv")
    print("Cross-validation results saved to: results/Tuned_Model_CrossValidation.csv")
    print("Saved model reloaded successfully.")

    return LoadedModel

"""
----------------------------------------------------------------------
Function Name : PredictTransaction
Description   : Predicts whether a credit card transaction is
                fraudulent using the trained classification model.
Input         : Trained model, transaction feature values,
                classification threshold
Output        : Fraud prediction and estimated fraud probability
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def PredictTransaction(Model, Transaction, Threshold=0.5):

    dfTransaction = pd.DataFrame([Transaction])

    FraudProbability = Model.predict_proba(dfTransaction)[0, 1]
    Prediction = int(FraudProbability >= Threshold)

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 70)

    print(f"Fraud Probability: {FraudProbability:.4%}")
    print(f"Threshold        : {Threshold:.4f}")
    print(
        "Prediction       :",
        "FRAUDULENT TRANSACTION" if Prediction else "LEGITIMATE TRANSACTION"
    )

    return Prediction, FraudProbability

"""
----------------------------------------------------------------------
Function Name : TuneLogisticRegression
Description   : Tunes Logistic Regression hyperparameters using
                RandomizedSearchCV and stratified cross-validation.
Input         : Preprocessing pipeline, X_train, y_train
Output        : Best trained model and hyperparameter search results
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def TuneLogisticRegression(Preprocessor, X_train, y_train):

    print("\nStarting Logistic Regression hyperparameter tuning...")

    Model = Pipeline([
        ("Preprocessor", Preprocessor),
        ("Classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    Parameters = {
        "Classifier__C": [0.01, 0.1, 1, 10, 100],
        "Classifier__solver": ["lbfgs", "liblinear"],
        "Classifier__class_weight": ["balanced"]
    }

    CrossValidator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    Search = RandomizedSearchCV(
        estimator=Model,
        param_distributions=Parameters,
        n_iter=5,
        scoring="average_precision",
        cv=CrossValidator,
        random_state=42,
        n_jobs=-1,
        refit=True
    )

    Search.fit(X_train, y_train)

    BestModel = Search.best_estimator_
    TuningResults = pd.DataFrame(Search.cv_results_)

    print("\nHyperparameter tuning completed successfully.")
    print("Best Parameters:", Search.best_params_)
    print(f"Best Average Precision: {Search.best_score_:.4f}")

    return BestModel, TuningResults

"""
----------------------------------------------------------------------
Function Name : main
Description   : Executes the complete Credit Card Fraud Detection
                project workflow.
Input         : None
Output        : Executes the project workflow and displays results.
Author        : Atharv Tushar Bhosale
Date          : 17/10/2026
----------------------------------------------------------------------
"""
def main():

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION".center(70))
    print("Machine Learning Classification Project".center(70))
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATASET")
    print("=" * 70)

    dfDataset = pd.read_csv("creditcard.csv")

    print("\n" + "=" * 70)
    print("STEP 2: DISPLAY DATASET INFORMATION")
    print("=" * 70)

    DisplayDatasetInfo(dfDataset)

    print("\n" + "=" * 70)
    print("STEP 3: VALIDATE DATASET")
    print("=" * 70)

    dfDataset = ValidateDataset(dfDataset)

    print("\n" + "=" * 70)
    print("STEP 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    PerformEDA(dfDataset)

    print("\n" + "=" * 70)
    print("STEP 5: PREPARE FEATURES AND TARGET")
    print("=" * 70)

    X, y = PrepareData(dfDataset)

    print("\n" + "=" * 70)
    print("STEP 6: SPLIT DATA")
    print("=" * 70)

    X_train, X_test, y_train, y_test = SplitData(X, y)

    print("\n" + "=" * 70)
    print("STEP 7: CREATE PREPROCESSING PIPELINE")
    print("=" * 70)

    Preprocessor = CreatePreprocessingPipeline()

    print("\n" + "=" * 70)
    print("STEP 8: TRAIN BASELINE LOGISTIC REGRESSION")
    print("=" * 70)

    BaselineModel = CreateLogisticRegressionModel(Preprocessor)

    BaselineModel, BaselineTrainingTime = TrainModel(
        BaselineModel, X_train, y_train
    )

    print("\n" + "=" * 70)
    print("STEP 9: EVALUATE BASELINE MODEL")
    print("=" * 70)

    BaselineMetrics = EvaluateModel(
        BaselineModel, X_test, y_test
    )

    BaselineMetrics["Training_Time"] = BaselineTrainingTime

    print("\n" + "=" * 70)
    print("STEP 10: CREATE ADDITIONAL MODELS")
    print("=" * 70)

    Models = CreateModels()

    print("\n" + "=" * 70)
    print("STEP 11: TRAIN AND EVALUATE ADDITIONAL MODELS")
    print("=" * 70)

    TrainedModels, ModelResults = TrainAndEvaluateModels(
        Models, X_train, y_train, X_test, y_test
    )

    ModelResults["Logistic Regression"] = BaselineMetrics

    print("\n" + "=" * 70)
    print("STEP 12: COMPARE MODELS")
    print("=" * 70)

    dfModelComparison = CompareModels(ModelResults)

    print("\n" + "=" * 70)
    print("STEP 13: HYPERPARAMETER TUNING")
    print("=" * 70)

    BestModel, TuningResults = TuneLogisticRegression(
        CreatePreprocessingPipeline(), X_train, y_train
    )

    print("\n" + "=" * 70)
    print("STEP 14: CROSS-VALIDATION OF TUNED MODEL")
    print("=" * 70)

    CVResults = EvaluateTunedModelCV(
        BestModel, X_train, y_train
    )

    print("\n" + "=" * 70)
    print("STEP 15: PREPARE VALIDATION DATA")
    print("=" * 70)

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    BestModel.fit(X_fit, y_fit)

    yValidationScore = BestModel.predict_proba(
        X_validation
    )[:, 1]

    print("\n" + "=" * 70)
    print("STEP 16: TUNE CLASSIFICATION THRESHOLD")
    print("=" * 70)

    BestThreshold, ThresholdResults = TuneClassificationThreshold(
        y_validation, yValidationScore
    )

    # STEP 17: FINAL MODEL TRAINING
    print("\n" + "=" * 70)
    print("STEP 17: RETRAIN MODEL ON FULL TRAINING DATA")
    print("=" * 70)

    BestModel.fit(X_train, y_train)

    print("\n" + "=" * 70)
    print("STEP 18: FINAL TEST EVALUATION")
    print("=" * 70)

    FinalMetrics = EvaluateFinalModel(
        BestModel, X_test, y_test, BestThreshold
    )

    print("\n" + "=" * 70)
    print("STEP 19: PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    dfErrors = AnalyzePredictionErrors(
        BestModel, X_test, y_test, BestThreshold
    )

    print("\n" + "=" * 70)
    print("STEP 20: ANALYZE AND VISUALIZE RESULTS")
    print("=" * 70)

    dfImportance, dfImbalance = AnalyzeAndVisualizeResults(
        BestModel, X_train, y_train, X_test, y_test
    )

    print("\n" + "=" * 70)
    print("STEP 21: SAVE MODEL AND RESULTS")
    print("=" * 70)

    LoadedModel = SaveModelAndResults(
        BestModel, FinalMetrics, ModelResults, CVResults
    )

    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
