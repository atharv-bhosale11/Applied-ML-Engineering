"""
----------------------------------------------------------------------
Project Name        : Breast Cancer Classification Using Ensemble Learning
Dataset File        : breast_cancer.csv
Target Variable     : target or diagnosis
Algorithms Used     : Random Forest, AdaBoost, Gradient Boosting,
                      Hard Voting Classifier, Soft Voting Classifier
Techniques Used     : Data Preparation, Missing-Value Handling,
                      Train-Test Splitting, Ensemble Learning,
                      Cross-Validation, Model Selection, Evaluation,
                      Confusion Matrix, Feature Importance,
                      Individual Predictions, Model Saving
Library             : Pandas, NumPy, Matplotlib, Scikit-Learn
Problem Type        : Binary Classification
Author              : Atharv Tushar Bhosale
Date                : 14/10/2026
----------------------------------------------------------------------
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    make_scorer
)

DATASET_PATH = "breast_cancer.csv"
MODEL_PATH = "BestBreastCancerModel.pkl"
COMPARISON_PATH = "Model_Comparison.csv"
ALGORITHM_GRAPH_PATH = "Algorithm_Comparison.png"
CONFUSION_MATRIX_PATH = "Confusion_Matrix.png"
FEATURE_IMPORTANCE_PATH = "Feature_Importance.png"

# Set this to the target value representing malignant if labels are numeric.
# Example: MALIGNANT_LABEL = 0 (only if your CSV confirms that 0 is malignant).
MALIGNANT_LABEL = 0


"""
----------------------------------------------------------------------
Function Name : LoadDataset
Description   : Loads and analyzes the dataset without displaying dtypes.
Input         : Dataset file path
Output        : Pandas DataFrame
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def LoadDataset(sFilePath):
    if not os.path.exists(sFilePath):
        raise FileNotFoundError(f"Dataset not found: {sFilePath}")

    df = pd.read_csv(sFilePath)

    print("\n" + "=" * 70)
    print("STEP 1: DATASET ANALYSIS")
    print("=" * 70)
    print("\nDataset Shape:\n", df.shape)
    print("\nFirst Five Records:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Records:", df.duplicated().sum())
    print("\nStatistical Summary:\n", df.describe(include="all"))

    return df


"""
----------------------------------------------------------------------
Function Name : PrepareDataset
Description   : Identifies the target and prepares feature columns.
                Missing feature values are handled during preprocessing.
Input         : Pandas DataFrame
Output        : Features, target, target column and malignant label
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def PrepareDataset(df):
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPARATION")
    print("=" * 70)

    df = df.drop_duplicates().copy()
    target_candidates = ["target", "diagnosis"]

    target_column = next(
        (column for column in df.columns
         if column.strip().lower() in target_candidates),
        None
    )

    if target_column is None:
        raise ValueError(
            "Target column not found. Expected 'target' or 'diagnosis'."
        )

    df = df.dropna(subset=[target_column])
    y = df[target_column]

    identifier_columns = [
        column for column in df.columns
        if column.strip().lower() in ["id", "id_number", "unnamed: 0"]
    ]

    X = df.drop(columns=[target_column] + identifier_columns)
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable feature columns were found.")

    if y.nunique() != 2:
        raise ValueError("The dataset must contain exactly two target classes.")

    if y.value_counts().min() < 5:
        raise ValueError(
            "Each target class needs at least five records for 5-fold "
            "stratified cross-validation."
        )

    if MALIGNANT_LABEL is not None:
        malignant_label = MALIGNANT_LABEL
        if malignant_label not in set(y.unique()):
            raise ValueError(
                f"MALIGNANT_LABEL={malignant_label!r} is not present in "
                f"the target values: {list(y.unique())}"
            )
    else:
        label_lookup = {str(value).strip().lower(): value for value in y.unique()}
        malignant_label = next(
            (
                label_lookup[key]
                for key in ["malignant", "m", "malignant tumor", "cancer"]
                if key in label_lookup
            ),
            None
        )

        if malignant_label is None:
            raise ValueError(
                f"Cannot safely identify the malignant label. Target values: "
                f"{list(y.unique())}. Verify the CSV and set MALIGNANT_LABEL "
                "at the top of this file to the value representing malignant."
            )

    print("\nTarget Column:", target_column)
    print("Feature Shape:", X.shape)
    print("\nTarget Distribution:\n", y.value_counts())
    print("\nTarget Labels:", list(y.unique()))
    print("Malignant Label:", malignant_label)

    return X, y, target_column, malignant_label


"""
----------------------------------------------------------------------
Function Name : SplitDataset
Description   : Splits data into training and testing sets.
Input         : Features and target
Output        : X_train, X_test, y_train and y_test
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def SplitDataset(X, y):
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN-TEST SPLITTING")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training Records:", len(X_train))
    print("Testing Records :", len(X_test))

    return X_train, X_test, y_train, y_test


"""
----------------------------------------------------------------------
Function Name : CreatePreprocessor
Description   : Imputes missing values and encodes categorical features.
Input         : Feature DataFrame
Output        : Scikit-Learn ColumnTransformer
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def CreatePreprocessor(X):
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers)


"""
----------------------------------------------------------------------
Function Name : CreateModels
Description   : Creates three individual ensemble classifiers.
Input         : None
Output        : Dictionary of classifier objects
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def CreateModels():
    print("\n" + "=" * 70)
    print("STEP 4: MODEL CREATION")
    print("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    for name in models:
        print(name, "created successfully.")

    return models


"""
----------------------------------------------------------------------
Function Name : CreateVotingModels
Description   : Creates hard and soft voting classifiers.
Input         : Dictionary of individual classifiers
Output        : Dictionary of voting classifiers
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def CreateVotingModels(models):
    print("\n" + "=" * 70)
    print("STEP 5: VOTING CLASSIFIER CREATION")
    print("=" * 70)

    estimators = [
        ("rf", clone(models["Random Forest"])),
        ("ada", clone(models["AdaBoost"])),
        ("gb", clone(models["Gradient Boosting"]))
    ]

    voting_models = {
        "Hard Voting": VotingClassifier(estimators=estimators, voting="hard"),
        "Soft Voting": VotingClassifier(estimators=estimators, voting="soft")
    }

    for name in voting_models:
        print(name, "created successfully.")

    return voting_models


"""
----------------------------------------------------------------------
Function Name : BuildPipelines
Description   : Combines preprocessing and classifiers into pipelines.
Input         : Models and training features
Output        : Dictionary of complete model pipelines
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def BuildPipelines(models, X_train):
    preprocessor = CreatePreprocessor(X_train)

    return {
        name: Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(model))
        ])
        for name, model in models.items()
    }


"""
----------------------------------------------------------------------
Function Name : EvaluateCrossValidation
Description   : Compares all five models using 5-fold cross-validation
                for accuracy, malignant recall and malignant F1-score.
Input         : Model pipelines, training data and malignant label
Output        : Cross-validation results DataFrame
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def EvaluateCrossValidation(models, X_train, y_train, malignant_label):
    print("\n" + "=" * 70)
    print("STEP 6: CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "malignant_recall": make_scorer(
            recall_score, pos_label=malignant_label, zero_division=0
        ),
        "malignant_f1": make_scorer(
            f1_score, pos_label=malignant_label, zero_division=0
        )
    }

    cv_results = {}

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scorers, n_jobs=-1
        )

        cv_results[name] = {
            "CV Accuracy (%)": scores["test_accuracy"].mean() * 100,
            "Malignant Recall (%)":
                scores["test_malignant_recall"].mean() * 100,
            "Malignant F1-Score (%)":
                scores["test_malignant_f1"].mean() * 100
        }

        print(f"\nModel: {name}")
        print(f"CV Accuracy: {cv_results[name]['CV Accuracy (%)']:.2f}%")
        print(
            "Malignant Recall: "
            f"{cv_results[name]['Malignant Recall (%)']:.2f}%"
        )
        print(
            "Malignant F1-Score: "
            f"{cv_results[name]['Malignant F1-Score (%)']:.2f}%"
        )

    results_df = pd.DataFrame.from_dict(cv_results, orient="index")
    results_df.index.name = "Algorithm"

    return results_df.reset_index()


"""
----------------------------------------------------------------------
Function Name : SelectBestModel
Description   : Selects a model by CV malignant recall, then malignant
                F1-score, then accuracy, and saves the fitted model.
Input         : CV results, model pipelines and training data
Output        : Best model name and fitted model
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def SelectBestModel(cv_results, models, X_train, y_train):
    print("\n" + "=" * 70)
    print("STEP 7: MODEL SELECTION")
    print("=" * 70)

    ranked = cv_results.sort_values(
        by=[
            "Malignant Recall (%)",
            "Malignant F1-Score (%)",
            "CV Accuracy (%)"
        ],
        ascending=False
    )

    best_name = ranked.iloc[0]["Algorithm"]
    best_model = clone(models[best_name])
    best_model.fit(X_train, y_train)

    print("Selection priority: malignant recall, malignant F1-score, accuracy.")
    print("Selected Model:", best_name)
    print("Mean CV Accuracy:", f"{ranked.iloc[0]['CV Accuracy (%)']:.2f}%")
    print(
        "Mean CV Malignant Recall:",
        f"{ranked.iloc[0]['Malignant Recall (%)']:.2f}%"
    )

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    print("Saved model:", MODEL_PATH)

    return best_name, best_model


"""
----------------------------------------------------------------------
Function Name : EvaluateTestModels
Description   : Evaluates all five models on held-out test data for
                reporting only; test scores are not used for selection.
Input         : Model pipelines, train/test data and malignant label
Output        : Test-results DataFrame and fitted models
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def EvaluateTestModels(
    models, X_train, y_train, X_test, y_test, malignant_label
):
    print("\n" + "=" * 70)
    print("STEP 8: FINAL TEST-SET EVALUATION")
    print("=" * 70)

    results = []
    fitted_models = {}

    for name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        predictions = fitted_model.predict(X_test)

        results.append({
            "Algorithm": name,
            "Accuracy (%)": accuracy_score(y_test, predictions) * 100,
            "Precision (Weighted %)": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ) * 100,
            "Recall (Weighted %)": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ) * 100,
            "F1-Score (Weighted %)": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ) * 100,
            "Malignant Recall (%)": recall_score(
                y_test, predictions, pos_label=malignant_label, zero_division=0
            ) * 100,
            "Malignant F1-Score (%)": f1_score(
                y_test, predictions, pos_label=malignant_label, zero_division=0
            ) * 100
        })

        fitted_models[name] = fitted_model

        print(f"\n{'-' * 70}\nAlgorithm: {name}\n{'-' * 70}")
        print(classification_report(y_test, predictions, zero_division=0))
        print(
            f"Malignant Recall: "
            f"{results[-1]['Malignant Recall (%)']:.2f}%"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(COMPARISON_PATH, index=False)
    print("\nTest-set results saved to:", COMPARISON_PATH)

    return results_df, fitted_models


"""
----------------------------------------------------------------------
Function Name : DisplayModelComparison
Description   : Plots held-out test metrics for all five models.
Input         : Test-results DataFrame
Output        : Saves model-comparison bar graph
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def DisplayModelComparison(results_df):
    print("\n" + "=" * 70)
    print("STEP 9: MODEL COMPARISON GRAPH")
    print("=" * 70)

    metrics = [
        "Accuracy (%)",
        "Precision (Weighted %)",
        "Recall (Weighted %)",
        "F1-Score (Weighted %)",
        "Malignant Recall (%)",
        "Malignant F1-Score (%)"
    ]

    ax = results_df.set_index("Algorithm")[metrics].plot(
        kind="bar", figsize=(14, 7)
    )
    ax.set_title("Test-Set Model Comparison")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(ALGORITHM_GRAPH_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("Comparison graph saved to:", ALGORITHM_GRAPH_PATH)


"""
----------------------------------------------------------------------
Function Name : DisplayConfusionMatrix
Description   : Displays and saves the selected model's confusion matrix.
Input         : Trained model, X_test and y_test
Output        : Saves confusion-matrix graph
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def DisplayConfusionMatrix(model, X_test, y_test):
    print("\n" + "=" * 70)
    print("STEP 10: CONFUSION MATRIX")
    print("=" * 70)

    predictions = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=model.classes_, cmap="Blues"
    )
    plt.title("Selected Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("Confusion matrix saved to:", CONFUSION_MATRIX_PATH)


"""
----------------------------------------------------------------------
Function Name : DisplayFeatureImportance
Description   : Displays feature importance for the selected tree-based
                model or averages importance across voting estimators.
Input         : Trained pipeline
Output        : Saves feature-importance graph when supported
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def DisplayFeatureImportance(model):
    print("\n" + "=" * 70)
    print("STEP 11: FEATURE IMPORTANCE")
    print("=" * 70)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if isinstance(classifier, VotingClassifier):
        estimators = [
            estimator for estimator in classifier.named_estimators_.values()
            if hasattr(estimator, "feature_importances_")
        ]
        if not estimators:
            print("Feature importance is unavailable for this model.")
            return
        importances = np.mean(
            [estimator.feature_importances_ for estimator in estimators],
            axis=0
        )
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        print("Feature importance is unavailable for this model.")
        return

    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        feature_names = np.array(
            [f"Feature {i + 1}" for i in range(len(importances))]
        )

    if len(feature_names) != len(importances):
        feature_names = np.array(
            [f"Feature {i + 1}" for i in range(len(importances))]
        )

    importance = pd.Series(
        importances, index=feature_names
    ).sort_values(ascending=False).head(10)

    print("\nTop 10 Features:\n", importance.to_string())
    importance.sort_values().plot(kind="barh", figsize=(10, 6))
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("Feature-importance graph saved to:", FEATURE_IMPORTANCE_PATH)


"""
----------------------------------------------------------------------
Function Name : DetectCancer
Description   : Compares predictions with actual labels for test records.
Input         : Trained model, X_test, y_test and prediction count
Output        : Displays predicted and actual labels
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def DetectCancer(model, X_test, y_test, count):
    print("\n" + "=" * 70)
    print("STEP 12: INDIVIDUAL PATIENT PREDICTIONS")
    print("=" * 70)

    samples = X_test.iloc[:count]
    predictions = model.predict(samples)

    for i, (prediction, actual) in enumerate(
        zip(predictions, y_test.loc[samples.index]), start=1
    ):
        result = "Correct" if prediction == actual else "Wrong"
        print(
            f"Patient {i}: Predicted = {prediction}, "
            f"Actual = {actual} -> {result}"
        )


"""
----------------------------------------------------------------------
Function Name : main
Description   : Runs the complete classification workflow.
Input         : None
Output        : Reports, three diagrams, comparison CSV and saved model
Author        : Atharv Tushar Bhosale
Date          : 14/10/2026
----------------------------------------------------------------------
"""


def main():
    df = LoadDataset(DATASET_PATH)
    X, y, target_column, malignant_label = PrepareDataset(df)
    X_train, X_test, y_train, y_test = SplitDataset(X, y)

    individual_models = CreateModels()
    voting_models = CreateVotingModels(individual_models)
    all_models = {**individual_models, **voting_models}
    model_pipelines = BuildPipelines(all_models, X_train)

    cv_results = EvaluateCrossValidation(
        model_pipelines, X_train, y_train, malignant_label
    )

    best_name, best_model = SelectBestModel(
        cv_results, model_pipelines, X_train, y_train
    )

    test_results, fitted_models = EvaluateTestModels(
        model_pipelines, X_train, y_train,
        X_test, y_test, malignant_label
    )

    DisplayModelComparison(test_results)
    DisplayConfusionMatrix(best_model, X_test, y_test)
    DisplayFeatureImportance(best_model)
    DetectCancer(best_model, X_test, y_test, count=10)

    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Selected Model:", best_name)


if __name__ == "__main__":
    main()
