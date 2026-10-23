"""
Project Name: IMDB Movie Review Sentiment Analysis Using NLP
Project Type: Natural Language Processing and Machine Learning

Description:
This project analyzes the IMDB Dataset of 50K Movie Reviews to classify
movie reviews into positive and negative sentiment categories using
Natural Language Processing (NLP) and Machine Learning techniques.

Objectives:
1. Understand and inspect the movie review dataset.
2. Perform exploratory data analysis (EDA).
3. Analyze positive and negative sentiment distributions.
4. Clean, preprocess, and tokenize text data.
5. Convert text into numerical features using Bag of Words and TF-IDF.
6. Train multiple machine learning classification models.
7. Evaluate and compare model performance using cross-validation.
8. Perform error analysis and improve model performance.
9. Select and save the best-performing model and text vectorizer.
10. Generate visualizations to present analytical findings.
11. Test sentiment predictions on new movie reviews.
12. Organize project outputs and files for GitHub.

Technologies Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Joblib

Expected Outcome:
An end-to-end NLP and Machine Learning project that classifies IMDB
movie reviews as positive or negative, compares multiple classification
models, evaluates prediction performance, and provides a reusable
sentiment prediction pipeline.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib

"""
----------------------------------------------------------------------
Function Name : setup_project
Description   : Defines project paths, directory locations, and
                configuration constants required for the IMDB
                sentiment analysis project.
Input         : None
Output        : Creates required project directories and defines
                paths and constants for dataset loading, model
                saving, and result generation.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

import os

############################################################
# Project Paths and Constants
############################################################

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAW_DIR = BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "IMDB_Dataset.csv")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

REQUIRED_COLUMNS = ["review", "sentiment"]

BORDER = "=" * 60


def setup_project():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(BORDER)
    print("IMDB SENTIMENT ANALYSIS PROJECT SETUP")
    print(BORDER)

    print("Base Directory :", BASE_DIR)
    print("Dataset Path   :", DATA_PATH)
    print("Images Folder  :", IMAGES_DIR)
    print("Models Folder  :", MODELS_DIR)
    print("Results Folder :", RESULTS_DIR)

    print(BORDER)
    print("Project setup completed successfully.")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : display_header
Description   : Displays the project title, author information, and
                a brief introduction to the IMDB sentiment analysis
                project.
Input         : None
Output        : Prints a formatted project header in the console.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def display_header():
    print("\n" + BORDER)
    print("       IMDB MOVIE REVIEW SENTIMENT ANALYSIS")
    print("          USING NLP AND MACHINE LEARNING")
    print(BORDER)

    print("Project  : Sentiment Classification")
    print("Dataset  : IMDB Dataset of 50K Movie Reviews")
    print("Author   : Atharv Tushar Bhosale")
    print("Language : Python")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : load_dataset
Description   : Loads the IMDB movie review dataset from the specified
                CSV file into a Pandas DataFrame.
Input         : None
Output        : Returns a Pandas DataFrame containing movie reviews
                and their corresponding sentiment labels.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def load_dataset():
    try:
        df = pd.read_csv(DATA_PATH)

        print(BORDER)
        print("STEP 1 : IMDB DATASET LOADED SUCCESSFULLY")
        print(BORDER)
        print("Dataset Path :", DATA_PATH)
        print("Total Records:", len(df))

        return df

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        print("Expected Location:", DATA_PATH)
        return None

    except Exception as e:
        print("Error while loading dataset:", e)
        return None
    
    """
----------------------------------------------------------------------
Function Name : understand_dataset
Description   : Provides an initial overview of the IMDB movie review
                dataset, including its size, columns, data types,
                missing values, and sample records.
Input         : df - Pandas DataFrame returned by load_dataset().
Output        : Displays an overview of the dataset to help understand
                its contents and structure.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def understand_dataset(df):
    print(BORDER)
    print("STEP 2 : UNDERSTANDING THE IMDB DATASET")
    print(BORDER)

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst Five Records:")
    print(df.head())

    print("\nRandom Five Records:")
    print(df.sample(5, random_state=RANDOM_STATE))


"""
----------------------------------------------------------------------
Function Name : check_structure
Description   : Validates the dataset structure by checking required
                columns, data types, missing values, and duplicate
                records.
Input         : df - Pandas DataFrame returned by load_dataset().
Output        : Displays structural validation results and identifies
                potential data quality issues.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def check_structure(df):
    print(BORDER)
    print("STEP 3 : CHECKING DATASET STRUCTURE")
    print(BORDER)

    print("\nRequired Columns:")
    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            print(f"{column}: Available")
        else:
            print(f"{column}: Missing")

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nUnique Sentiment Labels:")
    if "sentiment" in df.columns:
        print(df["sentiment"].value_counts(dropna=False))

"""
----------------------------------------------------------------------
Function Name : clean_dataset
Description   : Cleans the IMDB dataset by removing missing values,
                duplicate records, empty reviews, and invalid
                sentiment labels.
Input         : df - Pandas DataFrame returned by load_dataset().
Output        : Returns a cleaned Pandas DataFrame containing valid
                movie reviews and sentiment labels.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def clean_dataset(df):
    print(BORDER)
    print("STEP 4 : CLEANING THE IMDB DATASET")
    print(BORDER)

    initial_rows = len(df)

    # Remove rows with missing values
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove empty or whitespace-only reviews
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""]

    # Normalize sentiment labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Keep only valid sentiment labels
    df = df[df["sentiment"].isin(["positive", "negative"])].copy()

    removed_rows = initial_rows - len(df)

    print("Initial Records :", initial_rows)
    print("Final Records   :", len(df))
    print("Removed Records :", removed_rows)

    print("\nMissing Values After Cleaning:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    print("\nSentiment Labels After Cleaning:")
    print(df["sentiment"].value_counts())

    print(BORDER)
    print("DATASET CLEANING COMPLETED")
    print(BORDER)

    return df

"""
----------------------------------------------------------------------
Function Name : analyse_sentiment_distribution
Description   : Analyzes the distribution of positive and negative
                sentiments in the cleaned IMDB movie review dataset.
Input         : df - Cleaned Pandas DataFrame containing review and
                sentiment columns.
Output        : Displays sentiment counts and percentages and saves
                a bar chart showing the sentiment distribution.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def analyse_sentiment_distribution(df):
    print(BORDER)
    print("STEP 5 : SENTIMENT DISTRIBUTION ANALYSIS")
    print(BORDER)

    sentiment_counts = df["sentiment"].value_counts()
    sentiment_percentages = df["sentiment"].value_counts(normalize=True) * 100

    print("\nSentiment Counts:")
    print(sentiment_counts)

    print("\nSentiment Percentages:")
    print(sentiment_percentages.round(2))

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment", hue="sentiment", legend=False)

    plt.title("Distribution of IMDB Review Sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    output_path = os.path.join(
        IMAGES_DIR,
        "sentiment_distribution.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print("\nChart saved to:", output_path)
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : analyse_review_lengths
Description   : Analyzes the length of IMDB movie reviews by
                calculating word counts and generating visualizations
                to understand review length patterns.
Input         : df - Cleaned Pandas DataFrame containing the review
                and sentiment columns.
Output        : Displays review length statistics and saves
                visualizations showing review length distributions.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def analyse_review_lengths(df):
    print(BORDER)
    print("STEP 6 : REVIEW LENGTH ANALYSIS")
    print(BORDER)

    # Calculate the number of words in each review
    df = df.copy()
    df["review_length"] = df["review"].str.split().str.len()

    print("\nReview Length Statistics:")
    print(df["review_length"].describe())

    print("\nAverage Review Length by Sentiment:")
    print(df.groupby("sentiment")["review_length"].mean().round(2))

    # Plot review length distribution
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="review_length",
        hue="sentiment",
        bins=50,
        kde=True
    )

    plt.title("Distribution of IMDB Review Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    output_path = os.path.join(
        IMAGES_DIR,
        "review_length_distribution.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print("\nChart saved to:", output_path)
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : explore_common_words
Description   : Identifies the most frequently occurring words in
                positive and negative IMDB movie reviews to understand
                common language patterns in each sentiment category.
Input         : df - Cleaned Pandas DataFrame containing review and
                sentiment columns.
Output        : Displays the most common words for each sentiment
                and saves a visualization of their frequencies.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def explore_common_words(df):
    print(BORDER)
    print("STEP 7 : EXPLORING COMMON WORDS")
    print(BORDER)

    from collections import Counter

    for sentiment in ["positive", "negative"]:
        reviews = df.loc[df["sentiment"] == sentiment, "review"]

        words = " ".join(reviews).lower().split()
        common_words = Counter(words).most_common(20)

        word_df = pd.DataFrame(
            common_words,
            columns=["word", "frequency"]
        )

        print(f"\nTop 20 Words in {sentiment.title()} Reviews:")
        print(word_df.to_string(index=False))

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=word_df,
            x="frequency",
            y="word",
            hue="word",
            legend=False
        )

        plt.title(f"Top 20 Common Words in {sentiment.title()} Reviews")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.tight_layout()

        output_path = os.path.join(
            IMAGES_DIR,
            f"common_words_{sentiment}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.show()
        plt.close()

        print("Chart saved to:", output_path)

    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : clean_review_text
Description   : Cleans individual movie reviews by removing HTML tags,
                URLs, punctuation, and extra whitespace, and by
                converting text to lowercase.
Input         : text - A string containing a raw movie review.
Output        : Returns a cleaned string suitable for further
                natural language processing.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def clean_review_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

"""
----------------------------------------------------------------------
Function Name : preprocess_reviews
Description   : Applies the clean_review_text() function to every
                movie review to prepare the text for feature
                extraction and machine learning.
Input         : df - Cleaned Pandas DataFrame containing the review
                and sentiment columns.
Output        : Returns a DataFrame containing the original reviews,
                cleaned reviews, and sentiment labels.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def preprocess_reviews(df):
    print(BORDER)
    print("STEP 8 : PREPROCESSING MOVIE REVIEWS")
    print(BORDER)

    df = df.copy()

    # Apply text cleaning to every review
    df["cleaned_review"] = df["review"].apply(clean_review_text)

    print("\nOriginal Review:")
    print(df["review"].iloc[0])

    print("\nCleaned Review:")
    print(df["cleaned_review"].iloc[0])

    print("\nTotal Reviews Preprocessed:", len(df))

    print(BORDER)
    print("REVIEW PREPROCESSING COMPLETED")
    print(BORDER)

    return df

"""
----------------------------------------------------------------------
Function Name : encode_labels
Description   : Converts positive and negative sentiment labels into
                numerical values for machine learning algorithms.
Input         : df - Pandas DataFrame containing the cleaned_review
                and sentiment columns.
Output        : Returns the DataFrame with an additional sentiment_encoded
                column, where negative is 0 and positive is 1.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def encode_labels(df):
    print(BORDER)
    print("STEP 9 : ENCODING SENTIMENT LABELS")
    print(BORDER)

    df = df.copy()

    # Convert sentiment labels into numerical values
    sentiment_mapping = {
        "negative": 0,
        "positive": 1
    }

    df["sentiment_encoded"] = df["sentiment"].map(sentiment_mapping)

    print("\nSentiment Mapping:")
    print(sentiment_mapping)

    print("\nEncoded Sentiment Counts:")
    print(df["sentiment_encoded"].value_counts().sort_index())

    print("\nSample Encoded Labels:")
    print(df[["sentiment", "sentiment_encoded"]].head())

    print(BORDER)
    print("SENTIMENT LABEL ENCODING COMPLETED")
    print(BORDER)

    return df

"""
----------------------------------------------------------------------
Function Name : create_features
Description   : Converts preprocessed movie reviews into numerical
                feature representations using Bag of Words and TF-IDF.
Input         : df - Pandas DataFrame containing cleaned reviews and
                encoded sentiment labels.
Output        : Returns feature matrices and target labels for both
                Bag of Words and TF-IDF representations.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def create_features(df):
    print(BORDER)
    print("STEP 10 : CREATING TEXT FEATURES")
    print(BORDER)

    # Extract cleaned review text and encoded sentiment labels
    X_text = df["cleaned_review"]
    y = df["sentiment_encoded"]

    # Create Bag of Words features
    bow_vectorizer = CountVectorizer(max_features=10000)

    X_bow = bow_vectorizer.fit_transform(X_text)

    # Create TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)

    X_tfidf = tfidf_vectorizer.fit_transform(X_text)

    print("\nBag of Words Feature Shape:", X_bow.shape)
    print("TF-IDF Feature Shape      :", X_tfidf.shape)
    print("Target Labels Shape      :", y.shape)

    print("\nNumber of Bag of Words Features:", len(bow_vectorizer.get_feature_names_out()))
    print("Number of TF-IDF Features      :", len(tfidf_vectorizer.get_feature_names_out()))

    print(BORDER)
    print("TEXT FEATURE CREATION COMPLETED")
    print(BORDER)

    return X_bow, X_tfidf, y, bow_vectorizer, tfidf_vectorizer

"""
----------------------------------------------------------------------
Function Name : split_data
Description   : Splits the preprocessed movie reviews and sentiment
                labels into training and testing datasets.
Input         : df - Pandas DataFrame containing cleaned reviews and
                encoded sentiment labels.
Output        : Returns training and testing text data and target
                labels for model training and evaluation.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def split_data(df):
    print(BORDER)
    print("STEP 11 : SPLITTING DATA INTO TRAINING AND TESTING SETS")
    print(BORDER)

    X = df["cleaned_review"]
    y = df["sentiment_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTotal Records :", len(df))
    print("Training Records:", len(X_train))
    print("Testing Records :", len(X_test))

    print("\nTraining Sentiment Distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting Sentiment Distribution:")
    print(y_test.value_counts().sort_index())

    print(BORDER)
    print("DATA SPLITTING COMPLETED")
    print(BORDER)

    return X_train, X_test, y_train, y_test

"""
----------------------------------------------------------------------
Function Name : build_baseline_model
Description   : Builds and evaluates a baseline sentiment classifier
                that always predicts the most frequent sentiment
                class in the training dataset.
Input         : X_train - Training movie review text.
                X_test  - Testing movie review text.
                y_train - Training sentiment labels.
                y_test  - Testing sentiment labels.
Output        : Returns the trained baseline model and its accuracy
                score on the testing dataset.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def build_baseline_model(X_train, X_test, y_train, y_test):
    print(BORDER)
    print("STEP 12 : BUILDING BASELINE MODEL")
    print(BORDER)

    # Create a simple baseline classifier
    baseline_model = DummyClassifier(strategy="most_frequent")

    # Train the model using training labels
    baseline_model.fit(X_train, y_train)

    # Predict sentiments for the test dataset
    y_pred = baseline_model.predict(X_test)

    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test, y_pred)

    print("\nBaseline Model:", "DummyClassifier")
    print("Strategy      :", "most_frequent")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )

    print(BORDER)
    print("BASELINE MODEL EVALUATION COMPLETED")
    print(BORDER)

    return baseline_model, baseline_accuracy

"""
----------------------------------------------------------------------
Function Name : create_model_pipelines
Description   : Creates machine learning pipelines combining TF-IDF
                text vectorization with multiple sentiment
                classification algorithms.
Input         : None
Output        : Returns a dictionary containing three model pipelines
                for Multinomial Naive Bayes, Logistic Regression,
                and Linear Support Vector Machine.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def create_model_pipelines():
    print(BORDER)
    print("STEP 13 : CREATING MODEL PIPELINES")
    print(BORDER)

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("classifier", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("classifier", LinearSVC(
                random_state=RANDOM_STATE
            ))
        ])
    }

    print("\nAvailable Model Pipelines:")

    for model_name in models:
        print("-", model_name)

    print("\nEach pipeline includes:")
    print("1. TF-IDF text vectorization")
    print("2. A sentiment classification algorithm")

    print(BORDER)
    print("MODEL PIPELINES CREATED SUCCESSFULLY")
    print(BORDER)

    return models

"""
----------------------------------------------------------------------
Function Name : compare_models
Description   : Trains and compares multiple sentiment classification
                pipelines using cross-validation and test-set metrics.
Input         : models  - Dictionary of machine learning pipelines.
                X_train - Training movie review text.
                X_test  - Testing movie review text.
                y_train - Training sentiment labels.
                y_test  - Testing sentiment labels.
Output        : Returns a DataFrame containing model performance
                metrics and a dictionary of trained models.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def compare_models(models, X_train, X_test, y_train, y_test):
    print(BORDER)
    print("STEP 14 : COMPARING MACHINE LEARNING MODELS")
    print(BORDER)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining Model: {model_name}")

        # Evaluate using cross-validation on training data only
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1
        )

        # Train the model on the complete training dataset
        model.fit(X_train, y_train)

        # Predict sentiment on the test dataset
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        results.append({
            "Model": model_name,
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[model_name] = model

        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        print(f"Test Accuracy   : {accuracy:.4f}")
        print(f"Precision       : {precision:.4f}")
        print(f"Recall          : {recall:.4f}")
        print(f"F1 Score        : {f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    print("\nMODEL COMPARISON RESULTS")
    print(results_df.to_string(index=False))

    print(BORDER)
    print("MODEL COMPARISON COMPLETED")
    print(BORDER)

    return results_df, trained_models

"""
----------------------------------------------------------------------
Function Name : select_best_model
Description   : Selects the best-performing sentiment classification
                model based on its F1-score from the comparison results.
Input         : results_df    - DataFrame containing model evaluation
                                metrics.
                trained_models - Dictionary containing trained models.
Output        : Returns the name and trained pipeline of the
                best-performing model.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def select_best_model(results_df, trained_models):
    print(BORDER)
    print("STEP 15 : SELECTING THE BEST MODEL")
    print(BORDER)

    # Find the model with the highest F1-score
    best_model_name = results_df.loc[
        results_df["F1 Score"].idxmax(),
        "Model"
    ]

    best_model = trained_models[best_model_name]

    print("\nBest Model Name:", best_model_name)
    print(
        "Best F1-Score  :",
        f"{results_df['F1 Score'].max():.4f}"
    )

    print("\nBest Model Performance:")
    print(
        results_df.loc[
            results_df["Model"] == best_model_name
        ].to_string(index=False)
    )

    print(BORDER)
    print("BEST MODEL SELECTED SUCCESSFULLY")
    print(BORDER)

    return best_model_name, best_model

"""
----------------------------------------------------------------------
Function Name : tune_best_model
Description   : Tunes the hyperparameters of the selected sentiment
                classification pipeline using GridSearchCV.
Input         : best_model_name - Name of the selected model.
                best_model      - Trained pipeline to be tuned.
                X_train         - Training movie review text.
                y_train         - Training sentiment labels.
Output        : Returns the best estimator and its optimal
                hyperparameters identified through cross-validation.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def tune_best_model(best_model_name, best_model, X_train, y_train):
    print(BORDER)
    print("STEP 16 : TUNING THE BEST MODEL")
    print(BORDER)

    # Define hyperparameter grids for each classifier
    param_grids = {
        "Naive Bayes": {
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__alpha": [0.1, 0.5, 1.0]
        },

        "Logistic Regression": {
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1.0, 10.0]
        },

        "Linear SVM": {
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1.0, 10.0]
        }
    }

    if best_model_name not in param_grids:
        raise ValueError(
            f"No parameter grid defined for: {best_model_name}"
        )

    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grids[best_model_name],
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # Tune using training data only
    grid_search.fit(X_train, y_train)

    print("\nBest Model:", best_model_name)
    print("Best Cross-Validation F1-Score:", round(grid_search.best_score_, 4))

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    print(BORDER)
    print("HYPERPARAMETER TUNING COMPLETED")
    print(BORDER)

    return grid_search.best_estimator_, grid_search.best_params_

"""
----------------------------------------------------------------------
Function Name : train_final_model
Description   : Trains the selected and tuned sentiment classification
                pipeline using the complete training dataset.
Input         : best_model - Best pipeline returned by tune_best_model().
                X_train    - Training movie review text.
                y_train    - Training sentiment labels.
Output        : Returns the final trained sentiment classification
                pipeline.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def train_final_model(best_model, X_train, y_train):
    print(BORDER)
    print("STEP 17 : TRAINING FINAL SENTIMENT MODEL")
    print(BORDER)

    # Train the tuned pipeline on all training data
    best_model.fit(X_train, y_train)

    print("\nFinal Model Training Completed Successfully.")
    print("Training Records:", len(X_train))

    print("\nPipeline Steps:")
    for step_name, step in best_model.steps:
        print(f"- {step_name}: {step.__class__.__name__}")

    print(BORDER)

    return best_model

"""
----------------------------------------------------------------------
Function Name : evaluate_model
Description   : Evaluates the final trained sentiment classification
                pipeline using multiple classification metrics.
Input         : model  - Final trained sentiment classification pipeline.
                X_test - Testing movie review text.
                y_test - Actual testing sentiment labels.
Output        : Returns a dictionary containing evaluation metrics
                and the predicted sentiment labels.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def evaluate_model(model, X_test, y_test):
    print(BORDER)
    print("STEP 18 : FINAL MODEL EVALUATION")
    print(BORDER)

    # Predict sentiments for the test reviews
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Display evaluation results
    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )

    # Store evaluation metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(BORDER)
    print("MODEL EVALUATION COMPLETED")
    print(BORDER)

    return metrics, y_pred

"""
----------------------------------------------------------------------
Function Name : analyse_errors
Description   : Identifies and analyzes movie reviews that the final
                sentiment model incorrectly classifies.
Input         : df     - Preprocessed DataFrame containing original
                         reviews and sentiment labels.
                X_test - Testing movie review text.
                y_test - Actual testing sentiment labels.
                y_pred - Predicted sentiment labels from evaluate_model().
Output        : Displays misclassified reviews and returns a DataFrame
                containing actual and predicted sentiment information.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def analyse_errors(df, X_test, y_test, y_pred):
    print(BORDER)
    print("STEP 19 : ANALYZING MISCLASSIFIED REVIEWS")
    print(BORDER)

    # Create a DataFrame containing test results
    error_df = pd.DataFrame({
        "review": X_test,
        "actual_sentiment": y_test,
        "predicted_sentiment": y_pred
    })

    # Identify incorrectly classified reviews
    error_df = error_df[
        error_df["actual_sentiment"] != error_df["predicted_sentiment"]
    ].copy()

    # Convert numerical labels into readable sentiment names
    sentiment_mapping = {
        0: "negative",
        1: "positive"
    }

    error_df["actual_sentiment"] = (
        error_df["actual_sentiment"].map(sentiment_mapping)
    )

    error_df["predicted_sentiment"] = (
        error_df["predicted_sentiment"].map(sentiment_mapping)
    )

    print("\nTotal Testing Reviews:", len(X_test))
    print("Misclassified Reviews:", len(error_df))

    if len(X_test) > 0:
        error_percentage = len(error_df) / len(X_test) * 100
        print(f"Error Percentage: {error_percentage:.2f}%")

    if not error_df.empty:
        print("\nSample Misclassified Reviews:")

        for _, row in error_df.head(5).iterrows():
            print("\nReview:")
            print(row["review"][:500])

            print("Actual Sentiment   :", row["actual_sentiment"])
            print("Predicted Sentiment:", row["predicted_sentiment"])
            print("-" * 60)
    else:
        print("\nNo misclassified reviews found.")

    print(BORDER)
    print("ERROR ANALYSIS COMPLETED")
    print(BORDER)

    return error_df

"""
----------------------------------------------------------------------
Function Name : plot_model_results
Description   : Visualizes the final sentiment classification model's
                performance using a confusion matrix and evaluation
                metric comparison chart.
Input         : metrics - Dictionary containing accuracy, precision,
                          recall, and F1-score.
                y_test  - Actual testing sentiment labels.
                y_pred  - Predicted testing sentiment labels.
Output        : Displays and saves the confusion matrix and evaluation
                metrics chart in the images directory.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def plot_model_results(metrics, y_test, y_pred):
    print(BORDER)
    print("STEP 20 : PLOTTING MODEL RESULTS")
    print(BORDER)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title("Confusion Matrix - Sentiment Analysis")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()

    confusion_path = os.path.join(
        IMAGES_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=300)
    plt.show()
    plt.close()

    # Plot evaluation metrics
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=metric_names,
        y=metric_values,
        hue=metric_names,
        legend=False
    )

    plt.ylim(0, 1)
    plt.title("Final Model Evaluation Metrics")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.tight_layout()

    metrics_path = os.path.join(
        IMAGES_DIR,
        "model_evaluation_metrics.png"
    )

    plt.savefig(metrics_path, dpi=300)
    plt.show()
    plt.close()

    print("\nConfusion Matrix Saved To:", confusion_path)
    print("Metrics Chart Saved To   :", metrics_path)

    print(BORDER)
    print("MODEL RESULT VISUALIZATION COMPLETED")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : predict_sentiment
Description   : Predicts the sentiment of a new movie review using
                the final trained sentiment classification pipeline.
Input         : model  - Final trained TF-IDF and classification pipeline.
                review - A new movie review entered as text.
Output        : Returns and displays the predicted sentiment as
                positive or negative.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def predict_sentiment(model, review):
    print(BORDER)
    print("STEP 21 : MOVIE REVIEW SENTIMENT PREDICTION")
    print(BORDER)

    # Clean the input review
    cleaned_review = clean_review_text(review)

    if not cleaned_review:
        print("Error: Please enter a valid movie review.")
        return None

    # Predict sentiment using the trained pipeline
    prediction = model.predict([cleaned_review])[0]

    # Convert numerical prediction into a sentiment label
    sentiment_mapping = {
        0: "Negative",
        1: "Positive"
    }

    predicted_sentiment = sentiment_mapping[prediction]

    print("\nOriginal Review:")
    print(review)

    print("\nCleaned Review:")
    print(cleaned_review)

    print("\nPredicted Sentiment:", predicted_sentiment)

    print(BORDER)

    return predicted_sentiment

"""
----------------------------------------------------------------------
Function Name : save_model
Description   : Saves the final trained sentiment classification
                pipeline to a file using Joblib.
Input         : model - Final trained sentiment classification pipeline.
Output        : Saves the trained model as a Joblib file in the
                project's models directory.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def save_model(model):
    print(BORDER)
    print("STEP 22 : SAVING FINAL SENTIMENT MODEL")
    print(BORDER)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully.")
    print("Model Path:", MODEL_PATH)

    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : save_outputs
Description   : Saves the model evaluation metrics, model comparison
                results, and misclassified reviews as CSV files.
Input         : metrics    - Dictionary containing evaluation metrics.
                results_df - DataFrame containing model comparison
                             results.
                error_df   - DataFrame containing misclassified reviews.
Output        : Saves three CSV files in the project's results directory.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def save_outputs(metrics, results_df, error_df):
    print(BORDER)
    print("STEP 23 : SAVING PROJECT OUTPUTS")
    print(BORDER)

    # Save evaluation metrics
    metrics_path = os.path.join(
        RESULTS_DIR,
        "evaluation_metrics.csv"
    )

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Score"]
    )

    metrics_df.to_csv(metrics_path, index=False)
    print("\nEvaluation metrics saved to:", metrics_path)

    # Save model comparison results
    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(comparison_path, index=False)
    print("Model comparison saved to:", comparison_path)

    # Save misclassified reviews
    errors_path = os.path.join(
        RESULTS_DIR,
        "misclassified_reviews.csv"
    )

    error_df.to_csv(errors_path, index=False)
    print("Misclassified reviews saved to:", errors_path)

    print(BORDER)
    print("PROJECT OUTPUTS SAVED SUCCESSFULLY")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : display_summary
Description   : Displays a final summary of the IMDB sentiment
                analysis project, including dataset statistics,
                selected model, evaluation metrics, and output paths.
Input         : df             - Processed Pandas DataFrame.
                best_model_name - Name of the selected model.
                metrics        - Dictionary containing evaluation
                                 metrics.
Output        : Displays the project completion summary and results.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def display_summary(df, best_model_name, metrics):
    print("\n" + BORDER)
    print(" STEP 24 : IMDB SENTIMENT ANALYSIS - PROJECT SUMMARY")
    print(BORDER)

    print("\nDATASET SUMMARY")
    print("-" * 60)
    print("Total Reviews       :", len(df))
    print("Positive Reviews    :", (df["sentiment"] == "positive").sum())
    print("Negative Reviews    :", (df["sentiment"] == "negative").sum())

    print("\nMODEL SUMMARY")
    print("-" * 60)
    print("Selected Model      :", best_model_name)

    print("\nFINAL EVALUATION METRICS")
    print("-" * 60)

    for metric_name, score in metrics.items():
        print(f"{metric_name.replace('_', ' ').title():20}: {score:.4f}")

    print("\nPROJECT OUTPUTS")
    print("-" * 60)
    print("Trained Model       :", MODEL_PATH)
    print("Images Directory    :", IMAGES_DIR)
    print("Results Directory   :", RESULTS_DIR)

    print("\nPROJECT STATUS: COMPLETED")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : display_footer
Description   : Displays the project completion message and summarizes
                the technologies used in the IMDB sentiment analysis
                project.
Input         : None
Output        : Displays a formatted project footer in the console.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""

def display_footer():
    print("\n" + BORDER)
    print("IMDB MOVIE REVIEW SENTIMENT ANALYSIS")
    print(BORDER)

    print("\nProject completed successfully!")

    print("\nTechnologies Used:")
    print("- Python")
    print("- Pandas and NumPy")
    print("- Matplotlib and Seaborn")
    print("- Natural Language Processing (NLP)")
    print("- Scikit-learn")
    print("- Joblib")

    print("\nKey Project Steps:")
    print("- Dataset exploration and cleaning")
    print("- Text preprocessing and feature extraction")
    print("- Machine learning model comparison")
    print("- Hyperparameter tuning and evaluation")
    print("- Sentiment prediction and result saving")

    print("\nAuthor: Atharv Tushar Bhosale")
    print(BORDER)
    print("THANK YOU FOR RUNNING THIS PROJECT!")
    print(BORDER)

"""
----------------------------------------------------------------------
Function Name : main
Description   : Executes the complete IMDB movie review sentiment
                analysis workflow, from project setup and dataset
                loading to model training, evaluation, and saving.
Input         : None
Output        : Executes the complete project workflow and displays
                the final project summary.
Author        : Atharv Tushar Bhosale
Date          : 23/10/2026
----------------------------------------------------------------------
"""
def main():
    # Display project header
    display_header()

    # Step 1: Set up project directories
    setup_project()

    # Step 2: Load dataset
    df = load_dataset()

    if df is None:
        print("Project stopped: Dataset could not be loaded.")
        return

    # Step 3: Understand dataset
    understand_dataset(df)

    # Step 4: Check dataset structure
    check_structure(df)

    # Step 5: Clean dataset
    df = clean_dataset(df)

    if df.empty:
        print("Project stopped: No valid records remain.")
        return

    # Step 6: Analyze sentiment distribution
    analyse_sentiment_distribution(df)

    # Step 7: Analyze review lengths
    analyse_review_lengths(df)

    # Step 8: Explore common words
    explore_common_words(df)

    # Step 9 and 10: Clean and preprocess reviews
    df = preprocess_reviews(df)

    # Step 11: Encode sentiment labels
    df = encode_labels(df)

    # Step 12: Create features
    # Use pipelines for model training to prevent data leakage.
    X_bow, X_tfidf, y, bow_vectorizer, tfidf_vectorizer = create_features(df)

    # Step 13: Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 14: Build baseline model
    baseline_model, baseline_accuracy = build_baseline_model(
        X_train, X_test, y_train, y_test
    )

    # Step 15: Create model pipelines
    models = create_model_pipelines()

    # Step 16: Compare models
    results_df, trained_models = compare_models(
        models, X_train, X_test, y_train, y_test
    )

    # Step 17: Select best model
    best_model_name, best_model = select_best_model(
        results_df, trained_models
    )

    # Step 18: Tune best model
    tuned_model, best_params = tune_best_model(
        best_model_name, best_model, X_train, y_train
    )

    # Step 19: Train final model
    final_model = train_final_model(
        tuned_model, X_train, y_train
    )

    # Step 20: Evaluate final model
    metrics, y_pred = evaluate_model(
        final_model, X_test, y_test
    )

    # Step 21: Analyze prediction errors
    error_df = analyse_errors(
        df, X_test, y_test, y_pred
    )

    # Step 22: Plot model results
    plot_model_results(metrics, y_test, y_pred)

    # Step 23: Test a sample review
    predict_sentiment(
        final_model,
        "This movie was amazing, with excellent acting and a great story!"
    )

    # Step 24: Save final model
    save_model(final_model)

    # Step 25: Save project outputs 
    save_outputs(metrics, results_df, error_df)

    # Step 26: Display project summary
    display_summary(df, best_model_name, metrics)

    # Step 27: Display project footer
    display_footer()

if __name__ == "__main__":
    main()

