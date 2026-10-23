# IMDB Movie Review Dataset

## 📌 Dataset Overview

This project uses the **IMDB Dataset of 50K Movie Reviews** for sentiment analysis using Natural Language Processing (NLP) and Machine Learning.

The dataset contains 50,000 movie reviews labeled as either positive or negative.

## 📥 Dataset Download

The dataset is not included in this GitHub repository because its file size exceeds GitHub's 25 MB file limit.

**Download the dataset from Kaggle:**

🔗 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

**Dataset Name:** IMDB Dataset of 50K Movie Reviews
**Source:** Kaggle
**File Name:** `IMDB_Dataset.csv`

## 📊 Dataset Information

| Attribute         | Description                |
| ----------------- | -------------------------- |
| Total Records     | 50,000 movie reviews       |
| Input Feature     | `review`                   |
| Target Column     | `sentiment`                |
| Sentiment Classes | Positive and Negative      |
| Problem Type      | Binary Classification      |
| Application       | NLP and Sentiment Analysis |

## 📂 Dataset Placement

After downloading the dataset, place the CSV file in the same directory as the main Python script, according to the current project configuration.

Expected project structure:

```text
Machine_Learning/
├── data/
│   └── README.md
├── images/
├── models/
├── results/
├── IMDB_Dataset.csv
└── IMDB_SentimentAnalysis.py
```

**Important:** Ensure the downloaded CSV filename matches `IMDB_Dataset.csv`. If Kaggle downloads it as `IMDB Dataset.csv`, rename it to match the filename expected by the Python code.

## 🎯 Purpose

The dataset is used to:

* Explore movie review text and sentiment labels.
* Clean and preprocess textual data.
* Extract text features using TF-IDF.
* Train and compare machine learning classification models.
* Evaluate model performance.
* Predict the sentiment of new movie reviews.

## 📝 Attribution

Dataset source: Lakshmipathi N, Kaggle.

Please refer to the original Kaggle dataset page for its licensing and usage conditions.

