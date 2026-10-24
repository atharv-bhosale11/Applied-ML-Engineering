# IMDB Movie Review Sentiment Analysis Using NLP and Machine Learning

## 📌 Project Overview

This project implements a sentiment analysis system that classifies IMDB movie reviews as **positive** or **negative** using Natural Language Processing (NLP) and Machine Learning.

The project uses the IMDB Dataset of 50,000 Movie Reviews to explore review text, preprocess textual data, train and compare classification algorithms, evaluate model performance, and predict the sentiment of new reviews.

## 🎯 Objectives

* Explore and analyse the IMDB movie review dataset.
* Clean and preprocess review text.
* Convert text into numerical features using TF-IDF.
* Train and compare Machine Learning classification algorithms.
* Select and tune the best-performing model.
* Evaluate the final model using classification metrics.
* Analyse misclassified reviews.
* Generate visualizations to understand the data and model performance.
* Save the trained model for future predictions.

## 📂 Project Structure

```text
11-IMDB-Movie-Review-Sentiment-Analysis-Using-NLP/
│
├── IMDB_SentimentAnalysis.py
├── IMDB_Dataset.csv
├── requirements.txt
├── README.md
├── output
│
├── data/
│   └── README.md
│
├── images/
│   ├── README.md
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── common_words_positive.png
│   ├── common_words_negative.png
│   ├── confusion_matrix.png
│   └── model_evaluation_metrics.png
│
├── models/
│   ├── README.md
│   └── sentiment_model.joblib
│
└── results/
    ├── README.md
    ├── evaluation_metrics.csv
    ├── misclassified_reviews.csv
    └── Model_Comparison.csv
```

*Note: This structure reflects the files discussed for this project. Adjust it if your actual folder structure differs.*

## 📊 Dataset

**Dataset:** IMDB Dataset of 50K Movie Reviews

The dataset contains 50,000 movie reviews labelled as positive or negative.

| Column      | Description                                |
| ----------- | ------------------------------------------ |
| `review`    | The text of the movie review.              |
| `sentiment` | The sentiment label: positive or negative. |

The dataset is available on Kaggle:

[Download the IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Download the dataset and save the CSV file as `IMDB_Dataset.csv` in the location expected by the Python script.

The dataset is not included in this repository if its size exceeds GitHub's individual file upload limit.

## 🛠️ Technologies Used

* **Python** — Project implementation.
* **Pandas** — Dataset loading and analysis.
* **NumPy** — Numerical operations.
* **Scikit-learn** — Text feature extraction, model training, hyperparameter tuning, and evaluation.
* **TF-IDF** — Converts review text into numerical features.
* **Matplotlib** — Data and model visualizations.
* **Seaborn** — Statistical visualizations and confusion matrix heatmaps.
* **Joblib** — Saving and loading the trained model.

## 🤖 Machine Learning Algorithms

The project evaluates the following algorithms:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (Linear SVM)**

The algorithms are compared using the evaluation metrics implemented in the project. The best-performing model is selected and tuned before the final model is saved.

*The final selected algorithm should be identified from the actual model comparison results.*

## 🔄 Project Workflow

1. Set up the project and required directories.
2. Load the IMDB dataset.
3. Explore the dataset structure and sentiment distribution.
4. Analyse review lengths and frequently occurring words.
5. Clean and preprocess review text.
6. Convert review text into TF-IDF features.
7. Split the dataset into training and testing sets.
8. Train and compare Machine Learning models.
9. Select and tune the best-performing model.
10. Train the final model.
11. Evaluate model performance.
12. Analyse misclassified reviews.
13. Generate and save visualizations.
14. Save the trained model using Joblib.
15. Generate result files and demonstrate sentiment predictions.

## 📈 Visualizations

The `images/` directory contains the following visualizations:

| Visualization                    | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| Sentiment Distribution           | Shows the distribution of positive and negative reviews. |
| Review Length Distribution       | Compares review lengths by sentiment.                    |
| Common Words in Positive Reviews | Displays the 20 most common words in positive reviews.   |
| Common Words in Negative Reviews | Displays the 20 most common words in negative reviews.   |
| Confusion Matrix                 | Shows correct and incorrect classification results.      |
| Model Evaluation Metrics         | Visualizes accuracy, precision, recall, and F1-score.    |

## 📁 Results

The `results/` directory contains the generated analysis files.

| File                        | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `evaluation_metrics.csv`    | Stores the final model's evaluation metrics.                |
| `misclassified_reviews.csv` | Stores incorrectly classified reviews for error analysis.   |
| `Model_Comparison.csv`      | Stores the comparison of evaluated Machine Learning models. |

## 💾 Trained Model

**Model file:** `models/sentiment_model.joblib`

The trained model is saved using Joblib so that it can be loaded for future predictions without retraining.

The saved pipeline should contain the text feature extraction and classification steps used during training.

## ⚙️ Installation and Setup

### Prerequisites

* Python installed on your system.
* Git, if you want to clone the repository.
* The IMDB dataset downloaded from Kaggle.

### Step 1: Clone the Repository

```bash
git clone https://github.com/atharv-bhosale11/Applied-ML-Engineering.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Applied-ML-Engineering/11-IMDB-Movie-Review-Sentiment-Analysis-Using-NLP
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the IMDB dataset from Kaggle and save it as `IMDB_Dataset.csv` in the location expected by the script.

### Step 5: Run the Project

```bash
python IMDB_SentimentAnalysis.py
```

The script should generate the configured model, result files, and visualizations.

## 🔍 Example Prediction

Once the trained model has been saved, you can load it and predict the sentiment of a new review.

```python
import joblib

model = joblib.load("models/sentiment_model.joblib")

review = "This movie was fantastic, with excellent acting and a wonderful story."

prediction = model.predict([review])[0]

print("Predicted Sentiment:", prediction)
```

The prediction output depends on the label representation and output format used by the trained pipeline.

## 📋 Execution Output

The `output` file contains console output copied from the Command Prompt during project execution. It can be used as a reference for reviewing processing messages and results.

## 🚀 Future Improvements

* Improve text preprocessing and feature engineering.
* Experiment with additional classification algorithms.
* Evaluate more advanced NLP techniques.
* Test the model on new, unseen movie reviews.
* Develop a simple web interface for interactive sentiment prediction.

## 👨‍💻 Author

**Atharv Tushar Bhosale**

GitHub: [atharv-bhosale11](https://github.com/atharv-bhosale11)


