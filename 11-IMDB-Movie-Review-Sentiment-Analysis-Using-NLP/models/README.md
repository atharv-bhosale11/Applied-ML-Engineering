# Trained Sentiment Analysis Model

## 📌 Overview

This directory stores the trained Machine Learning model for the **IMDB Movie Review Sentiment Analysis Using NLP** project.

The model classifies movie reviews into two sentiment categories:

* **Positive**
* **Negative**

## 📂 Model File

| File Name                | Description                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `sentiment_model.joblib` | Saved, trained sentiment classification pipeline containing the TF-IDF vectorizer and classification algorithm. |

## ⚙️ Model Details

| Attribute           | Description                          |
| ------------------- | ------------------------------------ |
| Project             | IMDB Movie Review Sentiment Analysis |
| Task                | Binary Sentiment Classification      |
| Input               | Movie review text                    |
| Output              | Positive or Negative sentiment       |
| Feature Extraction  | TF-IDF                               |
| Framework           | Scikit-learn                         |
| Model Serialization | Joblib                               |

**Note:** The final classification algorithm and its performance metrics are documented in the project's model comparison and evaluation results.

## 🔄 Model Workflow

1. Accepts a movie review as input.
2. Cleans and preprocesses the review text.
3. Converts the text into numerical TF-IDF features.
4. Uses the trained classifier to predict sentiment.
5. Returns the predicted sentiment as positive or negative.

## 💻 Loading the Model

Use Joblib to load the saved model:

```python
import joblib

model = joblib.load("models/sentiment_model.joblib")
```

## 🧪 Example Prediction

```python
review = "This movie was amazing, with excellent acting and a great story!"

prediction = model.predict([review])[0]

sentiment = {
    0: "Negative",
    1: "Positive"
}

print("Predicted Sentiment:", sentiment[prediction])
```

**Important:** This example assumes that the saved pipeline was trained with `0` representing negative sentiment and `1` representing positive sentiment. Clean new reviews using the same preprocessing function used during training.

## 📦 Dependencies

* Python
* Scikit-learn
* Joblib

Install the required libraries using:

```bash
pip install scikit-learn joblib
```

## 📝 Notes

* The model is saved using Joblib.
* The TF-IDF vectorizer and classifier are stored together in the pipeline.
* The model can be reused without retraining, provided the required software dependencies are compatible.
* Only load Joblib files from trusted sources because loading serialized files can execute arbitrary code.

## 👨‍💻 Author

**Atharv Tushar Bhosale**
