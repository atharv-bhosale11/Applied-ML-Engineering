💳 Credit Card Fraud Detection

<p align="center">
  <b>A Machine Learning Classification Project using Python</b><br>
  Detecting potentially fraudulent credit card transactions with data analysis, model comparison, and classification-threshold tuning.
</p>

📌 Project Overview

Credit card fraud can cause financial losses for customers and financial institutions. This project explores how machine learning can help distinguish fraudulent transactions from legitimate ones using historical transaction data.

The project follows an end-to-end classification workflow: dataset validation, exploratory data analysis (EDA), preprocessing, model training, model comparison, hyperparameter tuning, threshold analysis, final evaluation, error analysis, and model saving.

Important: This is an educational portfolio project. Its predictions are not intended to be used as the sole basis for real financial decisions.

🎯 Objectives

Explore and validate a real-world transaction dataset.

Understand the severe class imbalance in fraud detection.

Build and compare multiple classification models.

Evaluate models using metrics appropriate for imbalanced data.

Tune Logistic Regression hyperparameters and the classification threshold.

Analyze false positives and false negatives.

Save the trained model and evaluation results for reuse.

📊 Dataset

Dataset: Credit Card Fraud Detection
Source: Kaggle — Credit Card Fraud Detection

The dataset contains European cardholder transactions from September 2013.

Dataset detail

Value

Total transactions in the original dataset

284,807

Total columns

31

Input features

30

Target column

Class

Legitimate transaction label

0

Fraudulent transaction label

1

Fraud transactions in the original dataset

492

Fraud share in the original dataset

Approximately 0.173%

The input columns include Time, Amount, and anonymized features V1 through V28. The Class column is the target.

Download the dataset

The CSV is not included in this repository. To run the project:

Download the dataset from Kaggle.

Extract creditcard.csv.

Place it in the location expected by the script (for example, data/creditcard.csv).

If necessary, update the dataset path in the Python script to match your local file location.

🧰 Technologies and Libraries

Python — project implementation

Pandas — dataset handling and analysis

NumPy — numerical operations

Matplotlib — plots and visualizations

Scikit-learn — preprocessing, classification, model selection, and evaluation

Imbalanced-learn — imbalance-handling experiments

Joblib — saving and loading the trained model

Install the project dependencies with:

pip install -r requirements.txt

🤖 Machine Learning Models

The project compares these classifiers:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

Linear Support Vector Classification (LinearSVC)

The workflow also includes Logistic Regression hyperparameter tuning, cross-validation, and classification-threshold analysis.

🔄 Project Workflow

Load the dataset.

Display dataset shape, columns, data types, and memory information.

Validate the dataset and inspect missing values and duplicate records.

Perform exploratory data analysis.

Separate input features and the target variable.

Split the data into training and testing sets.

Build a preprocessing pipeline.

Train the baseline Logistic Regression model.

Evaluate the baseline model.

Create additional classifiers.

Train and evaluate the additional models.

Compare model performance.

Tune Logistic Regression hyperparameters.

Evaluate the tuned model with cross-validation.

Prepare validation data for threshold selection.

Tune the classification threshold.

Retrain the selected model.

Evaluate the final model on the test set.

Analyze prediction errors.

Generate and save visualizations and result files.

Save and reload the trained model.

📏 Evaluation Metrics

The project reports metrics including:

Accuracy: Overall proportion of correct predictions.

Precision: Of transactions predicted as fraud, the proportion that were actually fraud.

Recall: Of actual fraud transactions, the proportion detected by the model.

F1-score: Harmonic mean of precision and recall.

Average Precision (AP): Summarizes performance across the precision-recall curve.

ROC-AUC: Measures how well the model ranks positive cases above negative cases.

Training time: Helps compare the computational cost of model fitting.

Because fraud represents a very small fraction of the dataset, accuracy by itself can be misleading. Precision, recall, F1-score, and Average Precision are especially useful for understanding fraud-detection performance. False negatives represent fraud transactions missed by the model; false positives represent legitimate transactions incorrectly flagged as fraud.

🖼️ Execution Screenshots

The following screenshots document key stages of the program run. They are stored in the images/ directory.

Output 2 — Dataset Validation and Statistical Summary

Shows missing-value checks, duplicate-record count, and descriptive statistics.



Output 3 — Validation, EDA, and Data Splitting

Shows class distribution, validation results, feature/target preparation, and training/testing split information.



Output 4 — Baseline Evaluation and Additional Model Training

Shows baseline Logistic Regression evaluation and the start of training and evaluating additional classifiers.



Output 5 — Model Comparison, Tuning, and Final Evaluation

Shows model comparison, hyperparameter tuning, cross-validation, threshold tuning, and final test-set evaluation.



Output 6 — Error Analysis and Project Completion

Shows false-positive and false-negative counts, imbalance-technique comparison, saved artifacts, and successful completion of the workflow.



📈 Visualizations and Results

The images/ folder contains plots such as:

Transaction class, amount, and time distributions

Top 15 important features

Confusion matrices

Precision-recall curve

The results/ folder stores generated evaluation files, which may include:

Model_Comparison.csv

Final_Model_Metrics.csv

Prediction_Errors.csv

Threshold_Analysis.csv

Tuned_Model_CrossValidation.csv

Feature_Importance.csv

Imbalance_Comparison.csv

The exact files depend on which outputs were generated by the script. See the README files inside images/ and results/ for details.

🗂️ Project Structure

09-Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv              # Download separately from Kaggle
├── images/
│   ├── README.md
│   ├── EDA_Analysis.png
│   ├── Feature_Importance.png
│   ├── Final_Confusion_Matrix.png
│   ├── Precision_Recall_Curve.png
│   └── output 2.png ... output 6.png
├── models/
│   ├── README.md
│   └── CreditCardFraudModel.pkl
├── results/
│   └── README.md
├── src/
│   ├── README.md
│   └── creditcardfrauddetection.py
├── requirements.txt
└── README.md

This is a representative structure; your repository may contain additional files or differently named image outputs.

⚙️ Setup and Run

1. Clone the repository

Replace the placeholder with your repository URL:

git clone <your-github-repository-url>
cd 09-Credit-Card-Fraud-Detection

2. Create and activate a virtual environment (recommended)

python -m venv venv

On Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Download and place the dataset

Download creditcard.csv from the Kaggle link above and put it at the path expected by the script.

5. Run the project

python src/creditcardfrauddetection.py

If your script is stored in a different location, adjust the command accordingly. Confirm that the dataset path in the script matches where you saved the CSV.

💾 Saved Model

The trained model is saved as:

models/CreditCardFraudModel.pkl

The project uses Joblib to serialize the model. For predictions, load the saved object using Joblib and provide input data with the same feature columns and preprocessing expectations used during training.

🧠 Key Learning Outcomes

Working with a large tabular dataset

Detecting missing values and duplicate records

Exploring highly imbalanced classification data

Building preprocessing pipelines

Training and comparing multiple classifiers

Using cross-validation and hyperparameter search

Understanding precision-recall trade-offs

Choosing classification thresholds

Interpreting confusion matrices and prediction errors

Saving a trained model and generated results

🚀 Possible Future Improvements

Evaluate additional algorithms and imbalance-handling strategies.

Compare precision and recall at business-relevant operating thresholds.

Add explainability methods for individual predictions.

Build a simple web interface for demonstration.

Add automated tests and a reproducible experiment configuration.

Evaluate performance on newer data before any real-world use.

📚 References

Kaggle dataset — Credit Card Fraud Detection

Scikit-learn documentation

Imbalanced-learn documentation

👨‍💻 Author

Atharv Tushar Bhosale

Credit Card Fraud Detection — Machine Learning Classification Project
