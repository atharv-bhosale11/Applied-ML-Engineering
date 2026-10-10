# 🚢 Titanic Survival Prediction using Logistic Regression

A machine learning classification project that predicts whether a passenger survived the Titanic disaster using **Logistic Regression**. The project covers dataset loading, data analysis, missing value handling, feature preprocessing, model training, model persistence, prediction, evaluation, and visualization.

## 🎯 Problem Statement

Given passenger information such as age, fare, sex, passenger class, number of siblings or spouses, number of parents or children, and embarkation point, predict whether the passenger survived the Titanic disaster.

The target variable is `Survived`:

- `0` – Did not survive
- `1` – Survived

The project demonstrates how Logistic Regression can be applied to a real-world **Binary Classification** problem.

## 🛠️ Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib

## 📊 Dataset

- **Dataset File:** `data/Titanic.csv`
- **Dataset:** Titanic Dataset
- **Total Records:** 1309
- **Total Columns:** 10
- **Target Variable:** `Survived`
- **Problem Type:** Binary Classification
- **Algorithm:** Logistic Regression

The dataset contains passenger-related information used to predict survival.

### Dataset Columns

```text
Passengerid
Age
Fare
Sex
sibsp
Parch
zero
Pclass
Embarked
Survived
```

### Target Variable

The `Survived` column represents the survival status of each passenger.

```text
0 → Did Not Survive
1 → Survived
```

## 🤖 Algorithm and Techniques

### Logistic Regression

**Logistic Regression** is used as the classification algorithm to predict whether a passenger survived or did not survive.

### Data Preprocessing

The project performs the following preprocessing operations:

- Removes unnecessary columns.
- Handles missing values.
- Processes numerical features such as `Age` and `Fare`.
- Processes the `Sex` feature.
- Encodes the `Embarked` categorical feature.
- Separates input features and target variable.

### Train-Test Split

The prepared dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

The training dataset is used to train the Logistic Regression model, while the testing dataset is used to evaluate model performance.

### Model Persistence

The trained model is saved using **Joblib** in `.pkl` format.

The saved model can be loaded later without retraining the model.

### Model Evaluation

The project evaluates the trained model using:

- Accuracy
- Confusion Matrix

## 🔍 Machine Learning Workflow

The complete workflow of the project is:

1. **Display Project Information**
   - Display project name, algorithm, dataset, and problem type.

2. **Load Dataset**
   - Load `Titanic.csv`.
   - Display the number of records and columns.

3. **Display Dataset Information**
   - Display sample records.
   - Display dataset shape.
   - Display column names.
   - Display data types and non-null values.

4. **Check Missing Values**
   - Identify missing values present in the dataset.

5. **Display Statistical Summary**
   - Generate statistical information for numerical features.

6. **Clean Dataset**
   - Remove unnecessary columns.
   - Handle missing values.
   - Process `Age`, `Fare`, `Embarked`, and `Sex`.

7. **Prepare Data**
   - Separate independent variables from the target variable.
   - Prepare the final feature matrix.

8. **Split Dataset**
   - Split the dataset into training and testing datasets.

9. **Train Model**
   - Train the Logistic Regression model using the training data.

10. **Display Model Coefficients**
    - Display the model intercept and feature coefficients.

11. **Save Model**
    - Save the trained model as `TitanicLogisticRegression.pkl`.

12. **Load Model**
    - Load the saved Logistic Regression model using Joblib.

13. **Generate Predictions**
    - Generate predictions for the testing dataset.

14. **Calculate Accuracy**
    - Calculate the accuracy of the trained model.

15. **Display Confusion Matrix**
    - Display the confusion matrix to evaluate classification performance.

16. **Generate Visualizations**
    - Display Survival Count.
    - Display Age Distribution.
    - Display Confusion Matrix.

17. **Display Project Summary**
    - Display the final model performance and project information.

## 📈 Model Performance

The Logistic Regression model achieved the following result on the testing dataset:

| Measure | Value |
|---|---:|
| Algorithm | Logistic Regression |
| Problem Type | Binary Classification |
| Test Accuracy | **76.72%** |
| Input Features | 8 |

### Confusion Matrix

```text
[[174  15]
 [ 46  27]]
```

The confusion matrix represents the comparison between actual and predicted survival classes.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| **Actual 0** | 174 | 15 |
| **Actual 1** | 46 | 27 |

Where:

- **174** → Correctly predicted as did not survive.
- **15** → Incorrectly predicted as survived.
- **46** → Incorrectly predicted as did not survive.
- **27** → Correctly predicted as survived.

## 📊 Visualizations

The project generates the following visualizations.

### 1. Survival Count

Displays the number of passengers who survived and did not survive.

### 2. Age Distribution

Displays the distribution of passenger ages using a histogram.

### 3. Confusion Matrix

Displays the classification performance of the Logistic Regression model.

## 📁 Project Structure

```text
05-Titanic-Survival-Using-Logistic-Regression
│
├── data
│   ├── Titanic.csv
│   └── README.md
│
├── models
│   ├── TitanicLogisticRegression.pkl
│   └── README.md
│
├── src
│   ├── TitanicSurvivalLogisticRegression.py
│   └── README.md
│
├── screenshots
│   ├── Confusion_Matrix.png
│   ├── Age_Distribution.png
│   ├── Survival_Count.png
│   └── README.md
│
├── requirements.txt
└── README.md
```

## 📦 Saved Model

The trained Logistic Regression model is saved in:

```text
models/TitanicLogisticRegression.pkl
```

The model is saved using Joblib and can be loaded without retraining.

Example:

```python
import joblib

model = joblib.load("TitanicLogisticRegression.pkl")
```

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd 05-Titanic-Survival-Using-Logistic-Regression
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Navigate to the Source Directory

```bash
cd src
```

### 5. Run the Project

```bash
python TitanicSurvivalLogisticRegression.py
```

Make sure the dataset is available at:

```text
data/Titanic.csv
```

## 🖥️ Sample Output

The program displays the complete Machine Learning workflow in the terminal.

Example:

```text
----------------------------------------------------------------------------------------------------
Step 1 : Project Information
----------------------------------------------------------------------------------------------------

Project Name       : Titanic Survival Prediction
Dataset             : Titanic.csv
Algorithm           : Logistic Regression
Problem Type       : Binary Classification
Target Variable    : Survived
```

The program continues with dataset information, missing value analysis, preprocessing, model training, predictions, accuracy, confusion matrix, visualizations, and project summary.

## 📸 Output Screenshots

The generated screenshots are available in the `screenshots` folder.

### Survival Count

Displays the number of passengers who survived and did not survive.

### Age Distribution

Displays the distribution of passenger ages.

### Confusion Matrix

Displays the actual versus predicted classification results.

## 📚 Learning Outcomes

This project provides practical understanding of:

- Loading datasets using pandas
- Exploring dataset structure
- Checking missing values
- Handling missing data
- Cleaning unnecessary features
- Preparing machine learning features
- Encoding categorical features
- Separating features and target variables
- Splitting data into training and testing sets
- Implementing Logistic Regression
- Understanding model coefficients
- Saving models using Joblib
- Loading saved machine learning models
- Generating predictions
- Evaluating classification accuracy
- Understanding confusion matrices
- Creating data visualizations
- Building a complete end-to-end Machine Learning project

## 🚀 Future Improvements

The project can be further improved by:

- Comparing Logistic Regression with other classification algorithms.
- Implementing feature scaling.
- Performing hyperparameter tuning.
- Using cross-validation for more reliable evaluation.
- Adding precision, recall, and F1-score.
- Adding ROC-AUC evaluation.
- Performing advanced feature engineering.
- Comparing multiple models and selecting the best-performing model.
- Improving the visualization section with additional analysis.

## 📝 Conclusion

The **Titanic Survival Prediction using Logistic Regression** project demonstrates a complete end-to-end Machine Learning classification workflow.

The project covers the complete process from **dataset loading and preprocessing to model training, prediction, evaluation, visualization, and model persistence**.

The Logistic Regression model achieved **76.72% test accuracy** and the trained model was saved using Joblib for future use.
 
This project provides a practical foundation for understanding how Machine Learning classification algorithms can be applied to real-world datasets.

## 👨‍💻 Author

**Atharv Tushar Bhosale**
