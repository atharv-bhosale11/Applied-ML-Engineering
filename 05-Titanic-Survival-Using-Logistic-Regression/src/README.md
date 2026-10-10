# Source Code

This folder contains the Python source code used for the Titanic Survival Prediction project using Logistic Regression.

## Source File

**File Name:** `TitanicSurvivalUsingLogisticRegression.py`

## Description

The Python program implements a complete machine learning workflow for predicting passenger survival using Logistic Regression.

The program is organized into separate functions so that each stage of the machine learning process can be understood and executed independently.

## Functions Used

### 1. DisplayProjectInformation()

Displays the project name, dataset, algorithm, problem type and author information.

### 2. LoadDataset()

Loads the Titanic dataset from the `Titanic.csv` file.

### 3. DisplayDatasetInformation()

Displays the first five records, dataset shape, column names and dataset information.

### 4. CheckMissingValues()

Checks and displays missing values present in each column.

### 5. DisplayStatisticalSummary()

Displays the statistical summary of the dataset.

### 6. CleanDataset()

Performs data preprocessing by:
- Removing unnecessary columns
- Handling missing Age values
- Handling missing Fare values
- Handling missing Embarked values
- Processing the Sex column
- Encoding the Embarked column
- Converting boolean columns into integers

### 7. PrepareData()

Separates the dataset into independent features (`X`) and the target variable (`Y`).

### 8. SplitData()

Splits the dataset into training and testing data using `train_test_split()`.

### 9. TrainModel()

Creates and trains the Logistic Regression model.

### 10. DisplayModelCoefficients()

Displays the model intercept and coefficients for each feature.

### 11. SaveModel()

Saves the trained Logistic Regression model using Joblib.

### 12. LoadModel()

Loads the previously saved model from secondary storage.

### 13. GeneratePredictions()

Generates survival predictions using the trained model.

### 14. CalculateAccuracy()

Calculates and displays the model accuracy.

### 15. DisplayConfusionMatrix()

Displays the confusion matrix of the trained model.

### 16. ShowGraphs()

Displays:
- Survival Count Graph
- Age Distribution Graph

### 17. DisplayProjectSummary()

Displays the final project summary including algorithm, accuracy and problem type.

### 18. DisplayFooter()

Displays the project completion message.

### 19. main()

Controls the complete execution of the Titanic Survival Prediction program by calling each function step-by-step.

## Machine Learning Workflow

```text
Load Dataset
      ↓
Display Dataset Information
      ↓
Check Missing Values
      ↓
Display Statistical Summary
      ↓
Clean Dataset
      ↓
Prepare Features and Target
      ↓
Split Dataset
      ↓
Train Logistic Regression Model
      ↓
Display Model Coefficients
      ↓
Save Model
      ↓
Load Model
      ↓
Generate Predictions
      ↓
Calculate Accuracy
      ↓
Display Confusion Matrix
      ↓
Display Graphs
      ↓
Display Project Summary
