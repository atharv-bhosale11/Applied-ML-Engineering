# 🏡 California Housing Price Prediction — Dataset

## 📌 Dataset Overview
 
This dataset contains information about housing districts in California. It includes demographic and housing-related features that can be used to predict the median house value.

The dataset is used in the **California Housing Price Prediction Using Machine Learning** project to train and evaluate regression models.

---

## 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset Name | California Housing Dataset |
| File Name | `california_housing.csv` |
| Total Records | 20,640 |
| Total Columns | 9 |
| Input Features | 8 |
| Target Variable | `target` |
| Missing Values | None reported |
| Problem Type | Regression |
| Application | House Price Prediction |

---

## 📂 Dataset Features

| Sr. No. | Feature | Description |
|---:|---|---|
| 1 | `MedInc` | Median income of households in the district |
| 2 | `HouseAge` | Median house age in the district |
| 3 | `AveRooms` | Average number of rooms per household |
| 4 | `AveBedrms` | Average number of bedrooms per household |
| 5 | `Population` | Population of the district |
| 6 | `AveOccup` | Average number of household members per household |
| 7 | `Latitude` | Geographic latitude of the district |
| 8 | `Longitude` | Geographic longitude of the district |
| 9 | `target` | Median house value, expressed in units of $100,000 |

---

## 🎯 Target Variable

The target variable is **`target`**.

It represents the median house value for a district, measured in units of $100,000.

For example:

- `target = 1.5` represents $150,000.
- `target = 2.0` represents $200,000.
- `target = 3.0` represents $300,000.

**Note:** The original California Housing dataset has an upper cap on recorded target values, so the target should not be interpreted as an unrestricted measure of actual market prices.

---

## 🧠 Machine Learning Application

This dataset can be used to:

- Predict median house values using regression algorithms.
- Study the relationship between household income and housing prices.
- Analyze the influence of housing characteristics on predicted values.
- Compare the performance of different regression models.
- Evaluate models using appropriate regression metrics.

---

## ⚙️ Dataset Usage

Load the CSV file using Pandas:

```python
import pandas as pd

df = pd.read_csv("california_housing.csv")

print(df.head())
print(df.shape)
print(df.info())
