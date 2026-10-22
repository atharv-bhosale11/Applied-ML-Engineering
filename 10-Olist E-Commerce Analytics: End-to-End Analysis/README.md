# 🛒 Olist Brazilian E-Commerce Analysis & Late Delivery Prediction

## 📌 Project Overview

This project analyzes the **Olist Brazilian E-Commerce Public Dataset** to uncover business insights and develop a machine learning model for predicting late deliveries.

The project combines **Exploratory Data Analysis (EDA), business analytics, customer segmentation, and machine learning** to understand order patterns, sales performance, customer satisfaction, delivery delays, and factors associated with late deliveries.

The analysis aims to transform raw e-commerce data into meaningful insights that can help businesses understand their operations and make data-driven decisions.

## 🎯 Project Objectives

* Analyze monthly order trends and sales performance.
* Evaluate delivery performance and identify delivery delays.
* Understand customer satisfaction through review scores.
* Analyze product category sales.
* Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
* Train and evaluate machine learning models for late delivery prediction.
* Identify important features associated with model predictions.
* Generate CSV outputs and visualizations for further analysis and reporting.

## 📂 Dataset

This project uses the **Olist Brazilian E-Commerce Public Dataset**, which contains information about orders, customers, sellers, products, payments, reviews, and geolocation.

The dataset consists of nine CSV files:

| Dataset File                            | Description                                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| `olist_customers_dataset.csv`           | Customer IDs, unique customer IDs, ZIP code prefixes, cities, and states              |
| `olist_geolocation_dataset.csv`         | Brazilian ZIP code prefixes and geographical coordinates                              |
| `olist_order_items_dataset.csv`         | Order items, products, sellers, prices, and freight values                            |
| `olist_order_payments_dataset.csv`      | Payment methods, installments, and payment values                                     |
| `olist_order_reviews_dataset.csv`       | Customer reviews and review scores                                                    |
| `olist_orders_dataset.csv`              | Order status and purchase, approval, shipping, delivery, and estimated delivery dates |
| `olist_products_dataset.csv`            | Product attributes, dimensions, weight, and category information                      |
| `olist_sellers_dataset.csv`             | Seller IDs, ZIP code prefixes, cities, and states                                     |
| `product_category_name_translation.csv` | Portuguese product category names and their English translations                      |

**Dataset source:** [Olist Brazilian E-Commerce Public Dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset files are stored in the `Olist/` directory. Refer to [`Olist/README.md`](Olist/README.md) for dataset-specific information.

## 🛠️ Technologies Used

* **Python** — Data analysis and machine learning.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical operations.
* **Matplotlib** — Data visualization.
* **Seaborn** — Statistical data visualization.
* **Scikit-learn** — Machine learning model training and evaluation.
* **Joblib** — Saving and loading the trained model.
* **CSV** — Storing processed analysis results.

## ⚙️ Project Workflow

### 1. Data Loading and Preparation

* Load the nine Olist CSV datasets.
* Inspect dataset dimensions, columns, and data types.
* Convert relevant date columns into appropriate datetime formats.
* Merge relevant datasets to create an analysis-ready dataset.

### 2. Exploratory Data Analysis

* Examine order volume over time.
* Analyze monthly sales trends.
* Investigate delivery delays and delivery timing.
* Study customer review score distributions.
* Explore product category sales and customer purchasing patterns.

### 3. Business Insights

* Analyze delivery performance.
* Identify product categories with high sales.
* Examine customer satisfaction.
* Perform RFM-based customer segmentation.
* Export selected business analysis results to CSV files.

### 4. Machine Learning

Train and compare the following models for late delivery prediction:

* Dummy Classifier — Baseline model.
* Logistic Regression.
* Decision Tree Classifier.
* Random Forest Classifier.
* Gradient Boosting Classifier.

### 5. Model Evaluation

Evaluate model performance using:

* Area Under the Precision-Recall Curve (AUPRC)
* Precision
* Recall
* F1-score
* Confusion Matrix
* Precision-Recall Curve
* Feature Importance

The model selection process considers AUPRC because the prediction task involves distinguishing late deliveries from on-time or early deliveries.

### 6. Output Generation

* Save the trained prediction model.
* Export business analysis results to CSV files.
* Generate visualizations for business analysis and model evaluation.
* Capture execution logs for reference.

## 📊 Key Findings

The analysis produced the following results from the current project execution:

* **Total orders:** 99,441
* **Unique order IDs:** 99,441
* **Customers:** 96,096
* **Average customer review score:** Approximately 4.09 out of 5
* **Selected machine learning model:** Gradient Boosting Classifier
* **Validation AUPRC:** Approximately 0.4679
* **Test AUPRC:** Approximately 0.4636
* **Test precision:** Approximately 0.5281
* **Test recall:** Approximately 0.0647
* **Test F1-score:** Approximately 0.1153

### Interpretation

The Gradient Boosting model achieved a test AUPRC of approximately 0.4636. However, its recall was low, indicating that it failed to identify many actual late deliveries.

These results show that model selection should not rely on a single evaluation metric. Further improvements are needed before the model could be considered useful for operational late delivery prediction.

The findings reflect the current implementation and may change as data preparation, feature engineering, and model parameters are improved.

## 📈 Project Visualizations

The `images/` directory contains visualizations generated during analysis and model evaluation.



## 📁 Generated Outputs

### Business Analysis Results

The `results/` directory contains the following CSV files:

| File                             | Description                                      |
| -------------------------------- | ------------------------------------------------ |
| `state_delivery_performance.csv` | Delivery performance summarized by state         |
| `product_category_sales.csv`     | Sales information summarized by product category |
| `rfm_customer_segments.csv`      | Customer segments generated through RFM analysis |

These files can be used for further analysis, reporting, and business intelligence dashboards.

Refer to [`outputs/README.md`](outputs/README.md) for details.

### Trained Model

The `models/` directory contains:

| File                           | Description                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| `Best_late_delivery_model.pkl` | Saved machine learning model selected for late delivery prediction |

Refer to [`models/README.md`](models/README.md) for additional details.

## 🚀 How to Run the Project

### Prerequisites

Install Python and ensure that `pip` is available.

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

Replace `<your-github-repository-url>` with your actual GitHub repository URL.

### Step 2: Navigate to the Project Directory

```bash
cd Machine_Learning
```

Adjust the directory name if your cloned repository uses a different name.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Check the Dataset

Ensure that all nine CSV files are available in the `Olist/` directory.

### Step 5: Execute the Analysis

If your Python script is named `olist_analysis.py` and is located in the project root, run:

```bash
python olist_analysis.py
```

**Important:** If you move the script into a `src/` directory, update the dataset, output, and model paths in the code before running it.

## ⚠️ Limitations

* The model's recall is low, so many late deliveries are missed.
* Model performance depends on the quality of the input data and selected features.
* Feature importance indicates how a model uses features; it does not establish causation.
* Review scores are generally recorded after an order is delivered. Using them to predict delivery delays may introduce **data leakage**. A realistic prediction system should use only information available at prediction time.
* The current results require further validation before the model can be used for real-world operational decisions.

## 🔮 Future Improvements

* Improve recall for late deliveries through threshold tuning and class imbalance techniques.
* Perform feature engineering using only information available before delivery.
* Investigate and eliminate potential data leakage.
* Tune model hyperparameters using cross-validation.
* Evaluate additional classification algorithms.
* Compare models using appropriate metrics for imbalanced classification.
* Add explainability techniques to better understand model predictions.
* Build an interactive dashboard using Power BI or another visualization tool.
* Develop a deployment interface for delivery risk prediction.

## 📚 References

* **Dataset:** [Olist Brazilian E-Commerce Public Dataset — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Machine Learning Library:** [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* **Data Analysis Library:** [Pandas Documentation](https://pandas.pydata.org/docs/)

## 👨‍💻 Author

**Atharv Tushar Bhosale**
