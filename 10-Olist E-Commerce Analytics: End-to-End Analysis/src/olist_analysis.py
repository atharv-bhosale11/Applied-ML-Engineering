"""
Project Name: Olist E-Commerce Analytics
Project Type: End-to-End Data Analysis and Machine Learning

Description:
This project analyzes the Olist Brazilian E-Commerce dataset to understand
sales performance, customer behavior, delivery performance, and customer reviews.

Objectives:
1. Analyze sales and order trends.
2. Understand customer purchasing behavior.
3. Analyze delivery performance and delays.
4. Identify product categories with strong sales.
5. Segment customers using RFM analysis.
6. Build machine learning models to predict late deliveries.
7. Generate business insights and recommendations.
8. Export analysis results for Power BI visualization.

Technologies Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Power BI

Expected Outcome:
An end-to-end analytics project with data cleaning, exploratory data analysis,
customer segmentation, delivery-delay prediction, and business recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, recall_score, f1_score
import joblib
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

path = "Olist/"

"""
----------------------------------------------------------------------
Function Name : load_datasets
Description   : Loads all nine Olist e-commerce CSV datasets into
                Pandas DataFrames.
Input         : None
Output        : Returns nine Pandas DataFrames containing customer,
                geolocation, order item, payment, review, order,
                product, seller, and category translation data.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def load_datasets():

    customers   = pd.read_csv(path + "olist_customers_dataset.csv")
    geolocation = pd.read_csv(path + "olist_geolocation_dataset.csv")
    items       = pd.read_csv(path + "olist_order_items_dataset.csv")
    payments    = pd.read_csv(path + "olist_order_payments_dataset.csv")
    reviews     = pd.read_csv(path + "olist_order_reviews_dataset.csv")
    orders      = pd.read_csv(path + "olist_orders_dataset.csv")
    products    = pd.read_csv(path + "olist_products_dataset.csv")
    sellers     = pd.read_csv(path + "olist_sellers_dataset.csv")
    category    = pd.read_csv(path + "product_category_name_translation.csv")

    print("All nine datasets loaded successfully!")

    return customers, geolocation, items, payments, reviews, orders, products, sellers, category

"""
----------------------------------------------------------------------
Function Name : understand_datasets
Description   : Displays the structure, dimensions, column names,
                data types, and statistical information of all
                nine Olist datasets.
Input         : Nine Pandas DataFrames returned by load_datasets().
Output        : Displays dataset information for understanding
                the structure and contents of each dataset.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def understand_datasets(customers, geolocation, items, payments, reviews,
                        orders, products, sellers, category):

    datasets = {
        "Customers": customers,
        "Geolocation": geolocation,
        "Order Items": items,
        "Payments": payments,
        "Reviews": reviews,
        "Orders": orders,
        "Products": products,
        "Sellers": sellers,
        "Category Translation": category
    }

    for name, df in datasets.items():

        print("\n", "=" * 60)
        print("Dataset Name:", name)
        print("=" * 60)

        print("Number of Rows:", df.shape[0])
        print("Number of Columns:", df.shape[1])

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nData Types:")
        print(df.dtypes)

        print("\nStatistical Information:")
        print(df.describe(include="all").T)

"""
----------------------------------------------------------------------
Function Name : check_missing_values
Description   : Identifies and displays the number and percentage
                of missing values in all nine Olist datasets.
Input         : Nine Pandas DataFrames returned by load_datasets().
Output        : Displays missing value counts and percentages for
                each column in every dataset.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def check_missing_values(customers, geolocation, items, payments, reviews,
                          orders, products, sellers, category):

    datasets = {
        "Customers": customers,
        "Geolocation": geolocation,
        "Order Items": items,
        "Payments": payments,
        "Reviews": reviews,
        "Orders": orders,
        "Products": products,
        "Sellers": sellers,
        "Category Translation": category
    }

    for name, df in datasets.items():

        print("\n", "=" * 60)
        print("Dataset Name:", name)
        print("=" * 60)

        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100

        missing_data = pd.DataFrame({
            "Missing Values": missing_count,
            "Missing Percentage": missing_percentage
        })

        missing_data = missing_data[missing_data["Missing Values"] > 0]

        if missing_data.empty:
            print("No missing values found.")
        else:
            print(missing_data)

"""
----------------------------------------------------------------------
Function Name : check_duplicates
Description   : Identifies and displays the number and percentage
                of duplicate rows in all nine Olist datasets.
Input         : Nine Pandas DataFrames returned by load_datasets().
Output        : Displays duplicate row counts and percentages for
                each dataset.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def check_duplicates(customers, geolocation, items, payments, reviews,
                     orders, products, sellers, category):

    datasets = {
        "Customers": customers,
        "Geolocation": geolocation,
        "Order Items": items,
        "Payments": payments,
        "Reviews": reviews,
        "Orders": orders,
        "Products": products,
        "Sellers": sellers,
        "Category Translation": category
    }

    for name, df in datasets.items():

        print("\n", "=" * 60)
        print("Dataset Name:", name)
        print("=" * 60)

        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100

        print("Total Rows:", len(df))
        print("Duplicate Rows:", duplicate_count)
        print("Duplicate Percentage:", round(duplicate_percentage, 2), "%")

        if duplicate_count == 0:
            print("No duplicate rows found.")

"""
----------------------------------------------------------------------
Function Name : clean_datasets
Description   : Cleans Olist datasets by converting date columns,
                translating product categories, and aggregating
                geolocation records by ZIP code prefix.
Input         : Nine Pandas DataFrames returned by load_datasets().
Output        : Returns all nine datasets after cleaning.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def clean_datasets(customers, geolocation, items, payments, reviews,
                   orders, products, sellers, category):

    # Convert date columns to datetime format
    date_columns = {
        "orders": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ],
        "reviews": [
            "review_creation_date",
            "review_answer_timestamp"
        ],
        "items": ["shipping_limit_date"]
    }

    datasets = {
        "orders": orders,
        "reviews": reviews,
        "items": items
    }

    for name, columns in date_columns.items():
        for column in columns:
            datasets[name][column] = pd.to_datetime(
                datasets[name][column], errors="coerce"
            )

    # Translate product categories into English
    products = products.merge(
        category,
        on="product_category_name",
        how="left"
    )

    products["product_category_name_english"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
    )

    # Aggregate geolocation records by ZIP code prefix
    geolocation = geolocation.groupby(
        "geolocation_zip_code_prefix", as_index=False
    ).agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
        "geolocation_city": "first",
        "geolocation_state": "first"
    })

    print("Dataset cleaning completed successfully!")

    return customers, geolocation, items, payments, reviews, orders, products, sellers, category

"""
----------------------------------------------------------------------
Function Name : merge_datasets
Description   : Merges the Olist datasets into a master table
                containing one row per order.
Input         : Nine cleaned Olist DataFrames.
Output        : Returns the master DataFrame with order, customer,
                item, payment, review, and product category details.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def merge_datasets(customers, geolocation, items, payments, reviews,
                   orders, products, sellers, category):

    # Aggregate order items
    order_items = items.groupby("order_id", as_index=False).agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum")
    )

    # Aggregate payment information
    order_payments = payments.groupby("order_id", as_index=False).agg(
        total_payment=("payment_value", "sum"),
        payment_installments=("payment_installments", "max")
    )

    # Aggregate review scores
    order_reviews = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )

    # Merge order and customer information
    master = orders.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # Merge aggregated order summaries
    for dataset in [order_items, order_payments, order_reviews]:
        master = master.merge(
            dataset,
            on="order_id",
            how="left",
            validate="one_to_one"
        )

    # Merge product information with order items
    item_products = items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Aggregate product categories by order
    category_sales = item_products.groupby(
        "order_id", as_index=False
    ).agg(
        product_categories=(
            "product_category_name_english",
            lambda values: ", ".join(values.dropna().unique())
        )
    )

    # Add product categories to the master table
    master = master.merge(
        category_sales,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Validate the master table
    print("Original Orders:", len(orders))
    print("Master Table Rows:", len(master))
    print("Unique Order IDs:", master["order_id"].nunique())

    if len(master) == len(orders) and master["order_id"].is_unique:
        print("Validation Successful: One row per order.")
    else:
        print("Warning: Check duplicate orders or missing records.")

    return master

"""
----------------------------------------------------------------------
Function Name : feature_engineering
Description   : Creates new features related to delivery performance,
                freight costs, product dimensions, and purchase dates.
Input         : Master DataFrame and cleaned product DataFrame.
Output        : Returns the master DataFrame with additional features.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def feature_engineering(master,items,products):

    # Calculate delivery duration in days
    master["delivery_days"] = (
        master["order_delivered_customer_date"]
        - master["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Calculate delivery delay in days
    master["delivery_delay_days"] = (
        master["order_delivered_customer_date"]
        - master["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Identify late deliveries
    master["is_late_delivery"] = (
        master["delivery_delay_days"] > 0
    ).astype("int")

    # Calculate freight ratio
    master["freight_ratio"] = (
        master["total_freight"]
        / master["total_price"].replace(0, np.nan)
    )

    # Calculate purchase time features
    master["purchase_year"] = (
        master["order_purchase_timestamp"].dt.year
    )

    master["purchase_month"] = (
        master["order_purchase_timestamp"].dt.month
    )

    master["purchase_day_of_week"] = (
        master["order_purchase_timestamp"].dt.dayofweek
    )

    # Calculate product volume in cubic centimeters
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    # Calculate average product weight and volume per order
    item_features = master[["order_id"]].merge(
        products[[
            "product_id",
            "product_weight_g",
            "product_volume_cm3"
        ]],
        how="cross"
    ) if False else None

    print("Feature engineering completed successfully!")

    return master

"""
----------------------------------------------------------------------
Function Name : perform_eda
Description   : Analyzes sales, order trends, delivery performance,
                and customer review scores using visualizations.
Input         : Master DataFrame containing order and sales features.
Output        : Displays EDA charts for business analysis.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def perform_eda(master):

    # 1. Monthly order trends
    monthly_orders = (
        master.groupby(master["order_purchase_timestamp"].dt.to_period("M"))
        ["order_id"].nunique()
    )

    plt.figure(figsize=(12, 5))
    monthly_orders.plot(kind="line", marker="o")
    plt.title("Monthly Order Trends")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Monthly sales trends
    monthly_sales = (
        master.groupby(master["order_purchase_timestamp"].dt.to_period("M"))
        ["total_price"].sum()
    )

    plt.figure(figsize=(12, 5))
    monthly_sales.plot(kind="line", marker="o")
    plt.title("Monthly Sales Trends")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Delivery delay distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        master["delivery_delay_days"].dropna(),
        bins=30,
        kde=True
    )
    plt.title("Delivery Delay Distribution")
    plt.xlabel("Days Late (Negative = Early)")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.show()

    # 4. Customer review score distribution
    review_counts = master["review_score"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    review_counts.plot(kind="bar")
    plt.title("Customer Review Score Distribution")
    plt.xlabel("Review Score")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("Exploratory Data Analysis completed successfully!")

"""
----------------------------------------------------------------------
Function Name : perform_eda
Description   : Analyzes sales, order trends, delivery performance,
                and customer review scores using visualizations.
Input         : Master DataFrame containing order and sales features.
Output        : Displays EDA charts for business analysis.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def perform_eda(master):

    # 1. Monthly order trends
    monthly_orders = (
        master.groupby(master["order_purchase_timestamp"].dt.to_period("M"))
        ["order_id"].nunique()
    )

    plt.figure(figsize=(12, 5))
    monthly_orders.plot(kind="line", marker="o")
    plt.title("Monthly Order Trends")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Monthly sales trends
    monthly_sales = (
        master.groupby(master["order_purchase_timestamp"].dt.to_period("M"))
        ["total_price"].sum()
    )

    plt.figure(figsize=(12, 5))
    monthly_sales.plot(kind="line", marker="o")
    plt.title("Monthly Sales Trends")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Delivery delay distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        master["delivery_delay_days"].dropna(),
        bins=30,
        kde=True
    )
    plt.title("Delivery Delay Distribution")
    plt.xlabel("Days Late (Negative = Early)")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.show()

    # 4. Customer review score distribution
    review_counts = master["review_score"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    review_counts.plot(kind="bar")
    plt.title("Customer Review Score Distribution")
    plt.xlabel("Review Score")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("Exploratory Data Analysis completed successfully!")

def create_target(master):
    """
    ----------------------------------------------------------------------
    Function Name : create_target
    Description   : Creates a target variable for late delivery prediction.
    Input         : master DataFrame
    Output        : Delivered orders with the target column
    Author        : Atharv Tushar Bhosale
    Date          : 21/10/2026
    ----------------------------------------------------------------------
    """

    data = master[master["order_status"] == "delivered"].copy()

    data = data.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    data["is_late_delivery"] = (
        data["order_delivered_customer_date"]
        > data["order_estimated_delivery_date"]
    ).astype(int)

    print("Target variable created successfully!")
    print(data["is_late_delivery"].value_counts())

    return data

def remove_leakage(data):
    """
    ----------------------------------------------------------------------
    Function Name : remove_leakage
    Description   : Removes post-delivery columns while preserving the target.
    Input         : DataFrame containing the target variable
    Output        : DataFrame without post-delivery leakage columns
    Author        : Atharv Tushar Bhosale
    Date          : 21/10/2026
    ----------------------------------------------------------------------
    """

    leakage_columns = [
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "delivery_days",
        "delivery_delay_days"
    ]

    data = data.drop(
        columns=leakage_columns,
        errors="ignore"
    )

    print("Leakage columns removed successfully!")
    print("Target column preserved:", "is_late_delivery" in data.columns)

    return data

"""
----------------------------------------------------------------------
Function Name : prepare_ml_data
Description   : Selects suitable input features and splits the dataset
                chronologically into training, validation, and test sets.
Input         : DataFrame returned by remove_leakage().
Output        : Returns training, validation, and test features and
                their corresponding target labels.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def prepare_ml_data(data):

    # Select target variable
    target = "is_late_delivery"

    # Select columns that should not be used as input features
    exclude_columns = [
        target,
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date"
    ]

    # Select numeric features
    X = data.drop(
        columns=exclude_columns,
        errors="ignore"
    ).select_dtypes(include=np.number)

    y = data[target]

    # Sort orders by purchase date
    data = data.sort_values("order_purchase_timestamp")

    X = X.loc[data.index]
    y = y.loc[data.index]

    # Split chronologically: 70% training, 15% validation, 15% testing
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print("ML data preparation completed!")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_validation))
    print("Testing samples:", len(X_test))
    print("Number of features:", X_train.shape[1])

    return (
        X_train, X_validation, X_test,
        y_train, y_validation, y_test
    )

"""
----------------------------------------------------------------------
Function Name : build_models
Description   : Creates a baseline classifier and four machine
                learning models for late-delivery prediction.
Input         : Training features (X_train) and target labels (y_train).
Output        : Returns a dictionary containing trained ML models.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def build_models(X_train, y_train):

    models = {
        "Dummy Classifier": DummyClassifier(strategy="prior"),

        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        ),

        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            )
        ),

        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        "Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=42)
        )
    }

    # Train each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(name, "trained successfully!")

    return models

"""
----------------------------------------------------------------------
Function Name : compare_models
Description   : Compares trained classification models using AUPRC,
                recall, and F1-score on validation data.
Input         : Dictionary of trained models, validation features,
                and validation target labels.
Output        : Returns a DataFrame containing model evaluation scores.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def compare_models(models, X_validation, y_validation):

    results = []

    for name, model in models.items():

        # Predict class probabilities
        if hasattr(model, "predict_proba"):
            y_probability = model.predict_proba(X_validation)[:, 1]
        else:
            y_probability = model.decision_function(X_validation)

        # Predict classes using the default threshold
        y_prediction = model.predict(X_validation)

        results.append({
            "Model": name,
            "AUPRC": average_precision_score(
                y_validation, y_probability
            ),
            "Recall": recall_score(
                y_validation, y_prediction, zero_division=0
            ),
            "F1-Score": f1_score(
                y_validation, y_prediction, zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="AUPRC",
        ascending=False
    ).reset_index(drop=True)

    print("\nModel Comparison Results:")
    print(results_df.round(4).to_string(index=False))

    return results_df

"""
----------------------------------------------------------------------
Function Name : select_best_model
Description   : Selects the best classification model based on
                validation AUPRC, recall, and F1-score.
Input         : Dictionary of trained models and validation comparison
                results returned by compare_models().
Output        : Returns the selected model name and trained model.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def select_best_model(models, model_results):

    # Rank models by AUPRC, then recall, then F1-score
    ranked_results = model_results.sort_values(
        by=["AUPRC", "Recall", "F1-Score"],
        ascending=False
    )

    best_model_name = ranked_results.iloc[0]["Model"]
    best_model = models[best_model_name]

    print("Best Model:", best_model_name)
    print("\nValidation Performance:")
    print(
        ranked_results.iloc[0][
            ["AUPRC", "Recall", "F1-Score"]
        ].to_string()
    )

    return best_model_name, best_model

"""
----------------------------------------------------------------------
Function Name : train_best_model
Description   : Retrains the selected machine learning model using
                the combined training and validation datasets.
Input         : Selected model, training features and labels,
                and validation features and labels.
Output        : Returns the retrained best-performing model.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def train_best_model(best_model, X_train, X_validation,
                     y_train, y_validation):

    # Combine training and validation datasets
    X_combined = pd.concat(
        [X_train, X_validation],
        axis=0
    )

    y_combined = pd.concat(
        [y_train, y_validation],
        axis=0
    )

    # Train the selected model
    best_model.fit(X_combined, y_combined)

    print("Best model retrained successfully!")
    print("Training samples:", len(X_combined))

    return best_model

"""
----------------------------------------------------------------------
Function Name : save_model
Description   : Saves the trained machine learning model to a file
                for future use.
Input         : Trained machine learning model and output file path.
Output        : Saves the model file and returns its file path.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def save_model(best_model, file_name="Best_late_delivery_model.pkl"):

    joblib.dump(best_model, file_name)

    print("Model saved successfully!")
    print("File:", file_name)

    return file_name

"""
----------------------------------------------------------------------
Function Name : load_model
Description   : Loads a previously saved machine learning model
                from a file.
Input         : File path of the saved machine learning model.
Output        : Returns the loaded machine learning model.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def load_model(file_name="best_late_delivery_model.pkl"):

    best_model = joblib.load(file_name)

    print("Model loaded successfully!")
    print("File:", file_name)

    return best_model

"""
----------------------------------------------------------------------
Function Name : evaluate_model
Description   : Evaluates the selected model on the test dataset
                and compares its performance with a baseline model.
Input         : Trained model, training labels, test features,
                and test target labels.
Output        : Returns a DataFrame containing model evaluation metrics.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def evaluate_model(best_model, y_train, X_test, y_test):

    # Predict late-delivery probabilities and classes
    y_probability = best_model.predict_proba(X_test)[:, 1]
    y_prediction = best_model.predict(X_test)

    # Create a baseline that always predicts the majority class
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)

    baseline_prediction = baseline.predict(
        np.zeros((len(y_test), 1))
    )
    baseline_probability = baseline.predict_proba(
        np.zeros((len(y_test), 1))
    )[:, 1]

    # Calculate evaluation metrics
    results = []

    for name, actual, predicted, probability in [
        ("Best Model", y_test, y_prediction, y_probability),
        (
            "Baseline",
            y_test,
            baseline_prediction,
            baseline_probability
        )
    ]:
        results.append({
            "Model": name,
            "AUPRC": average_precision_score(actual, probability),
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1-Score": f1_score(
                actual, predicted, zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    print("\nTest Evaluation Results:")
    print(results_df.round(4).to_string(index=False))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=["On-time/Early", "Late"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))

    return results_df

"""
----------------------------------------------------------------------
Function Name : plot_results
Description   : Visualizes model performance using metric comparison,
                a precision-recall curve, a confusion matrix, and
                permutation-based feature importance.
Input         : Model comparison results, trained model, test features,
                and test target labels.
Output        : Displays four visualizations for model interpretation.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def plot_results(test_results, best_model, X_test, y_test):

    # 1. Compare model evaluation metrics
    metrics = ["AUPRC", "Precision", "Recall", "F1-Score"]

    comparison = test_results.set_index("Model")[metrics]

    comparison.plot(kind="bar", figsize=(10, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 2. Precision-Recall curve
    y_probability = best_model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test, y_probability
    )

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()

    # 3. Confusion matrix
    y_prediction = best_model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_prediction,
        display_labels=["On-time/Early", "Late"],
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 4. Permutation feature importance
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    feature_importance = pd.Series(
        importance.importances_mean,
        index=X_test.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().plot(kind="barh")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Mean Importance (AUPRC decrease)")
    plt.tight_layout()
    plt.show()

    print("Result visualizations completed successfully!")

"""
----------------------------------------------------------------------
Function Name : predict_orders
Description   : Predicts whether orders will be delivered late and
                calculates the probability of late delivery.
Input         : Trained machine learning model and order feature data.
Output        : Returns the input data with predicted classes and
                late-delivery probabilities.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def predict_orders(best_model, order_data):

    # Generate predictions
    predictions = best_model.predict(order_data)

    # Calculate probability of late delivery
    probabilities = best_model.predict_proba(order_data)[:, 1]

    # Create prediction results
    results = order_data.copy()

    results["Predicted_Late"] = predictions
    results["Late_Delivery_Probability"] = (
        probabilities * 100
    ).round(2)

    # Convert numeric predictions into readable labels
    results["Delivery_Prediction"] = np.where(
        results["Predicted_Late"] == 1,
        "Likely Late",
        "Likely On-Time/Early"
    )

    print("Order predictions completed successfully!")

    print("\nPrediction Summary:")
    print(
        results["Delivery_Prediction"]
        .value_counts()
        .to_string()
    )

    return results

"""
----------------------------------------------------------------------
Function Name : generate_insights
Description   : Generates business insights about sales, delivery
                performance, customer reviews, and customer segments.
Input         : Master DataFrame and RFM analysis DataFrame.
Output        : Returns a list of business insights with evidence
                and actionable recommendations.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def generate_insights(master, rfm):

    insights = []

    # 1. Late delivery analysis
    delivery_delay = master["delivery_delay_days"].dropna()

    if not delivery_delay.empty:
        late_percentage = delivery_delay.gt(0).mean() * 100

        insights.append({
            "Finding": "Late Delivery Performance",
            "Evidence": (
                f"{late_percentage:.2f}% of delivered orders were late."
            ),
            "Recommendation": (
                "Investigate delayed shipments and improve delivery "
                "planning for frequently delayed orders."
            )
        })

    # 2. Sales by product category
    if "product_categories" in master.columns:

        category_sales = (
            master[["product_categories", "total_price"]]
            .dropna(subset=["product_categories"])
            .assign(
                category=lambda df: df["product_categories"].str.split(", ")
            )
            .explode("category")
            .groupby("category")["total_price"]
            .sum()
            .sort_values(ascending=False)
        )

        if not category_sales.empty:
            top_category = category_sales.index[0]
            top_sales = category_sales.iloc[0]

            insights.append({
                "Finding": "Top Product Category",
                "Evidence": (
                    f"{top_category} has the highest attributed "
                    f"category sales: {top_sales:.2f}."
                ),
                "Recommendation": (
                    "Review inventory and marketing opportunities "
                    "for the leading product category."
                )
            })

    # 3. Customer review analysis
    if "review_score" in master.columns:
        review_data = master["review_score"].dropna()

        if not review_data.empty:
            average_review = review_data.mean()
            low_review_percentage = review_data.le(2).mean() * 100

            insights.append({
                "Finding": "Customer Satisfaction",
                "Evidence": (
                    f"Average review score: {average_review:.2f}/5; "
                    f"{low_review_percentage:.2f}% of scored orders "
                    "received ratings of 1 or 2."
                ),
                "Recommendation": (
                    "Investigate low-rated orders to identify "
                    "delivery and service issues."
                )
            })

    # 4. Customer segmentation
    if not rfm.empty and "Customer_Segment" in rfm.columns:
        segments = rfm["Customer_Segment"].value_counts()

        largest_segment = segments.index[0]
        largest_count = segments.iloc[0]
        largest_percentage = largest_count / len(rfm) * 100

        insights.append({
            "Finding": "Customer Segmentation",
            "Evidence": (
                f"{largest_segment} is the largest customer segment, "
                f"representing {largest_percentage:.2f}% of customers."
            ),
            "Recommendation": (
                "Design targeted campaigns based on customer "
                "segment behavior and purchasing patterns."
            )
        })

    # Display insights
    for number, insight in enumerate(insights, start=1):
        print(f"\nInsight {number}: {insight['Finding']}")
        print(f"Finding: {insight['Finding']}")
        print(f"Evidence: {insight['Evidence']}")
        print(f"Recommendation: {insight['Recommendation']}")

    print("\nBusiness insight generation completed!")

    return insights

"""
----------------------------------------------------------------------
Function Name : save_outputs
Description   : Saves state delivery performance, product category
                sales, and RFM customer segment summaries as CSV files.
Input         : Master DataFrame and RFM analysis DataFrame.
Output        : Saves three CSV summary files for Power BI analysis.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def save_outputs(master, rfm):

    # 1. State delivery performance
    state_delivery = (
        master.groupby("customer_state", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            average_delivery_days=("delivery_days", "mean"),
            late_delivery_percentage=("is_late_delivery", "mean")
        )
    )

    state_delivery["late_delivery_percentage"] *= 100

    state_delivery.to_csv(
        "state_delivery_performance.csv",
        index=False
    )

    # 2. Product category sales
    category_sales = (
        master[["product_categories", "total_price"]]
        .dropna(subset=["product_categories"])
        .assign(
            category=lambda df: df["product_categories"].str.split(", ")
        )
        .explode("category")
        .groupby("category", as_index=False)
        .agg(total_sales=("total_price", "sum"))
        .sort_values("total_sales", ascending=False)
    )

    category_sales.to_csv(
        "product_category_sales.csv",
        index=False
    )

    # 3. RFM customer segments
    rfm_segments = (
        rfm.groupby("Customer_Segment", as_index=False)
        .agg(customer_count=("customer_unique_id", "nunique"))
        .sort_values("customer_count", ascending=False)
    )

    rfm_segments.to_csv(
        "rfm_customer_segments.csv",
        index=False
    )

    print("Power BI summary files saved successfully!")
    print("1. state_delivery_performance.csv")
    print("2. product_category_sales.csv")
    print("3. rfm_customer_segments.csv")

"""
----------------------------------------------------------------------
Function Name : display_summary
Description   : Displays a summary of the Olist project, including
                dataset statistics, delivery performance, customer
                segments, and model evaluation results.
Input         : Master DataFrame, RFM DataFrame, and model results.
Output        : Displays a summary of the project analysis.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""
def display_summary(master, rfm, test_results):

    print("\n" + "=" * 60)
    print("       OLIST E-COMMERCE PROJECT SUMMARY")
    print("=" * 60)

    # Dataset summary
    print("\n1. DATASET SUMMARY")
    print("-" * 40)
    print("Total Orders:", master["order_id"].nunique())
    print("Total Customers:", master["customer_unique_id"].nunique())
    print("Total Product Categories:", master["product_categories"].nunique())

    # Delivery summary
    print("\n2. DELIVERY PERFORMANCE")
    print("-" * 40)

    delivery_data = master.dropna(
        subset=["delivery_delay_days"]
    )

    if not delivery_data.empty:
        late_percentage = (
            delivery_data["delivery_delay_days"].gt(0).mean() * 100
        )

        print(f"Late Delivery Percentage: {late_percentage:.2f}%")

    # Customer segmentation
    print("\n3. CUSTOMER SEGMENTS")
    print("-" * 40)

    print(
        rfm["Customer_Segment"]
        .value_counts()
        .to_string()
    )

    # Model performance
    print("\n4. MACHINE LEARNING RESULTS")
    print("-" * 40)

    print(test_results.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("             PROJECT SUMMARY COMPLETED")
    print("=" * 60)

"""
----------------------------------------------------------------------
Function Name : display_footer
Description   : Displays the final completion message for the
                Olist E-Commerce Data Science project.
Input         : None.
Output        : Displays the project completion footer.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def display_footer():

    print("\n" + "=" * 60)
    print("       OLIST E-COMMERCE DATA SCIENCE PROJECT")
    print("=" * 60)
    print("Data Cleaning       : Successful")
    print("Exploratory Analysis: Successful")
    print("RFM Analysis        : Successful")
    print("Machine Learning    : Successful")
    print("Business Insights   : Generated")
    print("Power BI Outputs    : Saved")
    print("-" * 60)
    print("       THANK YOU FOR USING THIS PROJECT!")
    print("=" * 60)

"""
----------------------------------------------------------------------
Function Name : rfm_analysis
Description   : Segments customers using Recency, Frequency, and Monetary.
Input         : master (merged order DataFrame)
Output        : Customer RFM analysis DataFrame
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def rfm_analysis(master):
    
    data = master.dropna(
        subset=["customer_unique_id", "order_purchase_timestamp"]
    ).copy()

    reference_date = data["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = data.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp",
                 lambda x: (reference_date - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("total_price", "sum")
    ).reset_index()

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        4,
        labels=[4, 3, 2, 1]
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    rfm["Customer_Segment"] = np.select(
        [
            (rfm["R_Score"] >= 3) & (rfm["F_Score"] >= 3),
            (rfm["F_Score"] >= 3),
            (rfm["R_Score"] >= 3),
        ],
        [
            "Loyal/Active",
            "Previously Frequent",
            "Recent/New",
        ],
        default="Needs Attention"
    )

    print("\nRFM Customer Segmentation Completed!")
    print(rfm["Customer_Segment"].value_counts())

    return rfm

"""
----------------------------------------------------------------------
Function Name : main
Description   : Executes the complete Olist E-Commerce Data Science
                project, including data loading, cleaning, EDA,
                RFM analysis, machine learning, evaluation,
                business insights, and output generation.
Input         : None
Output        : Executes all project functions in the correct order.
Author        : Atharv Tushar Bhosale
Date          : 21/10/2026
----------------------------------------------------------------------
"""

def main():

    print("\n" + "=" * 70)
    print("OLIST E-COMMERCE DATA SCIENCE PROJECT".center(70))
    print("Exploratory Data Analysis & Machine Learning".center(70))
    print("Late Delivery Prediction and Customer Segmentation".center(70))
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATASETS")
    print("=" * 70)

    (
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    ) = load_datasets()

    print("\n" + "=" * 70)
    print("STEP 2: UNDERSTAND DATASETS")
    print("=" * 70)

    understand_datasets(
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    )

    print("\n" + "=" * 70)
    print("STEP 3: CHECK MISSING VALUES")
    print("=" * 70)

    check_missing_values(
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    )

    print("\n" + "=" * 70)
    print("STEP 4: CHECK DUPLICATES")
    print("=" * 70)

    check_duplicates(
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    )

    print("\n" + "=" * 70)
    print("STEP 5: CLEAN DATASETS")
    print("=" * 70)

    (
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    ) = clean_datasets(
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    )

    print("\n" + "=" * 70)
    print("STEP 6: MERGE DATASETS")
    print("=" * 70)

    master = merge_datasets(
        customers,
        geolocation,
        items,
        payments,
        reviews,
        orders,
        products,
        sellers,
        category
    )

    print("\n" + "=" * 70)
    print("STEP 7: FEATURE ENGINEERING")
    print("=" * 70)

    master = feature_engineering(
        master,
        items,
        products
    )

    print("\n" + "=" * 70)
    print("STEP 8: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    perform_eda(master)

    print("\n" + "=" * 70)
    print("STEP 9: RFM CUSTOMER ANALYSIS")
    print("=" * 70)

    rfm = rfm_analysis(master)

    print("\n" + "=" * 70)
    print("STEP 10: CREATE TARGET VARIABLE")
    print("=" * 70)

    ml_data = create_target(master)

    print("\n" + "=" * 70)
    print("STEP 11: REMOVE DATA LEAKAGE")
    print("=" * 70)

    ml_data = remove_leakage(ml_data)

    print("\n" + "=" * 70)
    print("STEP 12: PREPARE MACHINE LEARNING DATA")
    print("=" * 70)

    print("\nTarget column check:")
    print(ml_data.columns.tolist())
    print("Target exists:", "is_late_delivery" in ml_data.columns)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_ml_data(ml_data)


    print("\n" + "=" * 70)
    print("STEP 13: BUILD MACHINE LEARNING MODELS")
    print("=" * 70)

    models = build_models(
        X_train,
        y_train
    )

    print("\n" + "=" * 70)
    print("STEP 14: COMPARE MACHINE LEARNING MODELS")
    print("=" * 70)

    model_results = compare_models(
        models,
        X_validation,
        y_validation
    )


    print("\n" + "=" * 70)
    print("STEP 15: SELECT THE BEST MODEL")
    print("=" * 70)

    best_model_name, best_model = select_best_model(
        models,
        model_results
    )

    print("\n" + "=" * 70)
    print("STEP 16A: TRAIN THE BEST MODEL")
    print("=" * 70)

    best_model = train_best_model(
        best_model,
        X_train,
        X_validation,
        y_train,
        y_validation
    )


    print("\n" + "=" * 70)
    print("STEP 16B: SAVE THE BEST MODEL")
    print("=" * 70)

    model_path = save_model(best_model)

    print("\n" + "=" * 70)
    print("STEP 16C: LOAD THE SAVED MODEL")
    print("=" * 70)

    loaded_model = load_model(model_path)

    print("\n" + "=" * 70)
    print("STEP 17: EVALUATE THE MODEL")
    print("=" * 70)

    test_results = evaluate_model(
        loaded_model,
        y_train,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("STEP 18: PLOT MODEL RESULTS")
    print("=" * 70)

    plot_results(
        test_results,
        loaded_model,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("STEP 19: PREDICT ORDERS")
    print("=" * 70)

    predictions = predict_orders(
        loaded_model,
        X_test.head(10)
    )

    print("\nSample Order Predictions:")
    print(predictions[
        [
            "Predicted_Late",
            "Late_Delivery_Probability",
            "Delivery_Prediction"
        ]
    ])

    print("\n" + "=" * 70)
    print("STEP 20: GENERATE BUSINESS INSIGHTS")
    print("=" * 70)

    insights = generate_insights(
        master,
        rfm
    )

    print("\n" + "=" * 70)
    print("STEP 21: SAVE OUTPUTS FOR POWER BI")
    print("=" * 70)

    save_outputs(
        master, 
        rfm
    )

    print("\n" + "=" * 70)
    print("STEP 22A: DISPLAY PROJECT SUMMARY")
    print("=" * 70)

    display_summary(
        master,
        rfm,
        test_results
    )

    print("\n" + "=" * 60)
    print("STEP 22B: DISPLAY PROJECT FOOTER")
    print("=" * 60)

    display_footer()

if __name__ == "__main__":
    main()
