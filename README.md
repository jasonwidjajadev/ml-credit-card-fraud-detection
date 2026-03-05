# Credit Card Fraud Detection & Transaction Prediction

## Overview

This project implements an **end‑to‑end machine learning pipeline** for
analysing credit card transactions.\
The system performs two tasks:

1.  **Regression:** Predict the transaction amount (`amt`)
2.  **Classification:** Detect fraudulent transactions (`is_fraud`)

The project demonstrates a full machine learning workflow including:

-   Data preprocessing
-   Feature engineering
-   Model training
-   Evaluation and prediction generation

The implementation is written in **Python** using **Pandas, NumPy,
Scikit‑learn, and LightGBM**.

------------------------------------------------------------------------

# Dataset

The Dataset is a custom dataset, similar ones include:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset contains simulated credit card transactions including:

  Feature                     | Description
  --------------------------- | -----------------------------------------
  `trans_date_trans_time`     | Timestamp of the transaction
  `merchant`                  | Merchant name
  `category`                  | Transaction category
  `amt`                       | Transaction amount
  `city`, `state`, `zip`      | Customer location
  `lat`, `long`               | Customer coordinates
  `merch_lat`, `merch_long`   | Merchant coordinates
  `job`                       | Cardholder occupation
  `dob`                       | Date of birth
  `city_pop`                  | City population
  `is_fraud`                  | Fraud label (0 = legitimate, 1 = fraud)


------------------------------------------------------------------------

# Machine Learning Pipeline

## 1. Data Preprocessing

Key preprocessing steps include:

-   Parsing timestamps into hour, month, and day features
-   Extracting age from date of birth
-   Encoding categorical variables
-   Handling missing values
-   Creating geographic distance features between customer and merchant

These transformations help convert raw transaction data into useful
model features.

------------------------------------------------------------------------

## 2. Feature Engineering

Important engineered features include:

-   **Temporal features**
    -   transaction hour
    -   weekend indicator
    -   cyclic hour encoding (sin/cos)
-   **Location features**
    -   customer--merchant distance
    -   city population scaling
-   **Behavioural statistics**
    -   category transaction averages
    -   merchant transaction averages
    -   job-based spending averages
-   **Outlier detection features**
    -   z-score of transaction amounts
    -   category-specific amount deviations

These features help capture behavioural patterns associated with fraud.

------------------------------------------------------------------------

# Models

Two LightGBM models are trained:

### Regression Model

Predicts the **transaction amount (`amt`)**.

Model: - `LightGBM Regressor` - Optimised for RMSE performance

Output file:

    z{studentID}_regression.csv

Example:

  trans_num   | amt
  ----------- | ---------
  12345       | 39.10
  12346       | 0.87
  12347       | 1000.00

------------------------------------------------------------------------

### Classification Model

Detects **fraudulent transactions (`is_fraud`)**.

Model: - `LightGBM Classifier` - Optimised for **Macro F1 Score**

Output file:

    z{studentID}_classification.csv

Example:

  trans_num   | is_fraud
  ----------- | ----------
  12345       | 0
  12346       | 1
  12347       | 0

------------------------------------------------------------------------

# How to Run

## Install dependencies

    pip install -r requirements.txt

## Run the training script

    python3 z5494973.py <train_csv> <test_csv>

Example:

    python3 z5494973.py train.csv test.csv

The script will generate:

    z5494973_regression.csv
    z5494973_classification.csv

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit‑learn
-   LightGBM
-   Jupyter Notebook

------------------------------------------------------------------------

# Learning Outcomes

This project demonstrates:

-   Building **end‑to‑end ML pipelines**
-   **Feature engineering on tabular data**
-   Handling **regression and classification tasks simultaneously**
-   Training **tree‑based ensemble models**
-   Designing **reproducible ML experiments**
