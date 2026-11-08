# Machine Learning-Based Network Intrusion Detection System

## Project Overview

This project implements a **Machine Learning-Based Network Intrusion
Detection System (NIDS)** using supervised and unsupervised machine
learning techniques.

The system analyzes network traffic data and identifies whether network
activities are normal or potentially malicious.

It combines supervised classification and anomaly detection techniques
to detect known attacks and identify suspicious or previously unseen
(zero-day) network activities.

## Key Features

-   Network traffic analysis using machine learning.
-   Data preprocessing and feature engineering.
-   Comparison of multiple machine learning algorithms.
-   Random Forest-based intrusion classification.
-   Isolation Forest-based anomaly detection.
-   Hybrid intrusion detection approach.
-   Zero-day and unknown attack detection.
-   Model evaluation and performance comparison.
-   SHAP-based model explainability.

## Machine Learning Models

  Model                 Purpose
  --------------------- -----------------------------------------------
  Logistic Regression   Baseline classification
  Decision Tree         Intrusion classification
  Random Forest         Supervised attack detection
  XGBoost               Advanced classification
  Isolation Forest      Anomaly detection
  Hybrid Model          Combines classification and anomaly detection

## Project Structure

``` text
Network_Intrusion_model/
│
├── app.py
├── requirements.txt
├── README.md
│
└── final_ids_models/
    ├── model_comparison.xlsx
    ├── feature_names.pkl
    ├── imputer.pkl
    ├── isolation_forest.pkl
    ├── metadata.json
    ├── scaler.pkl
    ├── isolation_forest_model.pkl
    ├── preprocessor.pkl
    ├── xgb_fraud_model.pkl
    └── final_ids_models.zip
```

## Model Files

Due to the large size of the trained machine learning models, the
following two files are provided in compressed ZIP format:

-   `random_forest_zero_day.pkl`
-   `best_ids_model.pkl`

**Important:** Extract both model files from `final_ids_models.zip` into
the `final_ids_models` folder before running the application.

Do not rename the extracted model files or change their directory
structure.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   SHAP
-   Streamlit

## How the System Works

1.  Upload network traffic data in CSV format.
2.  The system validates and preprocesses the uploaded data.
3.  Extracted features are passed to the trained machine learning
    models.
4.  Random Forest predicts known network traffic classes.
5.  Isolation Forest identifies unusual network behaviour.
6.  The hybrid detection approach combines model predictions.
7.  The system displays the detection results.

## Project Objective

The main objective of this project is to develop an intelligent
intrusion detection system that can analyze network traffic, detect
malicious activities, and identify unusual network behaviour using
machine learning.

## Author

**Abhilash**
