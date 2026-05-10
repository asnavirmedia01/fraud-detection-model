<img width="1365" height="656" alt="Screenshot 2026-05-06 043537" src="https://github.com/user-attachments/assets/6ceef77b-37b9-4c66-9bfd-4f16cd87ae18" />

# Fraud Detection Project

## Project Overview
This project focuses on building and deploying a machine learning model to detect fraudulent transactions using a synthetic dataset. The goal is to identify suspicious activities accurately and efficiently, leveraging various transaction and user-related features.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Models Used](#4-models-used)
5. [Performance](#5-performance)
6. [Feature Importance](#6-feature-importance)
7. [Streamlit Application](#7-streamlit-application)
8. [Installation and Usage](#8-installation-and-usage)
9. [Technologies Used](#9-technologies-used)
10. [Conclusion](#10-conclusion)

## 1. Introduction
Financial fraud is a significant concern for institutions and individuals. This project aims to develop a robust fraud detection system capable of distinguishing between legitimate and fraudulent transactions. By analyzing patterns in historical transaction data, the model can flag potentially fraudulent activities in real-time or near real-time.

## 2. Dataset
The project utilizes a synthetic fraud dataset containing various features related to financial transactions. The dataset includes information such as:
-   `Transaction_ID` (Dropped due to being an identifier)
-   `Transaction_Amount`
-   `Transaction_Hour`
-   `Location_Change`
-   `Device_Mismatch`
-   `Transaction_Velocity`
-   `Account_Age_Days`
-   `AvgTransaction_Last30Days`
-   `Amount_To_Average_Ratio`
-   `User_Income_Estimate`
-   `Device_OS`
-   `Customer_Marital_Status`
-   `Last_Manual_Review_Note` (Dropped due to high missing values)
-   `RESULT` (Target variable: 0 for Legit, 1 for Fraudulent)

The dataset initially had missing values and class imbalance, which were addressed during preprocessing.

## 3. Methodology
The project followed these steps:
-   **Data Loading and Initial Exploration**: Loaded the dataset and performed initial checks for structure and missing values.
-   **Data Preprocessing**: 
    -   Dropped `Transaction_ID` and `Last_Manual_Review_Note` due to irrelevance or high missingness.
    -   Imputed missing categorical values (`Device_OS`, `Customer_Marital_Status`) with their respective modes.
    -   Imputed missing `User_Income_Estimate` with the mean.
    -   Created a new feature `High_Risk_Hour` based on `Transaction_Hour`.
    -   Encoded categorical features using `LabelEncoder`.
    -   Applied `StandardScaler` to numerical features.
-   **Handling Class Imbalance**: Used undersampling on the majority class (legitimate transactions) to balance the training data, ensuring the model doesn't overfit to the non-fraudulent class.
-   **Model Training**: Trained several classification models, including Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.
-   **Model Evaluation**: Evaluated models using metrics like Confusion Matrix, Classification Report, Accuracy, and ROC AUC score.
-   **Feature Importance Analysis**: Identified the most influential features for fraud detection.

## 4. Models Used
-   **Logistic Regression**: A linear model for binary classification.
-   **Random Forest Classifier**: An ensemble learning method using multiple decision trees.
-   **Gradient Boosting Classifier**: Another ensemble method that builds trees sequentially, each trying to correct the errors of the previous one.

All models achieved high performance on the test set, with the Random Forest Classifier being selected as the primary model due to its robustness and interpretability.

## 5. Performance
The models achieved exceptional performance on the balanced dataset after preprocessing and undersampling. For instance, the Random Forest Classifier showed:

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1517
           1       1.00      1.00      1.00       483

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

ROC AUC: 1.0
```

This near-perfect score indicates excellent separation between fraudulent and legitimate transactions.

## 6. Feature Importance
Analysis of feature importance from the Random Forest model revealed the following top features:

```
                      Feature  Importance
5            Account_Age_Days    0.418954
4        Transaction_Velocity    0.226672
0          Transaction_Amount    0.147206
1            Transaction_Hour    0.111950
11           High_Risk_Hour    0.044419
3             Device_Mismatch    0.023639
7     Amount_To_Average_Ratio    0.016758
2             Location_Change    0.008431
```

`Account_Age_Days`, `Transaction_Velocity`, and `Transaction_Amount` were found to be the most significant indicators of fraudulent activity.

## 7. Streamlit Application
The trained Random Forest model has been deployed as a web application using Streamlit. This interactive application allows users to input transaction details and receive an instant prediction of whether the transaction is fraudulent or legitimate.

**Access the Streamlit App here:** [Paste Streamlit App Link Here]

## 8. Installation and Usage
To run this project locally:

1.  **Clone the repository:**
    ```bash
    https://github.com/asnavirmedia01/fraud-detection-model.git

    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file would typically contain `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`)

4.  **Run the Streamlit application:**
    ```bash
    https://fraud-detection-model-eqvzth7d3gymhmyvpdpah6.streamlit.app/
    ```

## 9. Technologies Used
-   Python
-   Pandas (for data manipulation)
-   NumPy (for numerical operations)
-   Scikit-learn (for machine learning models and preprocessing)
-   Matplotlib & Seaborn (for data visualization)
-   Streamlit (for web application deployment)
-   Joblib (for model serialization)

## 10. Conclusion
This project successfully developed and deployed a robust fraud detection system. The high performance metrics demonstrate the model's effectiveness in identifying fraudulent transactions, and the Streamlit application provides an intuitive interface for practical use. Future work could involve exploring more advanced models, incorporating real-time data streams, and enhancing the explainability of predictions.
