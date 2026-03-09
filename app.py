import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Assets ---
# Load the trained Random Forest model
rf_model = joblib.load('random_forest_fraud_model.joblib')

# Load the StandardScaler
scaler = joblib.load('scaler.joblib')

# Load individual LabelEncoders
le_device_os = joblib.load('label_encoder_Device_OS.joblib')
le_marital_status = joblib.load('label_encoder_Customer_Marital_Status.joblib')

# Get the mean for User_Income_Estimate imputation (from previous step's output)
USER_INCOME_ESTIMATE_MEAN = 276027.06757072714

# Define the feature names in the order they were trained
# (This can be obtained from X.columns used during training)
FEATURE_COLUMNS = [
    'Transaction_Amount', 'Transaction_Hour', 'Location_Change',
    'Device_Mismatch', 'Transaction_Velocity', 'Account_Age_Days',
    'AvgTransaction_Last30Days', 'Amount_To_Average_Ratio',
    'User_Income_Estimate', 'Device_OS', 'Customer_Marital_Status',
    'High_Risk_Hour'
]

# --- 2. Preprocessing Function ---
def preprocess_input(
    transaction_amount, transaction_hour, location_change, device_mismatch,
    transaction_velocity, account_age_days, avg_transaction_last30days,
    amount_to_average_ratio, user_income_estimate, device_os, customer_marital_status
):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([[
        transaction_amount, transaction_hour, location_change, device_mismatch,
        transaction_velocity, account_age_days, avg_transaction_last30days,
        amount_to_average_ratio, user_income_estimate, device_os, customer_marital_status
    ]], columns=[
        'Transaction_Amount', 'Transaction_Hour', 'Location_Change', 'Device_Mismatch',
        'Transaction_Velocity', 'Account_Age_Days', 'AvgTransaction_Last30Days',
        'Amount_To_Average_Ratio', 'User_Income_Estimate_Raw', 'Device_OS_Raw', 'Customer_Marital_Status_Raw'
    ])

    # Impute missing User_Income_Estimate
    if pd.isna(input_data['User_Income_Estimate_Raw'].iloc[0]):
        input_data['User_Income_Estimate'] = USER_INCOME_ESTIMATE_MEAN
    else:
        input_data['User_Income_Estimate'] = input_data['User_Income_Estimate_Raw']

    # Create High_Risk_Hour feature
    input_data['High_Risk_Hour'] = input_data['Transaction_Hour'].apply(lambda x: 1 if x < 6 else 0)

    # Encode categorical features
    input_data['Device_OS'] = le_device_os.transform(input_data['Device_OS_Raw'])
    input_data['Customer_Marital_Status'] = le_marital_status.transform(input_data['Customer_Marital_Status_Raw'])

    # Select and order features as per training data
    processed_df = input_data[FEATURE_COLUMNS]

    # Scale numerical features
    scaled_input = scaler.transform(processed_df)

    return scaled_input

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title='Fraud Detection App', layout='centered')

st.title('Fraud Detection Application')
st.write("Enter the transaction details below to predict if it's fraudulent or legitimate.")
st.markdown("--- ")

# List of original categorical values for select boxes
DEVICE_OS_OPTIONS = le_device_os.classes_.tolist() # e.g., ['Android', 'iOS', 'Windows']
MARITAL_STATUS_OPTIONS = le_marital_status.classes_.tolist() # e.g., ['Divorced', 'Married', 'Single', 'Widowed']

# Input fields for features
st.subheader('Transaction Details')
col1, col2 = st.columns(2)
with col1:
    transaction_amount = st.number_input('Transaction Amount', min_value=0.0, value=100000.0, step=1000.0)
    transaction_hour = st.slider('Transaction Hour (0-23)', min_value=0, max_value=23, value=12)
    transaction_velocity = st.number_input('Transaction Velocity (Transactions in last hour)', min_value=0, value=5)
    account_age_days = st.number_input('Account Age (Days)', min_value=0, value=1000)

with col2:
    location_change = st.selectbox('Location Change', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    device_mismatch = st.selectbox('Device Mismatch', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    avg_transaction_last30days = st.number_input('Average Transaction in Last 30 Days', min_value=0.0, value=50000.0, step=500.0)
    amount_to_average_ratio = st.number_input('Amount to Average Ratio', min_value=0.0, value=1.0, step=0.1, format="%.2f")

st.subheader('User Information')
col3, col4 = st.columns(2)
with col3:
    user_income_estimate = st.number_input('User Income Estimate (Optional)', min_value=0.0, value=None, placeholder="Leave blank for average")
    device_os = st.selectbox('Device OS', options=DEVICE_OS_OPTIONS)

with col4:
    customer_marital_status = st.selectbox('Customer Marital Status', options=MARITAL_STATUS_OPTIONS)

st.markdown("--- ")
st.subheader('Important Features at a Glance (From Model Analysis)')
st.write("These features were identified as most influential in predicting fraud:")
st.markdown("**1. Account Age (Days)**: Older accounts tend to be less fraudulent.")
st.markdown("**2. Transaction Velocity**: High velocity often indicates fraud.")
st.markdown("**3. Transaction Amount**: Very high or very low amounts can be suspicious.")
st.markdown("**4. Transaction Hour**: Transactions during certain hours (e.g., late night) can be riskier.")
st.markdown("**5. High Risk Hour (System Derived)**: Transactions between 00:00 and 05:59.")
st.markdown("--- ")

# Prediction Button
if st.button('Predict Fraud'):
    # Preprocess the user inputs
    processed_data = preprocess_input(
        transaction_amount, transaction_hour, location_change, device_mismatch,
        transaction_velocity, account_age_days, avg_transaction_last30days,
        amount_to_average_ratio, user_income_estimate, device_os, customer_marital_status
    )

    # Make prediction
    prediction = rf_model.predict(processed_data)
    prediction_proba = rf_model.predict_proba(processed_data)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"\u274C Fraudulent Transaction Detected! (Probability: {prediction_proba[0]:.2f})")
        st.balloons()
    else:
        st.success(f"\u2705 Legitimate Transaction. (Probability: {prediction_proba[0]:.2f})")


st.caption("Note: This is a predictive model. Always exercise caution with suspicious transactions.")
