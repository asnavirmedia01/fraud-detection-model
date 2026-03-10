import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# ----------------------------
# Load Models and Encoders
# ----------------------------
rf_model = joblib.load("random_forest_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

le_device_os = joblib.load("label_encoder_Device_OS.joblib")
le_marital_status = joblib.load("label_encoder_Customer_Marital_Status.joblib")

USER_INCOME_ESTIMATE_MEAN = 276027.06757072714

FEATURE_COLUMNS = [
    'Transaction_Amount',
    'Transaction_Hour',
    'Location_Change',
    'Device_Mismatch',
    'Transaction_Velocity',
    'Account_Age_Days',
    'AvgTransaction_Last30Days',
    'Amount_To_Average_Ratio',
    'User_Income_Estimate',
    'Device_OS',
    'Customer_Marital_Status',
    'High_Risk_Hour'
]

# ----------------------------
# Session Storage
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_input(
    transaction_amount, transaction_hour, location_change, device_mismatch,
    transaction_velocity, account_age_days, avg_transaction_last30days,
    amount_to_average_ratio, user_income_estimate, device_os, customer_marital_status
):

    df = pd.DataFrame([[
        transaction_amount,
        transaction_hour,
        location_change,
        device_mismatch,
        transaction_velocity,
        account_age_days,
        avg_transaction_last30days,
        amount_to_average_ratio,
        user_income_estimate,
        device_os,
        customer_marital_status
    ]], columns=[
        'Transaction_Amount',
        'Transaction_Hour',
        'Location_Change',
        'Device_Mismatch',
        'Transaction_Velocity',
        'Account_Age_Days',
        'AvgTransaction_Last30Days',
        'Amount_To_Average_Ratio',
        'User_Income_Estimate_Raw',
        'Device_OS_Raw',
        'Customer_Marital_Status_Raw'
    ])

    if pd.isna(df['User_Income_Estimate_Raw'].iloc[0]):
        df['User_Income_Estimate'] = USER_INCOME_ESTIMATE_MEAN
    else:
        df['User_Income_Estimate'] = df['User_Income_Estimate_Raw']

    df['High_Risk_Hour'] = df['Transaction_Hour'].apply(lambda x: 1 if x < 6 else 0)

    df['Device_OS'] = le_device_os.transform(df['Device_OS_Raw'])
    df['Customer_Marital_Status'] = le_marital_status.transform(df['Customer_Marital_Status_Raw'])

    processed = df[FEATURE_COLUMNS]

    scaled = scaler.transform(processed)

    return scaled

# ----------------------------
# Excel Export
# ----------------------------
def export_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Fraud_Predictions")
    return output.getvalue()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Fraud Detection System")
st.caption("Machine Learning Model for Detecting Fraudulent Transactions")

DEVICE_OS_OPTIONS = le_device_os.classes_.tolist()
MARITAL_STATUS_OPTIONS = le_marital_status.classes_.tolist()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    # Transaction Amount with comma preview
    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )
    st.write("Entered Amount: ₦", f"{transaction_amount:,.0f}")

    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)

    transaction_velocity = st.number_input(
        "Transaction Velocity (Last Hour)",
        min_value=0,
        value=5
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=1000
    )

with col2:
    location_change = st.selectbox(
        "Location Change",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    device_mismatch = st.selectbox(
        "Device Mismatch",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    # Average Transaction Last 30 Days with comma preview
    avg_transaction_last30days = st.number_input(
        "Average Transaction (Last 30 Days)",
        min_value=0.0,
        value=50000.0,
        step=500.0
    )
    st.write("Formatted Average: ₦", f"{avg_transaction_last30days:,.0f}")

    amount_to_average_ratio = st.number_input(
        "Amount to Average Ratio",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

st.subheader("User Information")

col3, col4 = st.columns(2)

with col3:
    # User Income Estimate with comma preview
    user_income_estimate = st.number_input(
        "User Income Estimate",
        min_value=0.0,
        value=USER_INCOME_ESTIMATE_MEAN
    )
    st.write("Formatted Income: ₦", f"{user_income_estimate:,.0f}")

    device_os = st.selectbox(
        "Device OS",
        DEVICE_OS_OPTIONS
    )

with col4:
    customer_marital_status = st.selectbox(
        "Customer Marital Status",
        MARITAL_STATUS_OPTIONS
    )

st.divider()

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Fraud Risk"):

    processed = preprocess_input(
        transaction_amount,
        transaction_hour,
        location_change,
        device_mismatch,
        transaction_velocity,
        account_age_days,
        avg_transaction_last30days,
        amount_to_average_ratio,
        user_income_estimate,
        device_os,
        customer_marital_status
    )

    prediction = rf_model.predict(processed)[0]
    probability = rf_model.predict_proba(processed)[:, 1][0]

    if prediction == 1:
        st.error(f"⚠ Fraudulent Transaction (Risk: {probability:.2f})")
    else:
        st.success(f"✓ Legitimate Transaction (Risk: {probability:.2f})")

    # Store formatted numbers for history
    formatted_amount = f"{transaction_amount:,.0f}"
    formatted_avg = f"{avg_transaction_last30days:,.0f}"
    formatted_income = f"{user_income_estimate:,.0f}"

    record = {
        "Transaction Amount": formatted_amount,
        "Hour": transaction_hour,
        "Velocity": transaction_velocity,
        "Account Age": account_age_days,
        "Average Last30Days": formatted_avg,
        "Amount to Avg Ratio": amount_to_average_ratio,
        "Device OS": device_os,
        "Marital Status": customer_marital_status,
        "User Income Estimate": formatted_income,
        "Fraud Probability": probability,
        "Prediction": "Fraud" if prediction else "Legitimate"
    }

    st.session_state.history.append(record)

# ----------------------------
# Prediction History
# ----------------------------
st.subheader("Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    excel = export_excel(history_df)

    st.download_button(
        "Download Predictions (Excel)",
        data=excel,
        file_name="fraud_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("No predictions recorded yet.")
