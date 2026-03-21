import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import base64

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# ----------------------------
# BACKGROUND IMAGE FUNCTION
# ----------------------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("background.jpg")

page_bg = f"""
<style>
.stApp {{
background-image: url("data:image/jpg;base64,{img}");
background-size: cover;
background-position: center;
background-attachment: fixed;
}}

.block-container {{
background: rgba(0,0,0,0.65);
padding: 2rem;
border-radius: 15px;
color: white;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS
# ----------------------------
rf_model = joblib.load("random_forest_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

le_device_os = joblib.load("label_encoder_Device_OS.joblib")
le_marital_status = joblib.load("label_encoder_Customer_Marital_Status.joblib")

USER_INCOME_ESTIMATE_MEAN = 276027.06757072714

FEATURE_COLUMNS = [
    'Transaction_Amount','Transaction_Hour','Location_Change','Device_Mismatch',
    'Transaction_Velocity','Account_Age_Days','AvgTransaction_Last30Days',
    'Amount_To_Average_Ratio','User_Income_Estimate','Device_OS',
    'Customer_Marital_Status','High_Risk_Hour'
]

# ----------------------------
# SESSION STATE
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "transaction_amount":100000.0,
        "avg_transaction":50000.0,
        "user_income":USER_INCOME_ESTIMATE_MEAN
    }

# ----------------------------
# PREPROCESSING
# ----------------------------
def preprocess_input(*args):
    df = pd.DataFrame([args], columns=[
        'Transaction_Amount','Transaction_Hour','Location_Change','Device_Mismatch',
        'Transaction_Velocity','Account_Age_Days','AvgTransaction_Last30Days',
        'Amount_To_Average_Ratio','User_Income_Estimate_Raw',
        'Device_OS_Raw','Customer_Marital_Status_Raw'
    ])

    df['User_Income_Estimate'] = df['User_Income_Estimate_Raw'].fillna(USER_INCOME_ESTIMATE_MEAN)
    df['High_Risk_Hour'] = df['Transaction_Hour'].apply(lambda x: 1 if x < 6 else 0)

    df['Device_OS'] = le_device_os.transform(df['Device_OS_Raw'])
    df['Customer_Marital_Status'] = le_marital_status.transform(df['Customer_Marital_Status_Raw'])

    processed = df[FEATURE_COLUMNS]
    return scaler.transform(processed)

# ----------------------------
# EXPORT FUNCTION
# ----------------------------
def export_excel(dataframe):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False)
    return buffer.getvalue()

# ----------------------------
# UI
# ----------------------------
st.title("🛡️ AI Fraud Detection System")
st.caption("Smart Detection for Nigerian Digital Payments")

DEVICE_OS_OPTIONS = le_device_os.classes_.tolist()
MARITAL_STATUS_OPTIONS = le_marital_status.classes_.tolist()

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input("Transaction Amount (₦)", value=st.session_state.inputs["transaction_amount"])
    st.write(f"₦ {transaction_amount:,.0f}")

    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    transaction_velocity = st.number_input("Transaction Velocity", value=5)
    account_age_days = st.number_input("Account Age (Days)", value=1000)

with col2:
    location_change = st.selectbox("Location Change", [0,1], format_func=lambda x:"Yes" if x else "No")
    device_mismatch = st.selectbox("Device Mismatch", [0,1], format_func=lambda x:"Yes" if x else "No")

    avg_transaction_last30days = st.number_input("Avg Transaction (₦)", value=st.session_state.inputs["avg_transaction"])
    st.write(f"₦ {avg_transaction_last30days:,.0f}")

    amount_to_average_ratio = st.number_input("Amount to Avg Ratio", value=1.0)

st.subheader("User Info")

col3, col4 = st.columns(2)

with col3:
    user_income_estimate = st.number_input("User Income (₦)", value=st.session_state.inputs["user_income"])
    st.write(f"₦ {user_income_estimate:,.0f}")

    device_os = st.selectbox("Device OS", DEVICE_OS_OPTIONS)

with col4:
    customer_marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)

st.divider()

# ----------------------------
# BUTTONS
# ----------------------------
colA, colB = st.columns(2)

with colA:
    predict_btn = st.button("Predict Fraud")

with colB:
    reset_btn = st.button("Reset All Inputs")

# ----------------------------
# RESET LOGIC
# ----------------------------
if reset_btn:
    st.session_state.inputs = {
        "transaction_amount":100000.0,
        "avg_transaction":50000.0,
        "user_income":USER_INCOME_ESTIMATE_MEAN
    }
    st.session_state.history = []
    st.rerun()

# ----------------------------
# PREDICTION
# ----------------------------
if predict_btn:

    processed = preprocess_input(
        transaction_amount, transaction_hour, location_change, device_mismatch,
        transaction_velocity, account_age_days, avg_transaction_last30days,
        amount_to_average_ratio, user_income_estimate, device_os, customer_marital_status
    )

    prediction = rf_model.predict(processed)[0]
    probability = rf_model.predict_proba(processed)[:,1][0]

    if prediction == 1:
        st.error(f"⚠ Fraud Risk ({probability:.2f})")
    else:
        st.success(f"✓ Legitimate ({probability:.2f})")

    record = {
        "Amount": f"{transaction_amount:,.0f}",
        "Avg": f"{avg_transaction_last30days:,.0f}",
        "Income": f"{user_income_estimate:,.0f}",
        "Probability": probability,
        "Result": "Fraud" if prediction else "Safe"
    }

    st.session_state.history.append(record)

# ----------------------------
# HISTORY + EXPORT
# ----------------------------
st.subheader("History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Excel",
        data=export_excel(df),
        file_name="fraud_results.xlsx"
    )
else:
    st.info("No predictions yet.")
