pip install joblib
import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model and scaler
model = joblib.load("random_forest_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("AI Fraud Detection System")
st.write("Predict the risk of fraudulent digital payment transactions")

st.sidebar.header("Enter Transaction Details")

# User Inputs
Transaction_Amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
Transaction_Hour = st.sidebar.slider("Transaction Hour (0-23)", 0, 23)
Transaction_Velocity = st.sidebar.number_input("Transactions in Last Hour", min_value=0)
Account_Age_Days = st.sidebar.number_input("Account Age (Days)", min_value=0)
Device_Mismatch = st.sidebar.selectbox("Device Mismatch (0=No, 1=Yes)", [0,1])
Amount_To_Average_Ratio = st.sidebar.number_input("Amount to Average Ratio", min_value=0.0)
Location_Change = st.sidebar.selectbox("Location Change (0=No, 1=Yes)", [0,1])

# Feature Engineering: compute High_Risk_Hour
High_Risk_Hour = 1 if Transaction_Hour < 6 else 0

# Create DataFrame for the model
input_data = pd.DataFrame({
    'Transaction_Amount':[Transaction_Amount],
    'Transaction_Hour':[Transaction_Hour],
    'Location_Change':[Location_Change],
    'Device_Mismatch':[Device_Mismatch],
    'Transaction_Velocity':[Transaction_Velocity],
    'Account_Age_Days':[Account_Age_Days],
    'Amount_To_Average_Ratio':[Amount_To_Average_Ratio],
    'High_Risk_Hour':[High_Risk_Hour]
})

# Scale features
scaled_data = scaler.transform(input_data)

# Predict button
if st.button("Predict Fraud Risk"):
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Fraud Risk Transaction")
    else:
        st.success("✅ Transaction Appears Legitimate")
    
    st.write("Fraud Risk Score:", round(probability,3))
