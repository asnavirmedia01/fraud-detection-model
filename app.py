import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
from datetime import datetime

# -----------------------------
# 1. Load Model & Encoders
# -----------------------------
model = joblib.load("fraud_model.joblib")
le_device_os = joblib.load("label_encoder_Device_OS.joblib")
le_location = joblib.load("label_encoder_Location.joblib")

# -----------------------------
# 2. App Config
# -----------------------------
st.set_page_config(
    page_title="LIVE Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background image (optional)
page_bg_img = """
<style>
body {
background-image: url("https://images.unsplash.com/photo-1591696331117-b2e68c9f10ed");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
.stApp {
background-color: rgba(0,0,0,0.6);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------------------------
# 3. Session State Setup
# -----------------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "running" not in st.session_state:
    st.session_state.running = False

# -----------------------------
# 4. Control Panel
# -----------------------------
with st.sidebar:
    st.title("Control Panel")
    start = st.button("▶ Start Stream")
    stop = st.button("⏹ Stop Stream")
    reset = st.button("🔄 Reset App")
    speed = st.slider("Speed (sec per txn)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.write("Live session metrics below")

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False
if reset:
    st.session_state.transactions = []

# -----------------------------
# 5. Helper Functions
# -----------------------------
def generate_transaction():
    txn = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "amount": random.randint(100, 50000),
        "location": random.choice(["Lagos", "Abuja", "PH"]),
        "device_os": random.choice(["Android", "iOS"])
    }
    return txn

def preprocess(txn):
    # Encode categorical features
    txn_copy = txn.copy()
    txn_copy["device_os"] = le_device_os.transform([txn_copy["device_os"]])[0]
    txn_copy["location"] = le_location.transform([txn_copy["location"]])[0]
    return [txn_copy["amount"], txn_copy["device_os"], txn_copy["location"]]

def predict(txn):
    features = preprocess(txn)
    return model.predict([features])[0], model.predict_proba([features])[0][1]  # prediction, prob

def update_metrics(transactions):
    total = len(transactions)
    frauds = sum([t["prediction"] for t in transactions])
    safe = total - frauds
    avg_amount = np.mean([t["amount"] for t in transactions]) if total>0 else 0
    fraud_rate = (frauds/total*100) if total>0 else 0
    return total, frauds, safe, avg_amount, fraud_rate

# -----------------------------
# 6. Main Dashboard Layout
# -----------------------------
col1, col2, col3 = st.columns(3)
transactions = st.session_state.transactions

total_txn, fraud_count, safe_count, avg_amount, fraud_rate = update_metrics(transactions)

col1.metric("Total Transactions", total_txn)
col2.metric("Fraud Detected", fraud_count, delta=f"{fraud_rate:.2f}%")
col3.metric("Avg Transaction Amount", f"₦{avg_amount:,.0f}")

# -----------------------------
# 7. Live Charts & Feed
# -----------------------------
st.subheader("Transaction Stream & Analytics")

placeholder_line = st.empty()
placeholder_fraud = st.empty()
placeholder_feed = st.empty()

def run_stream():
    new_txn = generate_transaction()
    pred, prob = predict(new_txn)
    new_txn["prediction"] = pred
    new_txn["probability"] = round(prob,2)
    st.session_state.transactions.append(new_txn)

    # Limit history for charting
    MAX_POINTS = 100
    if len(st.session_state.transactions) > MAX_POINTS:
        st.session_state.transactions.pop(0)

    df = pd.DataFrame(st.session_state.transactions)

    # Line Chart: Transaction Amount
    placeholder_line.line_chart(df[["amount"]])

    # Fraud Timeline
    placeholder_fraud.bar_chart(df["prediction"].replace({0: np.nan, 1: 1}))

    # Live feed
    feed_text = ""
    for t in df.tail(10).itertuples():
        status = "⚠ FRAUD" if t.prediction==1 else "SAFE"
        color = "🔴" if t.prediction==1 else "🟢"
        feed_text += f"[{t.timestamp}] ₦{t.amount:,} – {t.device_os} – {t.location} → {color} {status}\n"
    placeholder_feed.text(feed_text)

# -----------------------------
# 8. Run Loop
# -----------------------------
if st.session_state.running:
    run_stream()
    time.sleep(speed)
    st.experimental_rerun()  # this is necessary for live updates

# -----------------------------
# 9. Export Data
# -----------------------------
st.subheader("Export Session Data")
if st.button("📥 Export to Excel"):
    df_export = pd.DataFrame(st.session_state.transactions)
    df_export.to_excel("live_fraud_session.xlsx", index=False)
    st.success("Data exported to live_fraud_session.xlsx")
