import streamlit as st
import pandas as pd
import joblib

# Load trained RandomForest model (updated file)
model = joblib.load("fraud_detection_rf.pkl")  # <-- updated file

# Best threshold (adjust if you have a new tuned threshold for this model)
DEFAULT_THRESHOLD = 0.27  # keep as is or change if needed

# --- Title ---
st.markdown(
    """
    <h1 style='margin-bottom:0.1em;'>💳 Fraud Detection App</h1>
    <p style='margin-top:0; margin-bottom:0.5em; color: #ccc;' >
        Enter the transaction details and adjust the threshold to check if it may be fraudulent.
    </p>
    """,
    unsafe_allow_html=True,
)

# --- Try Examples ---
st.markdown("### 🔎 Try Example Transactions", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

if col1.button("⚠️ Load Fraud Example", use_container_width=True):
    st.session_state.transaction_type = "TRANSFER"
    st.session_state.amount = 2000000.0
    st.session_state.oldbalanceOrg = 2000000.0
    st.session_state.newbalanceOrig = 0.0
    st.session_state.oldbalanceDest = 0.0
    st.session_state.newbalanceDest = 0.0

if col2.button("✅ Load Legit Example", use_container_width=True):
    st.session_state.transaction_type = "PAYMENT"
    st.session_state.amount = 500.0
    st.session_state.oldbalanceOrg = 500.0
    st.session_state.newbalanceOrig = 0.0
    st.session_state.oldbalanceDest = 0.0
    st.session_state.newbalanceDest = 500.0

# --- Inputs ---
st.markdown("<hr style='margin:0.4em 0;'>", unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
with colA:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
        key="transaction_type",
    )
with colB:
    amount = st.number_input("Amount", min_value=0.0, step=100.0, key="amount")
with colC:
    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)", min_value=0.0, step=100.0, key="oldbalanceOrg"
    )

colD, colE, colF = st.columns(3)
with colD:
    newbalanceOrig = st.number_input(
        "New Balance (Sender)", min_value=0.0, step=100.0, key="newbalanceOrig"
    )
with colE:
    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)", min_value=0.0, step=100.0, key="oldbalanceDest"
    )
with colF:
    newbalanceDest = st.number_input(
        "New Balance (Receiver)", min_value=0.0, step=100.0, key="newbalanceDest"
    )

# Threshold slider
threshold = st.slider(
    "Fraud Probability Threshold",
    0.0,
    1.0,
    DEFAULT_THRESHOLD,  # uses current threshold
    0.01,
    key="threshold",
    label_visibility="visible",
)

# --- Predict Button ---
if st.button("🔍 Predict", use_container_width=True):
    input_data = pd.DataFrame(
        [
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "balanceDiffOrig": newbalanceOrig - oldbalanceOrg,
                "balanceDiffDest": newbalanceDest - oldbalanceDest,
            }
        ]
    )

    prob = model.predict_proba(input_data)[0, 1]
    prediction = int(prob >= threshold)

    colR1, colR2 = st.columns(2)
    with colR1:
        st.write(f"**Fraud Probability:** {prob:.2%}")
    with colR2:
        st.write(f"**Threshold:** {threshold:.2f}")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")
