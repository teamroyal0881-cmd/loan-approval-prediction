import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details below to estimate the model's loan approval classification.")

model_path = Path("loan_model.pkl")
if not model_path.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(model_path)

dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=10000000, step=100000)
loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=40, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)

residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=3000000, step=100000)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=1000000, step=100000)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=5000000, step=100000)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=2000000, step=100000)

if st.button("Predict Loan Approval", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }])

    prediction = str(model.predict(input_data)[0]).strip()

    if prediction.lower() == "approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = [str(c).strip() for c in model.classes_]
        if "Approved" in classes:
            approved_probability = probabilities[classes.index("Approved")]
            st.write(f"Model confidence for approval: **{approved_probability:.1%}**")

st.caption("Educational ML project only. This is not a real lending decision system.")
