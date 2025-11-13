import streamlit as st
import pickle
import numpy as np

# =============================
# 1️⃣ Load Model
# =============================
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, le = pickle.load(f)

# =============================
# 2️⃣ App Title & Info
# =============================
st.set_page_config(page_title="🏦 Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill the details below to check if your loan is likely to be approved.")

# =============================
# 3️⃣ User Inputs
# =============================
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Co-applicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
        loan_term = st.number_input("Loan Amount Term (in days)", min_value=0, step=12)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("🔮 Predict Loan Approval")

# =============================
# 4️⃣ Manual Encoding
# =============================
if submitted:
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    education_map = {"Graduate": 1, "Not Graduate": 0}
    self_emp_map = {"Yes": 1, "No": 0}
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    input_data = [
        gender_map[gender],
        married_map[married],
        dependents_map[dependents],
        education_map[education],
        self_emp_map[self_employed],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_map[property_area],
    ]

    # =============================
    # 5️⃣ Preprocess & Predict
    # =============================
    X_input = np.array(input_data).reshape(1, -1)
    X_input_scaled = scaler.transform(X_input)

    prediction = model.predict(X_input_scaled)[0]

    # =============================
    # 6️⃣ Display Result
    # =============================
    st.markdown("---")
    if prediction == 1:
        st.success("✅ Loan Approved! Congratulations 🎉")
    else:
        st.error("❌ Loan Not Approved. Try improving your profile or credit history.")
