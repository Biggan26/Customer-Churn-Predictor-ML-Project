import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ১. পেজ কনফিগারেশন
st.set_page_config(
    page_title="Telco Churn Analytics (Full Profile)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ২. কাস্টম সিএসএস (UI স্টাইলিং)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4A90E2; color: white; border-radius: 8px;
        padding: 10px 24px; font-weight: bold; width: 100%; border: none;
    }
    .stButton>button:hover { background-color: #357ABD; color: white; }
    .prediction-box-stay {
        padding: 20px; background-color: #d4edda; color: #155724;
        border-radius: 10px; border-left: 6px solid #28a745; font-size: 20px;
    }
    .prediction-box-churn {
        padding: 20px; background-color: #f8d7da; color: #721c24;
        border-radius: 10px; border-left: 6px solid #dc3545; font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ৩. মডেল এবং স্কেলার লোড করা (পাথ ডাইনামিক করা হয়েছে যাতে ক্লাউডে এরর না আসে)
@st.cache_resource
def load_artifacts():
    # কারেন্ট ফাইলের ডিরেক্টরি অনুযায়ী পাথ নির্ধারণ করা (লোকাল ও ক্লাউড দুই জায়গাতেই কাজ করবে)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "churn_model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}. Please ensure 'churn_model.pkl' and 'scaler.pkl' are in the correct app directory.")

# ৪. সাইডবার এবং মেইন স্ক্রিনে ইনপুট ডিস্ট্রিবিউশন
st.sidebar.header("🎯 Demographics & Contract")
st.sidebar.write("Set core account attributes:")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.radio("Senior Citizen?", ["No", "Yes"])
partner = st.sidebar.radio("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.radio("Has Dependents?", ["No", "Yes"])

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("Paperless Billing?", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

st.title("📊 Telco Customer Churn Predictor")
st.subheader("Enterprise AI Solution with Comprehensive Feature Profile Input")

st.markdown("### 🛠️ Telecom Services Subscription Profile")
st.write("Configure the technical services activated for this customer profile:")

col_srv1, col_srv2, col_srv3 = st.columns(3)

with col_srv1:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])

with col_srv2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col_srv3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("### 💰 Financial & Usage Metrics")
col_fin1, col_fin2 = st.columns(2)
with col_fin1:
    tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12)
with col_fin2:
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)

st.markdown("---")

# ৫. গাণিতিক লজিক এবং ওয়ান-হট এনকোডিং ম্যাপিং
if st.button("Run Comprehensive Risk Assessment"):
    
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': float(tenure * monthly_charges),
        'gender_Male': 1 if gender == "Male" else 0,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # ৬. ইন্টারেক্টিভ আউটপুট ডিসপ্লে
    st.markdown("### 🔮 Prediction Analytics Output")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box-churn">
            ⚠️ <strong>High Attrition Risk Detected:</strong> This customer profile is highly likely to <strong>CHURN</strong>.<br>
            <strong>Churn Risk Probability:</strong> {probability * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)
        st.warning("Strategy Recommendation: Immediate CRM intervention required. Consider offering proactive retention incentives.")
    else:
        st.markdown(f"""
        <div class="prediction-box-stay">
            🎉 <strong>Loyal Profile Verified:</strong> This customer is predicted to <strong>STAY</strong> active with the service.<br>
            <strong>Retention Confidence:</strong> {(1 - probability) * 100:.2f}%
        </div>
        """, unsafe_allow_html=True) # এখানে unsafe_allow_html ফিক্স করা হয়েছে
        st.info("Strategy Recommendation: Standard automated nurturing. Eligible for premium value-added services cross-selling.")
else:
    st.info("💡 Instructions: Adjust the demographic controls on the sidebar and service checkboxes on the dashboard, then click 'Run Comprehensive Risk Assessment'.")
