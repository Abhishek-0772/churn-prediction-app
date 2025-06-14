import streamlit as st
import pandas as pd
from joblib import load

# Load the trained XGBoost model
model = load('xgboost_churn_model_v1.joblib')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# Sidebar Instructions
with st.sidebar:
    st.title("🧾 How to Use")
    st.markdown("""
    This app predicts if a customer is likely to **churn** using a trained **XGBoost** model.
    
    👉 Fill in the details on the main screen  
    👉 Click **Predict** to get the result  
    👉 Use this insight for retention strategies!
    """)

# App Title
st.title("📊 Customer Churn Prediction App")
st.markdown("Fill out the customer details below to get a prediction:")

# Input Form
with st.form("churn_form"):
    st.subheader("👤 Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("📅 Tenure (in months)", min_value=0, max_value=100, value=12)
        internet_service = st.selectbox("🌐 Internet Service", ('DSL', 'Fiber optic', 'No'))

    with col2:
        contract = st.selectbox("📄 Contract Type", ('Month-to-month', 'One year', 'Two year'))
        monthly_charges = st.number_input("💵 Monthly Charges", min_value=0, max_value=200, value=75)
        total_charges = st.number_input("🧾 Total Charges", min_value=0, max_value=10000, value=1500)

    submitted = st.form_submit_button("🔍 Predict Churn")

# Encoding
label_mapping = {
    'DSL': 0, 'Fiber optic': 1, 'No': 2,
    'Month-to-month': 0, 'One year': 1, 'Two year': 2
}

if submitted:
    encoded_internet = label_mapping[internet_service]
    encoded_contract = label_mapping[contract]

    # Match training column names exactly
    input_data = pd.DataFrame([{
        'tenure': tenure,
        'InternetService': encoded_internet,
        'Contract': encoded_contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, "predict_proba") else None

    st.subheader("📈 Prediction Result")

    if prediction[0] == 0:
        st.success("✅ This customer is likely to **stay**.")
        if probability is not None:
            st.markdown(f"**Confidence**: 🌟 {100 - probability:.2f}% chance to stay.")
    else:
        st.error("⚠️ This customer is likely to **churn**.")
        if probability is not None:
            st.markdown(f"**Confidence**: 🔥 {probability:.2f}% chance to churn.")

# Footer
st.markdown("""
<hr style="margin-top: 3rem; margin-bottom: 1rem;">
<small>
    Built with ❤️ using Streamlit | Model: <strong>XGBoost</strong>  
</small>
""", unsafe_allow_html=True)

# LinkedIn Button
st.markdown("### 🤝 Let's Connect")
st.markdown("""
<a href="https://www.linkedin.com/in/abhishekksharmma/" target="_blank">
    <button style='font-size:16px; background-color:#0072b1; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;'>
        🔗 Visit My LinkedIn
    </button>
</a>
""", unsafe_allow_html=True)
