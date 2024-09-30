import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap



# Page configuration
st.set_page_config(
    page_title="Medical Costs Concern Prediction",)
 
st.title('Predict Medical Costs Concern')
 
# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_best.joblib')
    scaler = joblib.load('scaler.joblib')
    return model_xgb, scaler
 
model_xgb, scaler = load_model_objects()
# Create SHAP explainer
explainer = shap.TreeExplainer(model_xgb)
 
# App description
with st.expander("What's this app?"):
    st.markdown("""
    This app predicts how worried a person is about medical costs, based on factors like age, education, income, and employment status.
    We've trained an AI model to analyze these inputs and give a prediction.
    Additionally, we've included an AI explainer to show how each factor impacts the prediction.
    """)
 
st.subheader('Describe yourself')
 
# User inputs
col1, col2 = st.columns(2)
 
with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    education = st.selectbox('Education Level', options=['Primary', 'Secondary', 'Tertiary'], index=1)
    income_quartile = st.radio('Income Quartile', options=['Lowest', 'Second', 'Third', 'Highest'])
 
with col2:
    employment_status = st.selectbox('Employment Status', options=['Unemployed', 'Employed', 'Self-employed', 'Student'], index=1)
 
# Map user inputs to numerical and categorical features
education_mapping = {'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
income_mapping = {'Lowest': 1, 'Second': 2, 'Third': 3, 'Highest': 4}
employment_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-employed': 2, 'Student': 3}
 
# Transform user input into a feature vector
education_num = education_mapping[education]
income_num = income_mapping[income_quartile]
employment_num = employment_mapping[employment_status]
 
# Prepare features for the model
num_features = pd.DataFrame({
    'age': [age],
    'educ': [education_num],
    'inc_q': [income_num],
    'emp_in': [employment_num]
})
num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
 
# Prediction button
if st.button('Predict Concern Level 🚀'):
    # Make prediction
    predicted_concern = model_xgb.predict(num_scaled)[0]
    # Display prediction
    st.metric(label="Predicted concern level", value=f'{round(predicted_concern)} (1: Not Worried, 3: Very Worried)')
    # SHAP explanation
    st.subheader('Concern Factors Explained 🤖')
    shap_values = explainer.shap_values(num_scaled)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, num_scaled), height=400, width=600)
    st.markdown("""
    This plot shows how each feature contributes to the predicted concern level:
    - Blue bars push the concern level lower
    - Red bars push the concern level higher
    - The length of each bar indicates the strength of the feature's impact
    """)
 
# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and AI")