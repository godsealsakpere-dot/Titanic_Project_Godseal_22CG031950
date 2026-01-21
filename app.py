import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# Custom CSS Loading
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/style.css")

# Title and Overview
st.title("🚢 Titanic Survival Prediction System")
st.write("Enter passenger details to predict if they would have survived the disaster.")

# Load Model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'titanic_survival_model.pkl')

try:
    package = joblib.load(model_path)
    model = package['model']
    scaler = package['scaler']
except FileNotFoundError:
    st.error("Model file not found. Please run Part A to generate 'titanic_survival_model.pkl'.")
    st.stop()

# Input Form
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare Price (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.1)

# Predict Button
if st.button("Predict Survival"):
    # 1. Preprocess Input
    # Convert Sex to number (Male=0, Female=1 matching our training)
    sex_numeric = 0 if sex == "Male" else 1
    
    # Create array: ['pclass', 'sex', 'age', 'sibsp', 'fare']
    input_data = np.array([[pclass, sex_numeric, age, sibsp, fare]])
    
    # 2. Scale Data (Using the loaded scaler)
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)
    
    # 4. Display Result
    if prediction == 1:
        st.success("Result: **Survived**")
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")
        st.balloons()
    else:
        st.error("Result: **Did Not Survive**")
        st.write(f"Confidence: {probability[0][0]*100:.2f}%")