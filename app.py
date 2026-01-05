import streamlit as st
import numpy as np
import joblib

# Load model & encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# Encode sex
sex_encoded = encoder.transform([sex])[0]
# Predict
if st.button("Predict"):
    new_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    prediction = model.predict(new_data)[0]

    if prediction == 1:
        st.success("✅ Passenger SURVIVED")
    else:
        st.error("❌ Passenger DID NOT SURVIVE")

