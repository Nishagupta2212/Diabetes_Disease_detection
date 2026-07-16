"""Streamlit UI for the PIMA diabetes prediction model."""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv("diabetes.csv")


model, scaler, metrics = load_artifacts()
diabetes_dataset = load_dataset()

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

st.title("🩺 Diabetes Prediction")
st.caption("PIMA Indians Diabetes Dataset — SVM (linear kernel) classifier")

tab_predict, tab_data, tab_model = st.tabs(["Predict", "Dataset", "Model"])

with tab_predict:
    st.subheader("Enter patient details")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

    if st.button("Predict", type="primary"):
        input_data = np.asarray(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        )
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)[0]

        if prediction == 0:
            st.success("Result: The person is **not diabetic**.")
        else:
            st.error("Result: The person is **diabetic**.")

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(diabetes_dataset.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Shape: {diabetes_dataset.shape[0]} rows x {diabetes_dataset.shape[1]} columns")
        st.write("Outcome counts (0 = Non-Diabetic, 1 = Diabetic)")
        st.bar_chart(diabetes_dataset["Outcome"].value_counts())
    with col2:
        st.write("Mean feature values by Outcome")
        st.dataframe(diabetes_dataset.groupby("Outcome").mean(), use_container_width=True)

    st.write("Statistical summary")
    st.dataframe(diabetes_dataset.describe(), use_container_width=True)

with tab_model:
    st.subheader("Model performance")
    col1, col2 = st.columns(2)
    col1.metric("Training accuracy", f"{metrics['train_accuracy'] * 100:.2f}%")
    col2.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
    st.write("Algorithm: Support Vector Machine (linear kernel)")
    st.write("Features used:", ", ".join(FEATURES))
