import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================
# LOAD MODEL & SCALER
# =====================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Final Grade Prediction", layout="centered")

# =====================
# TITLE
# =====================
st.title("🎓 Student Final Grade Prediction")
st.write("""
Aplikasi ini memprediksi **nilai akhir siswa** berdasarkan data akademik dan non-akademik.
Model dibangun menggunakan pendekatan regresi dan ditujukan sebagai **alat bantu analisis**, bukan keputusan mutlak.
""")

# =====================
# INPUT SECTION
# =====================
st.header("📌 Input Data Siswa")

attendance = st.slider("Attendance Rate (%)", 70, 100, 85)
study_hours = st.slider("Study Hours Per Week", 0, 40, 10)
previous_grade = st.slider("Previous Grade", 60, 100, 75)
extracurricular = st.selectbox(
    "Extracurricular Activities",
    options=[0, 1, 2, 3]
)

gender = st.radio("Gender", ["Male", "Female"])
parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
online_class = st.radio("Online Classes Taken", ["Yes", "No"])

# =====================
# PREPROCESS INPUT
# =====================
numerical_data = pd.DataFrame([[
    attendance,
    study_hours,
    previous_grade,
    extracurricular
]], columns=[
    "AttendanceRate",
    "StudyHoursPerWeek",
    "PreviousGrade",
    "ExtracurricularActivities"
])

numerical_scaled = scaler.transform(numerical_data)
numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_data.columns)

categorical_data = pd.DataFrame([{
    "OnlineClassesTaken": 1 if online_class == "Yes" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "ParentalSupport_High": 1 if parental_support == "High" else 0,
    "ParentalSupport_Low": 1 if parental_support == "Low" else 0,
    "ParentalSupport_Medium": 1 if parental_support == "Medium" else 0
}])

final_input = pd.concat([numerical_scaled_df, categorical_data], axis=1)

# =====================
# PREDICTION
# =====================
if st.button("🔍 Predict Final Grade"):
    prediction = model.predict(final_input)[0]

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Final Grade: **{prediction:.2f}**")

    st.info("""
    Model ini memiliki keterbatasan karena hubungan antar fitur tidak menunjukkan korelasi kuat.
    Hasil prediksi digunakan untuk tujuan analisis dan pembelajaran.
    """)

