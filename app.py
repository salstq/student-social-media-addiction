import streamlit as st
import pandas as pd
import joblib
# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Addicted Score Prediction", layout="centered")
st.title("📱 Social Media Addiction Prediction")
st.write("Masukkan data pengguna untuk memprediksi tingkat kecanduan media sosial.")


# ============================
# Input Numerik
# ============================
age = st.number_input("Age", min_value=15, max_value=60, value=21)
avg_usage = st.slider("Average Daily Usage (Hours)", 0, 12, 7)
sleep_hours = st.slider("Sleep Hours Per Night", 0, 12, 6)
mental_health = st.slider("Mental Health Score (1–10)", 1, 10, 6)
st.write("1 = Buruk, 10 = Sangat Baik.")
conflicts = st.slider("Conflicts Over Social Media", 0, 10, 3)
st.write("Banyak konflik akibat sosial media.")

num_df = pd.DataFrame([[age, avg_usage, sleep_hours, mental_health, conflicts]],
columns=[
'Age',
'Avg_Daily_Usage_Hours',
'Sleep_Hours_Per_Night',
'Mental_Health_Score',
'Conflicts_Over_Social_Media'
])


scaled_num = scaler.transform(num_df)
scaled_num_df = pd.DataFrame(scaled_num, columns=num_df.columns)


# ============================
# Input Kategorikal
# ============================
gender = st.selectbox("Gender", ["Male", "Female"])
academic = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
affects = st.selectbox("Affects Academic Performance?", ["Yes", "No"])
platform = st.selectbox("Most Used Platform", [
'Instagram', 'Facebook', 'TikTok', 'Twitter', 'YouTube', 'WhatsApp',
'LinkedIn', 'Snapchat', 'LINE', 'KakaoTalk', 'WeChat', 'VKontakte'
])
relationship = st.selectbox("Relationship Status", ["Single", "In Relationship", "Complicated"])


# Encoding manual
cat_data = {
'Gender': 1 if gender == 'Male' else 0,
'Academic_Level': {'High School': 0, 'Undergraduate': 1, 'Graduate': 2}[academic],
'Affects_Academic_Performance': 1 if affects == 'Yes' else 0,
}


cat_df = pd.DataFrame([cat_data])


platform_cols = [
'Most_Used_Platform_Facebook','Most_Used_Platform_Instagram','Most_Used_Platform_KakaoTalk',
'Most_Used_Platform_LINE','Most_Used_Platform_LinkedIn','Most_Used_Platform_Snapchat',
'Most_Used_Platform_TikTok','Most_Used_Platform_Twitter','Most_Used_Platform_VKontakte',
'Most_Used_Platform_WeChat','Most_Used_Platform_WhatsApp','Most_Used_Platform_YouTube'
]


relationship_cols = [
'Relationship_Status_Complicated','Relationship_Status_In Relationship','Relationship_Status_Single'
]


for col in platform_cols:
  cat_df[col] = 1 if col == f'Most_Used_Platform_{platform}' else 0


for col in relationship_cols:
  cat_df[col] = 1 if col == f'Relationship_Status_{relationship}' else 0


# ============================
# Gabungkan semua fitur
# ============================
final_input = pd.concat([scaled_num_df, cat_df], axis=1)


# Samakan urutan kolom
final_input = final_input[model.feature_names_in_]


# ============================
# Prediksi
# ============================
if st.button("Predict Addicted Score"):
  prediction = model.predict(final_input)[0]
  st.success(f"🎯 Predicted Addicted Score: **{prediction:.2f}**")
  st.write("1 = Kecanduan Rendah, 10 = Sangat Kecanduan.")
