import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Hujan Besok", page_icon="🌧️", layout="centered")

MODEL_FILE = "weather_best_model_5_features.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.title("🌧️ Prediksi Hujan Besok")
st.subheader("Rain in Australia Prediction")
st.write("Masukkan kondisi cuaca hari ini untuk memperkirakan apakah besok akan terjadi hujan.")

if not os.path.exists(MODEL_FILE):
    st.error("File model belum ditemukan. Jalankan cell Save Best Model di notebook terlebih dahulu.")
    st.stop()

model = load_model()

st.header("Form Input Kondisi Cuaca")
with st.form("weather_form"):
    humidity_3pm = st.number_input("Kelembapan pukul 15:00 (%)", min_value=0.0, max_value=100.0, value=50.0)
    cloud_3pm = st.number_input("Tutupan awan pukul 15:00 (okta)", min_value=0.0, max_value=10.0, value=5.0)
    rainfall = st.number_input("Curah hujan hari ini (mm)", min_value=0.0, value=0.0)
    humidity_9am = st.number_input("Kelembapan pukul 09:00 (%)", min_value=0.0, max_value=100.0, value=60.0)
    rain_today = st.selectbox("Apakah hari ini hujan?", options=["No", "Yes"])
    submitted = st.form_submit_button("Prediksi Cuaca Besok")

if submitted:
    input_data = pd.DataFrame([{
        "Humidity3pm": humidity_3pm,
        "Cloud3pm": cloud_3pm,
        "Rainfall": rainfall,
        "Humidity9am": humidity_9am,
        "RainToday": rain_today
    }])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    rain_probability = probabilities[list(model.classes_).index(1)]

    st.header("Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Prediksi: BESOK HUJAN")
    else:
        st.success("☀️ Prediksi: BESOK TIDAK HUJAN")
    st.write(f"Probabilitas hujan: **{rain_probability * 100:.2f}%**")
    st.progress(float(rain_probability))
    st.caption("Hasil ini merupakan prediksi berdasarkan model machine learning, bukan kepastian kondisi cuaca.")
