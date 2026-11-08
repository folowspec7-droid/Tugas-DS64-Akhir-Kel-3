%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
from features import add_features   # wajib ada agar pipeline bisa di-load

model = joblib.load("rain_model.joblib")
meta = joblib.load("meta.joblib")
med = meta["medians"]
winds = meta["wind_dirs"]

st.title("Prediksi Hujan Besok di Australia")
st.caption("Model klasifikasi (XGBoost) berdasarkan data cuaca hari ini")

c1, c2 = st.columns(2)
with c1:
    location = st.selectbox("Lokasi", meta["locations"])
    month = st.slider("Bulan", 1, 12, 6)
    rain_today = st.selectbox("Hujan hari ini?", ["No", "Yes"])
    min_temp = st.number_input("MinTemp (°C)", value=med["MinTemp"])
    max_temp = st.number_input("MaxTemp (°C)", value=med["MaxTemp"])
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    evap = st.number_input("Evaporation (mm)", min_value=0.0, value=med["Evaporation"])
    sunshine = st.number_input("Sunshine (jam)", min_value=0.0, value=med["Sunshine"])
    gust_dir = st.selectbox("WindGustDir", winds)
    gust_speed = st.number_input("WindGustSpeed (km/h)", min_value=0.0, value=med["WindGustSpeed"])
with c2:
    dir9 = st.selectbox("WindDir9am", winds)
    dir3 = st.selectbox("WindDir3pm", winds)
    ws9 = st.number_input("WindSpeed9am (km/h)", min_value=0.0, value=med["WindSpeed9am"])
    ws3 = st.number_input("WindSpeed3pm (km/h)", min_value=0.0, value=med["WindSpeed3pm"])
    h9 = st.slider("Humidity9am (%)", 0, 100, int(med["Humidity9am"]))
    h3 = st.slider("Humidity3pm (%)", 0, 100, int(med["Humidity3pm"]))
    p9 = st.number_input("Pressure9am (hPa)", value=med["Pressure9am"])
    p3 = st.number_input("Pressure3pm (hPa)", value=med["Pressure3pm"])
    cl9 = st.slider("Cloud9am (oktas)", 0, 9, int(med["Cloud9am"]))
    cl3 = st.slider("Cloud3pm (oktas)", 0, 9, int(med["Cloud3pm"]))
    t9 = st.number_input("Temp9am (°C)", value=med["Temp9am"])
    t3 = st.number_input("Temp3pm (°C)", value=med["Temp3pm"])

if st.button("Prediksi"):
    row = pd.DataFrame([{
        "Location": location, "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
        "Evaporation": evap, "Sunshine": sunshine, "WindGustDir": gust_dir,
        "WindGustSpeed": gust_speed, "WindDir9am": dir9, "WindDir3pm": dir3,
        "WindSpeed9am": ws9, "WindSpeed3pm": ws3, "Humidity9am": h9, "Humidity3pm": h3,
        "Pressure9am": p9, "Pressure3pm": p3, "Cloud9am": cl9, "Cloud3pm": cl3,
        "Temp9am": t9, "Temp3pm": t3, "RainToday": rain_today, "Month": month}])
    proba = model.predict_proba(row)[0, 1]
    st.metric("Probabilitas hujan besok", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.error("Besok kemungkinan HUJAN, bawa payung.")
    else:
        st.success("Besok kemungkinan TIDAK hujan.")
