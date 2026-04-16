import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Air Quality Predictor", layout="wide")

# Title
st.title("🌍 Air Quality Prediction System")
st.markdown("Predict AQI and Health Risk using AI + IoT Sensors")

# Sidebar
st.sidebar.header("Enter Sensor Values")

pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.slider("PM10", 0.0, 500.0, 80.0)
co = st.sidebar.slider("CO", 0.0, 10.0, 1.0)
o3 = st.sidebar.slider("O3", 0.0, 200.0, 30.0)
nh3 = st.sidebar.slider("NH3", 0.0, 200.0, 20.0)

# Prediction
if st.sidebar.button("Predict AQI"):

    data = np.array([[pm25, pm10, co, o3, nh3]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    # Risk classification
    if prediction <= 50:
        risk = "Good 😊"
        color = "green"
    elif prediction <= 100:
        risk = "Moderate 😐"
        color = "yellow"
    elif prediction <= 200:
        risk = "Poor 😷"
        color = "orange"
    elif prediction <= 300:
        risk = "Very Poor 🤒"
        color = "red"
    else:
        risk = "Severe ☠️"
        color = "darkred"

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Predicted AQI")
        st.metric(label="AQI Value", value=round(prediction, 2))

    with col2:
        st.subheader("⚠️ Health Risk")
        st.markdown(f"<h2 style='color:{color}'>{risk}</h2>", unsafe_allow_html=True)

    # 📊 Bar Chart (Input values)
    st.subheader("📊 Sensor Data Visualization")

    features = ['PM2.5', 'PM10', 'CO', 'O3', 'NH3']
    values = [pm25, pm10, co, o3, nh3]

    fig, ax = plt.subplots()
    ax.bar(features, values)
    ax.set_title("Sensor Values")
    st.pyplot(fig)

    # 📈 AQI Gauge-like visualization
    st.subheader("📈 AQI Level Indicator")

    aqi_range = ['Good', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    aqi_values = [50, 100, 200, 300, 500]

    fig2, ax2 = plt.subplots()
    ax2.plot(aqi_values, marker='o')
    ax2.axhline(prediction, linestyle='--')
    ax2.set_title("AQI Scale")
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("💡 Built using Machine Learning + IoT Sensors")