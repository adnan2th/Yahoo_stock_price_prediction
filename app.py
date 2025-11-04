import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Yahoo Stock High Price Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
        body {
            background-color: #F8F9FA;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #2E86C1;
            font-size: 40px;
            font-weight: bold;
        }
        .subheader {
            text-align: center;
            color: #117A65;
            font-size: 20px;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 50px;
            width: 200px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
            color: white;
        }
        .prediction-box {
            background-color: #EAF2F8;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2E86C1;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER SECTION ----
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    
 st.markdown("<h1 class='main-title'>Yahoo Stock High Price Prediction</h1>", unsafe_allow_html=True)
 st.markdown("<p class='subheader'>Powered by an LSTM Neural Network</p>", unsafe_allow_html=True)
 st.markdown("---")

# ---- APP INTRO ----
st.markdown("""
This application uses a trained **LSTM (Long Short-Term Memory)** neural network model to predict the **future high price** of a stock based on historical data.  
The model was trained using Yahoo Finance stock data focusing on the **'High'** price indicator.
""")

# ---- LOAD MODEL ----
model = load_model(r"lstm_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ---- USER INPUT ----
sequence_length = 5
st.markdown(f"""
### 🧾 How to Use:
Please enter the last **{sequence_length}** historical high prices of the stock.  
Enter them in order from **oldest → newest**.
""")

st.markdown("#### 📊 Input Historical High Prices")
input_prices = []
cols = st.columns(sequence_length)
for i in range(sequence_length):
    with cols[i]:
        price = st.number_input(f'High {i+1}', min_value=0.0, step=1.0, key=f"price_{i}")
        input_prices.append(price)

# ---- SCALING ----
scaler = MinMaxScaler()
dummy_data = np.array([[2000.0], [2500.0], [3000.0], [3500.0], [3600.0]])
scaler.fit(dummy_data)
input_prices_scaled = scaler.transform(np.array(input_prices).reshape(-1, 1))
X_predict = input_prices_scaled.reshape(1, sequence_length, 1)

# ---- PREDICTION ----
if st.button("🚀 Predict Next High Price"):
    predicted_price_scaled = model.predict(X_predict)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    st.subheader("📈 Predicted High Price:")
    st.success(f"${predicted_price[0][0]:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("---")
st.markdown("""
### ⚠️ About the Prediction
- The predicted high price represents the model’s forecast for the **next time step** (e.g., the next day’s high price).  
- This model is for **educational and demonstration purposes only** and should **not** be considered financial advice.

### 💡 Tip
To improve accuracy, retrain the model with a larger and more recent dataset and include other indicators like 'Open', 'Close', 'Volume', etc.
""")

# ---- SIDEBAR ----

st.sidebar.header("📘 About")
st.sidebar.markdown("""
This app demonstrates **time series forecasting** using an **LSTM** neural network.  
Developed with ❤️ using **Streamlit**, **TensorFlow**, and **scikit-learn**.
""")
st.sidebar.markdown("---")
st.sidebar.info("Author: *Adnan Hassan*  \nGitHub: [adnan2th](https://github.com/adnan2th)")