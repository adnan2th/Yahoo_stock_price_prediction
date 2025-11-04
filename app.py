import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title('Yahoo Stock High Price Prediction')

st.markdown("""
This application uses a trained LSTM (Long Short-Term Memory) neural network model to predict the future high price of a stock based on a sequence of historical high prices.

The model was trained on historical stock data and specifically focuses on the 'High' price indicator.
""")

# Load the trained model
model = load_model(r"lstm_model.h5", compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define the sequence length (should match the training sequence length)
sequence_length = 5

st.markdown(f"""
**How to use:**

Please provide the last **{sequence_length}** historical high prices of the stock you want to predict.
Enter the prices in the order from **oldest to newest** in the input fields below.
""")

# Input fields for the latest high prices
input_prices = []
for i in range(sequence_length):
    price = st.number_input(f'High Price {i+1}:', value=0.0, key=f"price_{i}")
    input_prices.append(price)

# Preprocess the input data
# Instantiate and fit a scaler (using dummy data as a workaround since the original scaler isn't available)
# In a real application, you would load the trained scaler.
scaler = MinMaxScaler()
# Fit on a small sample of data with a similar expected range
# Assuming the high prices are generally in the range of 2000-3600 based on the original notebook analysis
dummy_data = np.array([[2000.0], [2500.0], [3000.0], [3500.0], [3600.0]])
scaler.fit(dummy_data)


# Scale the input prices
input_prices_scaled = scaler.transform(np.array(input_prices).reshape(-1, 1))

# Reshape the scaled input prices
X_predict = input_prices_scaled.reshape(1, sequence_length, 1)

if st.button('Predict'):
    # Make prediction
    predicted_price_scaled = model.predict(X_predict)

    # Inverse transform the scaled prediction to get the actual price
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    st.subheader('Predicted High Price:')
    st.write(f"${predicted_price[0][0]:,.2f}")

st.markdown("""
**About the Prediction:**

The predicted high price shown above is the model's forecast for the **next** time step (e.g., the next day's high price, if your input data is daily).

**Limitations:**

This model is a simplified demonstration for educational purposes. Stock price prediction is complex and influenced by many factors not included in this model. The predictions provided here should not be considered financial advice.
""")
