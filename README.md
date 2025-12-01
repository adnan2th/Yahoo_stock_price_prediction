Click here to see deployment: https://yahoostockpriceprediction-nhh4kbxa3uh354erp25pyk.streamlit.app/

# 📈 Yahoo Stock Price Prediction

This project uses **Yahoo Finance historical stock data** to build a machine learning / deep learning model that predicts future stock prices. The project demonstrates data extraction, preprocessing, feature engineering, model training, evaluation, and visualization of predictions.

---

## 🚀 Features
- Fetches real stock market data using **yfinance**
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Moving averages & technical indicators  
- Machine learning / deep learning model training (LSTM, Linear Regression, etc.)
- Visualization of real vs predicted prices  
- Easy-to-use prediction script

---

## 📊 Dataset
The data is downloaded directly from **Yahoo Finance API** using the `yfinance` library.

Typical fields include:
- Open  
- High  
- Low  
- Close  
- Volume  
- Adjusted Close  

You can change the ticker symbol (e.g., AAPL, TSLA, GOOG) inside the notebook or script.

---

## 🧠 Model(s) Used
Common predictive models used in this project:
- **LSTM Neural Networks**  
- Linear Regression  
- Random Forest  
- ARIMA / SARIMAX (optional)  

You can update this section with the model you actually trained.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **yfinance**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **TensorFlow / Keras** (if using LSTM)

📈 Results
MSE : 0.0003083672294658811
MAE : 0.014182060817946894
R2  : 0.9569219431930995
- **Matplotlib / Seaborn**
- **Jupyter Notebook**
