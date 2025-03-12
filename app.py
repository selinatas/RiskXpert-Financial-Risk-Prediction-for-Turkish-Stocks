import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define stock tickers
tickers = ["AKBNK.IS", "GARAN.IS", "THYAO.IS", "SASA.IS", "TUPRS.IS"]

st.title("📈 RiskXpert: Financial Risk Prediction for Turkish Stocks")
st.markdown("Predict stock risk based on historical volatility and financial indicators.")

# Sidebar for stock selection
selected_stock = st.sidebar.selectbox("Select a stock to analyze", tickers)

# Fetch data
def get_stock_data(ticker):
    data = yf.download(ticker, period="2y", interval="1d")
    data["Daily Return"] = data["Adj Close"].pct_change()
    data["5-day MA"] = data["Adj Close"].rolling(window=5).mean()
    data["20-day MA"] = data["Adj Close"].rolling(window=20).mean()
    data["Volatility"] = data["Daily Return"].rolling(window=20).std()
    data.dropna(inplace=True)
    return data

# Load data
data = get_stock_data(selected_stock)

# Display stock price chart
st.subheader(f"Stock Price History for {selected_stock}")
st.line_chart(data["Adj Close"], use_container_width=True)

# Display feature plots
st.subheader("Feature Analysis")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data["Daily Return"], bins=50, kde=True, ax=ax[0])
ax[0].set_title("Daily Return Distribution")
sns.histplot(data["Volatility"], bins=50, kde=True, ax=ax[1])
ax[1].set_title("Volatility Distribution")
st.pyplot(fig)

# Define risk label
threshold = data["Volatility"].quantile(0.75)
data["Risk_Label"] = (data["Volatility"] > threshold).astype(int)

# Train model
features = ["Daily Return", "5-day MA", "20-day MA", "Volatility"]
X = data[features]
y = data["Risk_Label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Predict risk level
latest_data = X_scaled[-1].reshape(1, -1)
risk_prediction = model.predict(latest_data)

# Display risk prediction result
st.subheader("📊 Risk Prediction")
if risk_prediction[0] == 1:
    st.error("🚨 High Risk: This stock shows high volatility!")
else:
    st.success("✅ Low Risk: This stock has moderate or low volatility.")
