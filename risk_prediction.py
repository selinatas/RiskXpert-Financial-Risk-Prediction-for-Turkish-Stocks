import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define Turkish stocks to analyze (BIST stocks on Yahoo Finance)
tickers = ["AKBNK.IS", "GARAN.IS", "THYAO.IS", "SASA.IS", "TUPRS.IS"]

# Fetch historical data (last 2 years)
data = {}
for ticker in tickers:
    stock_data = yf.download(ticker, period="2y", interval="1d")
    stock_data["Ticker"] = ticker
    data[ticker] = stock_data

# Combine all stock data
df = pd.concat(data.values())

# Feature Engineering: Calculate returns, volatility, and moving averages
df["Daily Return"] = df["Adj Close"].pct_change()
df["5-day MA"] = df["Adj Close"].rolling(window=5).mean()
df["20-day MA"] = df["Adj Close"].rolling(window=20).mean()
df["Volatility"] = df["Daily Return"].rolling(window=20).std()
df["Risk_Label"] = (df["Volatility"] > df["Volatility"].quantile(0.75)).astype(int)  # Top 25% volatility as high risk

# Drop NaN values
df.dropna(inplace=True)

# Select features and target
features = ["Daily Return", "5-day MA", "20-day MA", "Volatility"]
X = df[features]
y = df["Risk_Label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Plot feature importance
plt.figure(figsize=(8, 5))
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance for Financial Risk Prediction')
plt.ylabel('Importance')
plt.show()

# Display results
print(f'Model Accuracy: {accuracy:.2f}')
print(report)
