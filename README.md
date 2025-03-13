# **RiskXpert: Financial Risk Prediction for Turkish Stocks** 🚀📈

## **Overview**
RiskXpert is a **machine learning-powered financial risk prediction model** that analyzes Turkish stock market data to assess **volatility risk**. By leveraging historical stock data from **Borsa Istanbul (BIST)**, the model predicts whether a stock falls into a **high-risk category** based on volatility patterns.

This project is ideal for investors, analysts, and finance enthusiasts who want to explore how **AI-driven risk assessment** can be applied in real-world trading scenarios. 📊✨

---
## **Features**
✅ **Stock Data Collection**: Automatically fetches data from Yahoo Finance (BIST stocks)  
✅ **Feature Engineering**: Computes **moving averages, volatility, and daily returns**  
✅ **Machine Learning Model**: Uses **Random Forest Classifier** for risk prediction  
✅ **Performance Evaluation**: Model accuracy, feature importance visualization  
✅ **Interactive Dashboard**: (Upcoming) A Streamlit-based web app for real-time stock risk insights  

---
## **Data Collection** 📜
The model fetches **2 years** of historical data for selected **BIST stocks**:
- **AKBNK.IS** (Akbank)
- **GARAN.IS** (Garanti Bank)
- **THYAO.IS** (Turkish Airlines)
- **SASA.IS** (SASA Polyester)
- **TUPRS.IS** (Tupras)

Features Extracted:
- **Daily Return**: Percentage change in adjusted closing price
- **5-day & 20-day Moving Averages**
- **Volatility**: 20-day rolling standard deviation of returns
- **Risk Label**: High risk if volatility is in the top 25%

---
## **Installation & Usage** ⚙️
### **1. Clone Repository**
```bash
git clone https://github.com/selinatas/riskxpert.git
cd riskxpert
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Run Risk Prediction Model**
```bash
python risk_prediction.py
```
### **4. (Upcoming) Run Streamlit Dashboard**
```bash
streamlit run app.py
```

---
## **Model Performance & Insights** 🎯
### **Feature Importance**
After training the **Random Forest Classifier**, we analyze which factors contribute most to risk prediction.

📌 **Most Important Features:**
1️⃣ **Volatility** 🔥
2️⃣ **Daily Return**
3️⃣ **Moving Averages (5-day & 20-day)**

### **Model Accuracy**
The model achieves **high accuracy** in identifying high-risk stocks based on historical data. Detailed classification metrics are provided in the output.

---
## **Upcoming Enhancements** 🚀
🔹 **Deploy a Streamlit Dashboard** for real-time visualization  
🔹 **Incorporate Fundamental Data** (P/E Ratio, Market Cap)  
🔹 **Test with XGBoost for performance improvements**  

---
## **License & Acknowledgment**
🔗 Open-source project under MIT License. Data sourced via **Yahoo Finance API**. Contributions & feedback are welcome! 🎉

📩 Questions? Reach out via [LinkedIn](https://www.linkedin.com/in/selin-atas/) or open an issue!
