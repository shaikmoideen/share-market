from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import streamlit as st
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

from utils.stock_search import search_stock

from config import (
    START_DATE,
    MODEL_FILE
)

from utils.indicators import calculate_indicators
from utils.sentiment import get_news_sentiment


# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Stock Prediction Dashboard",
    layout="wide"
)

st.title("📈 AI-Based Stock Recommendation Dashboard")

# ---------------------------------
# Load Trained Model
# ---------------------------------
model = joblib.load(MODEL_FILE)
accuracy = joblib.load("model_accuracy.pkl")

# ---------------------------------
# Stock Selection
# ---------------------------------
search_query = st.text_input(
    "Search Indian Stock (Example: TCS, RELIANCE, INFY)"
)

if not search_query:
    st.stop()

suggestions = search_stock(search_query)

if not suggestions:
    st.warning("No stock suggestions found.")
    st.stop()

selected = st.selectbox(
    "Select Stock",
    [f"{s['name']} ({s['symbol']})" for s in suggestions]
)

selected_data = next(
    s for s in suggestions
    if f"{s['name']} ({s['symbol']})" == selected
)

stock = selected_data["symbol"]
company_name = selected_data["name"]

st.success(
    f"Selected: {company_name} ({stock})"
)

# ---------------------------------
# Download Latest Data
# ---------------------------------
data = yf.download(
    stock,
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

if data.empty:
    st.error("No stock data found.")
    st.stop()

# ---------------------------------
# Calculate Indicators
# ---------------------------------
data = calculate_indicators(data)

data = data.dropna()

if data.empty:
    st.error("No valid data available after calculations.")
    st.stop()

# ---------------------------------
# News Sentiment
# ---------------------------------
news_sentiment = get_news_sentiment(company_name)

# ---------------------------------
# Latest Row
# ---------------------------------
latest = data.iloc[-1:]

ma50 = latest["MA50"].iloc[0]
ma200 = latest["MA200"].iloc[0]
rsi = latest["RSI"].iloc[0]
macd = latest["MACD"].iloc[0]
signal = latest["Signal_Line"].iloc[0]

# ---------------------------------
# AI Prediction
# ---------------------------------
features = [
    "MA50",
    "MA200",
    "RSI",
    "MACD",
    "Signal_Line"
]

latest_features = data[features].tail(1)

future_prediction = model.predict(latest_features)[0]

if future_prediction == 1:
    ai_prediction = "📈 Tomorrow Price May Go UP"
else:
    ai_prediction = "📉 Tomorrow Price May Go DOWN"

# ---------------------------------
# Score-Based Recommendation
# ---------------------------------
score = 0

if ma50 > ma200:
    score += 30

if 30 <= rsi <= 50:
    score += 20
elif rsi < 30:
    score += 10

if macd > signal:
    score += 30

if rsi < 70:
    score += 20

if news_sentiment >= 0.2:
    score += 20
elif news_sentiment < -0.2:
    score -= 20

if score >= 80:
    recommendation = "🔥 STRONG BUY"
elif score >= 60:
    recommendation = "📈 BUY"
elif score >= 40:
    recommendation = "⚠️ HOLD / WATCH"
else:
    recommendation = "🔴 AVOID"

# ---------------------------------
# UI Layout
# ---------------------------------
# st.header(selected_company)
st.subheader("📊 Latest Analysis")

st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
st.write(f"## AI Prediction: {ai_prediction}")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 📈 Trend Analysis")

    st.metric("MA50", f"{ma50:.2f}")
    st.metric("MA200", f"{ma200:.2f}")

    if ma50 > ma200:
        st.success("MA50 > MA200 → Bullish Trend")
    else:
        st.warning("MA50 < MA200 → Bearish Trend")

    # News Sentiment Score
    st.metric("News Sentiment", f"{news_sentiment:.2f}")

    # News Sentiment Meaning
    if news_sentiment >= 0.5:
        st.success("News Sentiment → Very Positive 😊 >= 0.5")

    elif 0.2 <= news_sentiment < 0.5:
        st.success("News Sentiment → Positive 📈 >= 0.2 and < 0.5")

    elif -0.2 < news_sentiment < 0.2:
        st.info("News Sentiment → Neutral 😐 > -0.2 and < 0.2")

    elif -0.5 <= news_sentiment <= -0.2:
        st.warning("News Sentiment → Negative 📉 >= -0.5 and <= -0.2")

    else:
        st.error("News Sentiment → Very Negative 🔴")


with right_col:
    st.markdown("### ⚡ Momentum Analysis")

    st.metric("RSI", f"{rsi:.2f}")

    # RSI status
    if rsi > 70:
        st.error("RSI > 70 → Overbought")
    elif rsi < 30:
        st.success("RSI < 30 → Oversold")
    else:
        st.info("RSI between 30–70 → Neutral Zone")

    st.metric("MACD", f"{macd:.2f}")
    st.metric("Signal Line", f"{signal:.2f}")

    if macd > signal:
        st.success("MACD > Signal Line → Bullish Momentum")
    else:
        st.warning("MACD < Signal Line → Bearish Momentum")

# ---------------------------------
# Final Recommendation
# ---------------------------------
st.divider()

score_col1, score_col2 = st.columns(2)

with score_col1:
    st.metric("Total Score", f"{score}/100")

with score_col2:
    st.markdown("### Final Recommendation")
    st.markdown(f"# {recommendation}")

# ---------------------------------
# Charts
# ---------------------------------
st.subheader("📈 Technical Analysis Charts")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Price + MA50 + MA200")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(data["Close"], label="Close Price")
    ax1.plot(data["MA50"], label="MA50")
    ax1.plot(data["MA200"], label="MA200")
    ax1.legend()
    st.pyplot(fig1)

with col_right:
    st.markdown("### RSI Indicator")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(data["RSI"], label="RSI")
    ax2.axhline(70, linestyle="--", label="Overbought")
    ax2.axhline(30, linestyle="--", label="Oversold")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("### MACD Indicator")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(data["MACD"], label="MACD Line")
ax3.plot(data["Signal_Line"], label="Signal Line")
ax3.legend()
st.pyplot(fig3)