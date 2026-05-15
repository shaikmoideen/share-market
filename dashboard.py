import streamlit as st
import numpy as np
import os

import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.news_sentiment import get_stock_sentiment
from utils.sector_strength import get_sector, calculate_sector_strength, get_sector_label
from train_lstm import run_on_demand_training

from utils.stock_search import search_stock
import pandas as pd
import tensorflow as tf

from config import (
    START_DATE,
    MODEL_FILE,
    ACCURACY_FILE,
    TOP_STOCKS_CACHE_FILE,
    PORTFOLIO_ADVISOR
)

from utils.indicators import calculate_indicators

# ---------------------------------
# Dynamic LSTM Loader (Per Stock)
# ---------------------------------

@st.cache_resource
def load_lstm(symbol):
    """
    Load LSTM model from:
    1. daily
    2. weekly
    3. ondemand
    """

    symbol = symbol.upper().strip().replace(".NS", "")

    possible_paths = [
        f"models/daily/{symbol}_model.keras",
        f"models/weekly/{symbol}_model.keras",
        f"models/ondemand/{symbol}_model.keras"
    ]

    possible_scalers = [
        f"models/daily/{symbol}_scaler.pkl",
        f"models/weekly/{symbol}_scaler.pkl",
        f"models/ondemand/{symbol}_scaler.pkl"
    ]

    seq_len_path = "models/lstm_seq_len.pkl"

    model_path = None
    scaler_path = None

    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    for path in possible_scalers:
        if os.path.exists(path):
            scaler_path = path
            break

    if not model_path or not scaler_path:
        return None, None, None

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    seq_len = joblib.load(seq_len_path)

    return model, scaler, seq_len


def predict_next_days(data, model, scaler, seq_len, days=7):
    """
    Predict future stock prices using stock-specific LSTM
    """

    if model is None or scaler is None or seq_len is None:
        return None

    close_prices = data["Close"].values.reshape(-1, 1)

    if len(close_prices) < seq_len:
        st.warning("Not enough historical data for LSTM prediction.")
        return None

    try:
        scaled_data = scaler.transform(close_prices)

        last_sequence = scaled_data[-seq_len:]
        predictions = []

        current_seq = last_sequence.copy()

        for _ in range(days):
            X = current_seq.reshape(1, seq_len, 1)

            pred = model.predict(
                X,
                verbose=0
            )[0][0]

            predictions.append(pred)

            current_seq = np.append(
                current_seq[1:],
                [[pred]],
                axis=0
            )

        predictions = np.array(predictions).reshape(-1, 1)

        predicted_prices = scaler.inverse_transform(
            predictions
        )

        return predicted_prices.flatten()

    except Exception as e:
        st.error(f"LSTM Prediction Error: {str(e)}")
        return None

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Stock Prediction Dashboard",
    layout="wide"
)

st.title("📈 AI Stock Prediction Dashboard")

st.caption(
    "Live Market Analysis • AI Prediction • Portfolio Advisor • Smart Recommendations"
)

st.markdown("---")

col_nav1, col_nav2 = st.columns([3, 1])

with col_nav2:
    st.page_link(
        PORTFOLIO_ADVISOR,
        label="🚀 Portfolio Advisor",
        icon="📊"
    )

st.markdown("---")

# ---------------------------------
# Today's Top 3 Buy Opportunities
# ---------------------------------

try:
    top_stocks = joblib.load(
        TOP_STOCKS_CACHE_FILE
    )
except Exception:
    top_stocks = []

st.subheader("🔥 Today's Top 3 Buy Opportunities")
st.caption("AI-ranked stocks with the strongest buy potential today")

if not top_stocks:
    st.warning(
        "Top opportunities not available yet. Please run daily cache update first."
    )

else:
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for idx, item in enumerate(top_stocks[:3]):
        with cols[idx]:

            # Use company name if available
            company_name = item.get(
                "Company",
                item["Stock"].replace(".NS", "")
            )

            stock_symbol = item["Stock"]
            stock_score = item["Score"]

            # Score-based signal
            if stock_score >= 85:
                signal = "🔥 Strong Buy"
                status = "Excellent momentum"

            elif stock_score >= 70:
                signal = "📈 Buy"
                status = "Good opportunity"

            elif stock_score >= 50:
                signal = "⚠️ Watchlist"
                status = "Needs confirmation"

            else:
                signal = "🔴 Avoid"
                status = "Weak setup"

            with st.container(border=True):

                # Company Name
                st.markdown(
                    f"### {medals[idx]} {company_name}"
                )

                # Small symbol below name
                st.caption(
                    f"Symbol: {stock_symbol}"
                )

                # AI Score
                st.metric(
                    label="AI Score",
                    value=f"{stock_score}/100"
                )

                # Recommendation
                if "Strong Buy" in signal:
                    st.success(signal)

                elif "Buy" in signal:
                    st.info(signal)

                elif "Watchlist" in signal:
                    st.warning(signal)

                else:
                    st.error(signal)

                # Status text
                st.caption(status)

# ---------------------------------
# Load Trained Model
# ---------------------------------
saved_model = joblib.load(MODEL_FILE)

model = saved_model["model"]
features = saved_model["features"]

accuracy = joblib.load(ACCURACY_FILE)

# ---------------------------------
# Stock Selection (Working Autocomplete)
# ---------------------------------

st.markdown("## 🔍 Search Indian Stock")

search_query = st.text_input(
    "Search Stock",
    placeholder="Type stock like TCS, TCS.NS, Reliance, Infosys...",
    key="stock_search",
    label_visibility="collapsed"
)

if not search_query:
    st.info("Start typing stock name...")
    st.stop()

# Convert safely to string
search_query = str(search_query).strip().upper()

# Remove .NS suffix if user types it
if search_query.endswith(".NS"):
    search_query = search_query.replace(".NS", "")

# Get suggestions using your existing function
suggestions = search_stock(search_query)

if not suggestions:
    st.warning("No stock suggestions found.")
    st.stop()

# Auto dropdown suggestions
selected = st.selectbox(
    "Select Matching Stock",
    [f"{s['name']} ({s['symbol']})" for s in suggestions],
    index=0
)

selected_data = next(
    s for s in suggestions
    if f"{s['name']} ({s['symbol']})" == selected
)

stock = selected_data["symbol"]
clean_stock = stock.replace(".NS", "")
ticker = stock if stock.endswith(".NS") else f"{stock}.NS"
company_name = selected_data["name"]

# ---------------------------------
# Choice 3 → On-Demand LSTM Training
# ---------------------------------

model_path_daily = f"models/daily/{clean_stock}_model.keras"
model_path_weekly = f"models/weekly/{clean_stock}_model.keras"
model_path_ondemand = f"models/ondemand/{clean_stock}_model.keras"

if (
    not os.path.exists(model_path_daily)
    and not os.path.exists(model_path_weekly)
    and not os.path.exists(model_path_ondemand)
):
    with st.spinner(
        f"Training LSTM model for {clean_stock}..."
    ):
        run_on_demand_training(clean_stock)

    st.success(
        f"LSTM model created successfully for {clean_stock}"
    )

# Load stock-specific LSTM model dynamically
lstm_model, lstm_scaler, lstm_seq_len = load_lstm(clean_stock)

st.info(
    f"Selected: {company_name} ({clean_stock})"
)

# ---------------------------------
# Live Market Price Only
# ---------------------------------

@st.fragment(run_every="10s")
def live_price_section(stock):
    st.subheader("💰 Live Market Price")

    try:
        ticker = yf.Ticker(stock)
        stock_info = ticker.fast_info

        current_price = (
            stock_info.get("lastPrice")
            or stock_info.get("currentPrice")
        )

        previous_close = stock_info.get("previousClose")

        if current_price and previous_close:
            price_change = (
                current_price - previous_close
            )

            price_change_percent = (
                price_change / previous_close
            ) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Price",
                    f"₹{current_price:.2f}"
                )

            with col2:
                st.metric(
                    "Previous Close",
                    f"₹{previous_close:.2f}"
                )

            with col3:
                st.metric(
                    "Today's Change",
                    f"₹{price_change:.2f}",
                    f"{price_change_percent:.2f}%"
                )

        else:
            st.warning(
                "Unable to fetch live price."
            )

    except Exception:
        st.warning(
            "Unable to fetch live market price."
        )

# ---------------------------------
# Download Latest Data
# ---------------------------------

data = yf.download(
    ticker,
    start=START_DATE,
    auto_adjust=True,
    progress=False
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    st.warning(
        f"No data found for {stock}"
    )
    st.stop()

live_price_section(ticker)

# ---------------------------------
# Use Cached Sector Strength Data
# ---------------------------------

CACHE_FILE = "sector_strength_cache.pkl"

try:
    sector_strengths = joblib.load(CACHE_FILE)
except Exception:
    sector_strengths = {}

# Current selected stock sector
current_sector = get_sector(
    stock.replace(".NS", "")
)

# Get selected sector value
sector_value = sector_strengths.get(
    current_sector,
    0
)

# Convert numeric value → label
sector_label = get_sector_label(
    sector_value
)

# ---------------------------------
# Calculate Indicators
# ---------------------------------
data = calculate_indicators(data)

data = data.dropna()

if data.empty:
    st.error("No valid data available after calculations.")
    st.stop()

future_prices = predict_next_days(
    data,
    lstm_model,
    lstm_scaler,
    lstm_seq_len,
    days=7
)

# ---------------------------------
# News Sentiment
# ---------------------------------
sentiment_score, sentiment_label = get_stock_sentiment(company_name)

# keep numeric score for logic
news_sentiment = sentiment_score

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
latest_features = data[features].tail(1)

future_prediction = model.predict(latest_features)[0]
prediction_probability = model.predict_proba(latest_features)[0]

confidence = round(
    max(prediction_probability) * 100
)

if future_prediction == 1:
    ai_prediction = f"📈 Tomorrow Price May Go UP ({confidence}% Confidence)"
else:
    ai_prediction = f"📉 Tomorrow Price May Go DOWN ({confidence}% Confidence)"

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

score = max(0, min(score, 100))

volatility = data["Close"].pct_change().std() * 100

if volatility < 1.5:
    risk_level = "🟢 Low Risk"
elif volatility < 3:
    risk_level = "🟡 Medium Risk"
else:
    risk_level = "🔴 High Risk"

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
    st.metric("News Sentiment", sentiment_label)

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

score_col1, score_col2, score_col3 = st.columns(3)

with score_col1:
    st.metric("Total Score", f"{score}/100")

with score_col2:
    st.markdown("### Final Recommendation")

    if "STRONG BUY" in recommendation:
        st.success(recommendation)
    elif "BUY" in recommendation:
        st.info(recommendation)
    elif "HOLD" in recommendation:
        st.warning(recommendation)
    else:
        st.error(recommendation)

with score_col3:
    st.metric("Risk Level", risk_level)
    st.metric("Sector Strength", f"{sector_label} ({sector_value}%)")

# ---------------------------------
# Charts
# ---------------------------------
st.markdown(
    """
    <h2 style='text-align: center;'>
        📈 Technical Analysis Charts
    </h2>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns(2)

with col_left:

    st.subheader("📅 Next 7 Days Price Prediction")

    if future_prices is not None and len(future_prices) > 0:

        future_df = pd.DataFrame({
            "Day": [
                f"Day {i+1}"
                for i in range(len(future_prices))
            ],
            "Predicted Price": future_prices
        })

        future_df = future_df.set_index("Day")

        st.line_chart(future_df)

        if future_prices[-1] > future_prices[0]:
            st.success(
                "📈 LSTM Trend: Bullish (Price likely to increase)"
            )
        else:
            st.warning(
                "📉 LSTM Trend: Bearish (Price may decline)"
            )

    else:
        st.warning(
            "LSTM future prediction unavailable for this stock."
        )

    # ---------------------------------
    # Today's Live Intraday Chart
    # ---------------------------------

    live_chart_data = yf.download(
        ticker,
        period="1d",
        interval="5m",
        auto_adjust=True,
        progress=False
    )

    if isinstance(live_chart_data.columns, pd.MultiIndex):
        live_chart_data.columns = (
            live_chart_data.columns.get_level_values(0)
        )

    if not live_chart_data.empty:
        st.subheader("⚡ Today's Live Intraday Chart")

        live_chart_data["MA50"] = live_chart_data["Close"].rolling(window=50).mean()
        live_chart_data["MA200"] = live_chart_data["Close"].rolling(window=200).mean()

        fig_live = go.Figure()

        # Candlestick chart
        fig_live.add_trace(
            go.Candlestick(
                x=live_chart_data.index,
                open=live_chart_data["Open"],
                high=live_chart_data["High"],
                low=live_chart_data["Low"],
                close=live_chart_data["Close"],
                name="Price"
            )
        )

        # MA50 Line
        fig_live.add_trace(
            go.Scatter(
                x=live_chart_data.index,
                y=live_chart_data["MA50"],
                mode="lines",
                name="MA50"
            )
        )

        # MA200 Line
        fig_live.add_trace(
            go.Scatter(
                x=live_chart_data.index,
                y=live_chart_data["MA200"],
                mode="lines",
                name="MA200"
            )
        )

        fig_live.update_layout(
            title="Live Stock Price Movement",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600
        )

        st.plotly_chart(
            fig_live,
            use_container_width=True
        )

    else:
        st.subheader("⚡ Last 1 Month Intraday Chart")
        st.info(
            "Live intraday chart unavailable. Showing recent trend instead."
        )

        fallback_chart = yf.download(
            ticker,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(fallback_chart.columns, pd.MultiIndex):
            fallback_chart.columns = (
                fallback_chart.columns.get_level_values(0)
            )

        if not fallback_chart.empty:
            st.line_chart(
                fallback_chart["Close"]
            )
        else:
            st.warning(
                "Unable to load chart data."
            )

with col_right:
    # ---------------------------------
    # Historical Trend + Moving Average
    # ---------------------------------
    st.subheader("📊 Historical Trend + Moving Averages")

    chart_data = data[
        ["Close", "MA50", "MA200"]
    ].copy()

    if not chart_data.empty:
        st.line_chart(chart_data)
    else:
        st.warning(
            "Unable to load historical chart."
        )

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