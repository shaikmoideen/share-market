import streamlit as st
import yfinance as yf
import pandas as pd
import os
import joblib

from config import START_DATE, PORTFOLIO, MANAGE_PORTFOLIO, TOP_STOCKS_CACHE_FILE
from utils.indicators import calculate_indicators
from utils.sentiment import get_news_sentiment

st.set_page_config(
    page_title="Portfolio Advisor",
    layout="wide"
)

st.title("📊 Portfolio Advisor")
st.markdown("---")

st.page_link(
    MANAGE_PORTFOLIO,
    label="⚙️ Manage Portfolio",
    icon="👜"
)

st.markdown(
    """
    <h4 style='text-align: center;'>
        Today's Investment Opportunities + Hold / Sell Analysis
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------
# User Existing Portfolio
# ---------------------------------

st.subheader("👜 Your Current Holdings")

if os.path.exists(PORTFOLIO):
    try:
        portfolio_df = pd.read_csv(PORTFOLIO)
    except:
        portfolio_df = pd.DataFrame(columns=["Stock"])
else:
    portfolio_df = pd.DataFrame(columns=["Stock"])

portfolio_stocks = portfolio_df.to_dict("records")

@st.cache_data(ttl=3600)
def analyze_stock(stock, buy_price=0, quantity=0):
    try:
        data = yf.download(
            stock,
            start=START_DATE,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            return None

        data = calculate_indicators(data)
        data = data.dropna()

        if data.empty:
            return None

        latest = data.iloc[-1:]

        ma50 = latest["MA50"].iloc[0]
        ma200 = latest["MA200"].iloc[0]
        rsi = latest["RSI"].iloc[0]
        macd = latest["MACD"].iloc[0]
        signal = latest["Signal_Line"].iloc[0]

        company_name = stock.replace(".NS", "")
        sentiment = get_news_sentiment(company_name)

        score = 0

        if ma50 > ma200:
            score += 30

        if 30 <= rsi <= 60:
            score += 20
        elif rsi < 30:
            score += 10

        if macd > signal:
            score += 30

        if sentiment >= 0.2:
            score += 20

        score = max(0, min(score, 100))

        if score >= 80:
            action = "🔥 STRONG BUY"
        elif score >= 60:
            action = "📈 HOLD"
        elif score >= 40:
            action = "⚠️ WATCH"
        else:
            action = "🔴 SELL"
        
        current_price = round(
            latest["Close"].iloc[0],
            2
        )

        profit_loss = round(
            (current_price - buy_price) * quantity,
            2
        )

        return {
            "Stock": stock,
            "Buy Price": buy_price,
            "Quantity": quantity,
            "Current Price": current_price,
            "P/L": f"₹{profit_loss:+,.2f}",
            "MA Trend": "Bullish" if ma50 > ma200 else "Bearish",
            "RSI": round(rsi, 2),
            "Sentiment": round(sentiment, 2),
            "Score": score,
            "Action": action
        }

    except Exception as e:
        print(f"{stock}: {e}")
        return None


# ---------------------------------
# Analyze Current Holdings
# ---------------------------------

if portfolio_stocks:
    portfolio_results = []

    for item in portfolio_stocks:
        result = analyze_stock(
            stock=item["Stock"],
            buy_price=item["Buy Price"],
            quantity=item["Quantity"]
        )

        if result:
            portfolio_results.append(result)

    if portfolio_results:
        df = pd.DataFrame(portfolio_results)

        action_order = {
            "🔥 STRONG BUY": 1,
            "📈 HOLD": 2,
            "⚠️ WATCH": 3,
            "🔴 SELL": 4
        }

        df["Sort_Order"] = df["Action"].map(
            action_order
        )

        df = df.sort_values(
            by="Sort_Order"
        ).drop(columns=["Sort_Order"])
        
        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )
    else:
        st.warning(
            "Unable to load portfolio analysis."
        )
else:
    st.info("No stocks added in portfolio yet.")

# ---------------------------------
# Today's Top Opportunities
# ---------------------------------

st.markdown("---")
st.subheader("🔥 Today's Best Stocks To Invest")


@st.cache_data(ttl=86400)
def load_top_stocks():
    return joblib.load(TOP_STOCKS_CACHE_FILE)


try:
    top_results = load_top_stocks()

    if top_results:
        df = pd.DataFrame(top_results)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )
    else:
        st.warning(
            "No top investment opportunities available."
        )

except Exception:
    st.warning(
        "Unable to load top investment opportunities."
    )