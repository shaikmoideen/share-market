import joblib
import pandas as pd
import yfinance as yf

from config import START_DATE, NSE_STOCK_MASTER, TOP_STOCKS_CACHE_FILE
from utils.indicators import calculate_indicators
from utils.sentiment import get_news_sentiment

def analyze_stock(stock):
    try:
        ticker = stock if stock.endswith(".NS") else f"{stock}.NS"

        data = yf.download(
            ticker,
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

        return {
            "Stock": ticker,
            "Current Price": current_price,
            "MA Trend": "Bullish" if ma50 > ma200 else "Bearish",
            "RSI": round(rsi, 2),
            "Sentiment": round(sentiment, 2),
            "Score": score,
            "Action": action
        }

    except Exception as e:
        print(f"{stock}: {e}")
        return None


def update_top_stocks():
    print("Updating Top Investment Opportunities...")

    try:
        stock_df = pd.read_csv(NSE_STOCK_MASTER)

        top_stocks = (
            stock_df["symbol"]
            .dropna()
            .head(300)
            .tolist()
        )

    except Exception as e:
        print(f"Unable to load stock master: {e}")
        return

    top_results = []

    for stock in top_stocks:
        result = analyze_stock(stock)

        if result:
            top_results.append(result)

    if not top_results:
        print("No valid stock results found")
        return

    df = pd.DataFrame(top_results)

    df = df[df["Score"] >= 40]

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
        by=["Sort_Order", "Score"],
        ascending=[True, False]
    ).drop(columns=["Sort_Order"])

    final_results = df.head(10).to_dict("records")

    joblib.dump(final_results, TOP_STOCKS_CACHE_FILE)

    print(f"Saved → {TOP_STOCKS_CACHE_FILE} ✅")


if __name__ == "__main__":
    update_top_stocks()