import yfinance as yf
import joblib
import pandas as pd

from config import (
    NSE_STOCK_MASTER,
    START_DATE,
    TOP_STOCKS_CACHE_FILE
)

from utils.indicators import calculate_indicators
from utils.sentiment import get_news_sentiment


print("🚀 Updating Daily Top 3 Stocks Cache...\n")

results = []

nse_df = pd.read_csv(
    NSE_STOCK_MASTER
)

# Limit for performance
stocks_list = (
    nse_df["symbol"]
    .dropna()
    .head(300)
    .apply(lambda x: x + ".NS")
    .tolist()
)

for stock in stocks_list:
    try:
        print(f"Checking: {stock}")

        data = yf.download(
            stock,
            start=START_DATE,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            continue

        data = calculate_indicators(data)
        data = data.dropna()

        if data.empty:
            continue

        latest = data.iloc[-1:]

        ma50 = float(latest["MA50"].iloc[0])
        ma200 = float(latest["MA200"].iloc[0])
        rsi = float(latest["RSI"].iloc[0])
        macd = float(latest["MACD"].iloc[0])
        signal = float(latest["Signal_Line"].iloc[0])
        latest_close = float(latest["Close"].iloc[0])
        prev_close = float(data["Close"].iloc[-2])

        company_name = nse_df.loc[
            nse_df["symbol"] + ".NS" == stock,
            "name"
        ].values[0]

        try:
            news_sentiment = float(get_news_sentiment(company_name))
        except:
            news_sentiment = 0

        score = 0

        # ---------------------------------
        # MA Trend Strength
        # ---------------------------------

        if ma50 > ma200:
            score += 25

            # Strong bullish crossover bonus
            if (ma50 - ma200) / ma200 > 0.03:
                score += 10

        # ---------------------------------
        # RSI Strength
        # ---------------------------------

        if 40 <= rsi <= 60:
            score += 15

        elif 30 <= rsi < 40:
            score += 10

        elif rsi < 30:
            score += 5

        elif rsi > 75:
            score -= 10

        # ---------------------------------
        # MACD Strength
        # ---------------------------------

        if macd > signal:
            score += 20

            # Strong momentum bonus
            if (macd - signal) > 1:
                score += 10

        # ---------------------------------
        # Price Momentum
        # ---------------------------------

        price_change_pct = (
            (latest_close - prev_close) / prev_close
        ) * 100

        if price_change_pct > 1:
            score += 10

        elif price_change_pct < -2:
            score -= 10

        # ---------------------------------
        # Volume Breakout
        # ---------------------------------

        avg_volume = data["Volume"].tail(20).mean()
        latest_volume = float(latest["Volume"].iloc[0])

        if latest_volume > avg_volume * 1.5:
            score += 10

        # ---------------------------------
        # News Sentiment
        # ---------------------------------

        if news_sentiment >= 0.3:
            score += 15

        elif news_sentiment <= -0.2:
            score -= 10

        # ---------------------------------
        # Volatility Check
        # ---------------------------------

        volatility = data["Close"].pct_change().std() * 100

        if volatility < 2:
            score += 10

        elif volatility > 5:
            score -= 10

        # ---------------------------------
        # Final Score Clamp
        # ---------------------------------

        score = max(0, min(score, 100))

        results.append({
            "Stock": stock,
            "Score": score
        })

    except Exception as e:
        print(f"Error: {stock} → {e}")

df = pd.DataFrame(results)

filtered_df = df[
    df["Score"] >= 70
]

if len(filtered_df) >= 3:
    top_3 = filtered_df.sort_values(
        by="Score",
        ascending=False
    ).drop_duplicates(
        subset=["Score"]
    ).head(3)
else:
    # fallback if market is weak today
    top_3 = df.sort_values(
        by="Score",
        ascending=False
    ).drop_duplicates(
        subset=["Score"]
    ).head(3)

joblib.dump(
    top_3.to_dict("records"),
    TOP_STOCKS_CACHE_FILE
)

print("\n✅ Top 3 Stocks Cache Updated Successfully!")
print(top_3)