from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import os
from config import NEWS_API_KEY

# Initialize
analyzer = SentimentIntensityAnalyzer()

# 🔑 Put your NewsAPI key here
NEWS_API_KEY = NEWS_API_KEY

newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def fetch_news(stock_name):
    """
    Fetch latest news articles for a stock
    """
    try:
        articles = newsapi.get_everything(
            q=stock_name,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )
        return articles["articles"]
    except Exception as e:
        print(f"Error fetching news for {stock_name}: {e}")
        return []


def calculate_sentiment(news_articles):
    """
    Calculate average sentiment score
    """
    if not news_articles:
        return 0

    scores = []

    for article in news_articles:
        title = article.get("title", "")
        description = article.get("description", "")

        text = f"{title}. {description}"

        sentiment = analyzer.polarity_scores(text)
        scores.append(sentiment["compound"])

    avg_score = sum(scores) / len(scores)

    return round(avg_score, 3)


def get_sentiment_label(score):
    """
    Convert score to label
    """
    if score >= 0.2:
        return "🟢 Positive"
    elif score <= -0.2:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

def get_stock_sentiment(stock_name):
    """
    Main function used in dashboard
    Always returns:
        score (float)
        label (string)
    """

    try:
        news = fetch_news(stock_name)

        # If API fails or no news found
        if not news:
            return 0, "🟡 Neutral"

        score = calculate_sentiment(news)
        label = get_sentiment_label(score)

        return score, label

    except Exception as e:
        print(f"News Sentiment Error: {e}")

        # Safe fallback
        return 0, "🟡 Neutral"