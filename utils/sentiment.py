import requests
from textblob import TextBlob
from config import NEWS_API_KEY


def get_news_sentiment(company_name):
    """
    Returns average sentiment score:
    
    Range:
    -1 → Very Negative
     0 → Neutral
    +1 → Very Positive
    """

    try:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={company_name}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&apiKey={NEWS_API_KEY}"
        )

        response = requests.get(url)
        news_data = response.json()

        sentiments = []

        if "articles" in news_data:
            for article in news_data["articles"][:10]:
                title = article.get("title", "")

                if title:
                    analysis = TextBlob(title)
                    polarity = analysis.sentiment.polarity
                    sentiments.append(polarity)

        if len(sentiments) == 0:
            return 0

        return round(
            sum(sentiments) / len(sentiments),
            2
        )

    except Exception:
        return 0