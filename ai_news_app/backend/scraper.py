import feedparser
from datetime import datetime

def scrape_ai_news(limit=5):
    """
    Fetches the latest AI news from Google News RSS (Taiwan/Traditional Chinese).
    """
    rss_url = "https://news.google.com/rss/search?q=Artificial+Intelligence+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(rss_url)
    
    news_items = []
    
    for entry in feed.entries[:limit]:
        item = {
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary if 'summary' in entry else "No summary available."
        }
        news_items.append(item)
        
    return news_items

if __name__ == "__main__":
    # Test the scraper
    news = scrape_ai_news()
    for n in news:
        print(f"- {n['title']}")
        print(f"  {n['link']}")
