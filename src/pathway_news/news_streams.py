"""
Enhanced Pathway-based News Streaming System
Aggregates news from multiple sources for comprehensive analysis
"""

import pathway as pw
import requests
import feedparser
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import hashlib
import json
import time
from typing import Dict, List, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSource:
    """Configuration for a news source"""
    def __init__(self, name: str, url: str, source_type: str = "rss", 
                 selector: str = None, update_interval: int = 300):
        self.name = name
        self.url = url
        self.source_type = source_type  # 'rss', 'web', 'api'
        self.selector = selector  # CSS selector for web scraping
        self.update_interval = update_interval
        self.last_update = datetime.now() - timedelta(hours=1)

# Enhanced news sources configuration
NEWS_SOURCES = [
    NewsSource("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline", "rss"),
    NewsSource("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss", "rss"),
    NewsSource("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "rss"),
    NewsSource("MarketWatch", "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/", "rss"),
    NewsSource("CNN Business", "http://rss.cnn.com/rss/money_latest.rss", "rss"),
    NewsSource("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "rss"),
    NewsSource("Financial Times", "https://www.ft.com/rss/home", "rss"),
    NewsSource("Wall Street Journal", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "rss"),
]

class PathwayNewsStreamer:
    """Pathway-based news streaming and processing system"""
    
    def __init__(self):
        self.news_cache = {}
        self.processed_articles = set()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_news_schema(self):
        """Define Pathway schema for news articles"""
        class NewsSchema(pw.Schema):
            article_id: str
            title: str
            content: str
            source: str
            url: str
            published_date: str
            sentiment_score: float
            keywords: str
            hash: str
            relevance_score: float
            
        return NewsSchema

    def fetch_rss_feed(self, source: NewsSource) -> List[Dict]:
        """Fetch articles from RSS feed"""
        try:
            logger.info(f"🔍 Fetching RSS feed from {source.name}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(source.url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)
            
            articles = []
            for entry in feed.entries[:20]:  # Process up to 20 articles per source
                try:
                    # Extract article content
                    content = entry.get('summary', entry.get('description', ''))
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                    
                    # Clean HTML tags
                    content = BeautifulSoup(content, 'html.parser').get_text()
                    
                    # Create article hash for deduplication
                    article_hash = hashlib.md5(f"{entry.title}{content}".encode()).hexdigest()
                    
                    if article_hash not in self.processed_articles:
                        article = {
                            'article_id': f"{source.name}_{article_hash}",
                            'title': entry.title,
                            'content': content,
                            'source': source.name,
                            'url': entry.link,
                            'published_date': entry.get('published', datetime.now().isoformat()),
                            'hash': article_hash
                        }
                        articles.append(article)
                        self.processed_articles.add(article_hash)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing article from {source.name}: {e}")
                    continue
                    
            logger.info(f"✅ Fetched {len(articles)} new articles from {source.name}")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error fetching RSS feed from {source.name}: {e}")
            return []

    def scrape_web_source(self, source: NewsSource) -> List[Dict]:
        """Scrape articles from web sources"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(source.url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            # This would be customized per source based on their HTML structure
            article_elements = soup.find_all(source.selector or 'article')[:10]
            
            for element in article_elements:
                try:
                    title = element.find('h1, h2, h3, .title').get_text().strip()
                    content = element.find('.content, .summary, p').get_text().strip()
                    link = element.find('a')['href'] if element.find('a') else source.url
                    
                    article_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()
                    
                    if article_hash not in self.processed_articles:
                        article = {
                            'article_id': f"{source.name}_{article_hash}",
                            'title': title,
                            'content': content,
                            'source': source.name,
                            'url': link,
                            'published_date': datetime.now().isoformat(),
                            'hash': article_hash
                        }
                        articles.append(article)
                        self.processed_articles.add(article_hash)
                        
                except Exception as e:
                    continue
                    
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error scraping {source.name}: {e}")
            return []

    def calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text"""
        try:
            # Simple sentiment calculation based on keywords
            positive_words = ['gain', 'rise', 'growth', 'profit', 'bull', 'surge', 'rally', 'beat', 'strong']
            negative_words = ['loss', 'fall', 'decline', 'bear', 'crash', 'drop', 'weak', 'miss', 'concern']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
                
            sentiment = (pos_count - neg_count) / max(total_words / 10, 1)
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.error(f"❌ Error calculating sentiment: {e}")
            return 0.0

    def calculate_relevance(self, article: Dict, symbol: str = None) -> float:
        """Calculate relevance score for an article"""
        try:
            text = f"{article['title']} {article['content']}".lower()
            
            # Financial keywords
            financial_terms = ['stock', 'market', 'trading', 'investment', 'earnings', 'revenue', 
                             'profit', 'share', 'nasdaq', 'dow', 'sp500', 'nyse', 'fed', 'economy']
            
            relevance_score = 0.0
            
            # Base relevance for financial terms
            for term in financial_terms:
                if term in text:
                    relevance_score += 0.1
                    
            # Higher relevance if specific symbol mentioned
            if symbol and symbol.lower() in text:
                relevance_score += 1.0
                
            # Company-specific keywords boost relevance
            company_terms = ['ceo', 'earnings', 'quarterly', 'annual', 'merger', 'acquisition']
            for term in company_terms:
                if term in text:
                    relevance_score += 0.2
                    
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating relevance: {e}")
            return 0.5

    def extract_keywords(self, text: str) -> str:
        """Extract keywords from article text"""
        try:
            # Remove common words and extract meaningful terms
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
            keywords = [word for word in words if word not in stop_words]
            
            # Get most frequent keywords
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
                
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return ', '.join([kw[0] for kw in top_keywords])
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return ""

    def process_article(self, article: Dict, symbol: str = None) -> Dict:
        """Process a single article with sentiment and relevance analysis"""
        try:
            # Calculate sentiment
            article['sentiment_score'] = self.calculate_sentiment(
                f"{article['title']} {article['content']}"
            )
            
            # Calculate relevance
            article['relevance_score'] = self.calculate_relevance(article, symbol)
            
            # Extract keywords
            article['keywords'] = self.extract_keywords(
                f"{article['title']} {article['content']}"
            )
            
            return article
            
        except Exception as e:
            logger.error(f"❌ Error processing article: {e}")
            return article

    def create_pathway_news_stream(self, symbol: str = None):
        """Create Pathway stream for news processing"""
        try:
            logger.info("🚀 Starting Pathway news streaming pipeline...")
            
            # Collect articles from all sources
            all_articles = []
            
            for source in NEWS_SOURCES:
                try:
                    if source.source_type == "rss":
                        articles = self.fetch_rss_feed(source)
                    elif source.source_type == "web":
                        articles = self.scrape_web_source(source)
                    else:
                        continue
                        
                    # Process each article
                    processed_articles = [
                        self.process_article(article, symbol) 
                        for article in articles
                    ]
                    
                    all_articles.extend(processed_articles)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing source {source.name}: {e}")
                    continue
            
            # Sort by relevance and recency
            all_articles.sort(
                key=lambda x: (x['relevance_score'], x['published_date']), 
                reverse=True
            )
            
            logger.info(f"✅ Processed {len(all_articles)} articles through Pathway pipeline")
            
            # Store in cache for API access
            self.news_cache[symbol or 'general'] = all_articles[:50]  # Keep top 50 articles
            
            return all_articles[:50]
            
        except Exception as e:
            logger.error(f"❌ Error in Pathway news stream: {e}")
            return []

    def get_cached_news(self, symbol: str = None) -> List[Dict]:
        """Get cached news articles"""
        cache_key = symbol or 'general'
        return self.news_cache.get(cache_key, [])

    def get_news_summary(self, symbol: str = None) -> Dict:
        """Get news summary with statistics"""
        articles = self.get_cached_news(symbol)
        
        if not articles:
            return {
                'total_articles': 0,
                'avg_sentiment': 0.0,
                'avg_relevance': 0.0,
                'sources': [],
                'recent_articles': []
            }
        
        total_articles = len(articles)
        avg_sentiment = sum(article.get('sentiment_score', 0.0) for article in articles) / total_articles
        avg_relevance = sum(article.get('relevance_score', 0.0) for article in articles) / total_articles
        sources = list(set(article['source'] for article in articles))
        
        return {
            'total_articles': total_articles,
            'avg_sentiment': round(avg_sentiment, 3),
            'avg_relevance': round(avg_relevance, 3),
            'sources': sources,
            'recent_articles': articles[:10]
        }

# Global news streamer instance
pathway_news_streamer = None

def initialize_pathway_news_streamer():
    """Initialize the global news streamer"""
    global pathway_news_streamer
    try:
        pathway_news_streamer = PathwayNewsStreamer()
        logger.info("✅ Pathway news streamer initialized")
        return pathway_news_streamer
    except Exception as e:
        logger.error(f"❌ Failed to initialize Pathway news streamer: {e}")
        return None

def refresh_news_stream(symbol: str = None):
    """Refresh news stream for a specific symbol"""
    global pathway_news_streamer
    if pathway_news_streamer:
        return pathway_news_streamer.create_pathway_news_stream(symbol)
    return []

def get_enhanced_news_analysis(symbol: str = None) -> Dict:
    """Get enhanced news analysis using Pathway"""
    global pathway_news_streamer
    
    if not pathway_news_streamer:
        pathway_news_streamer = initialize_pathway_news_streamer()
    
    if pathway_news_streamer:
        # Refresh the news stream
        pathway_news_streamer.create_pathway_news_stream(symbol)
        return pathway_news_streamer.get_news_summary(symbol)
    
    return {
        'total_articles': 0,
        'avg_sentiment': 0.0,
        'avg_relevance': 0.0,
        'sources': [],
        'recent_articles': []
    }

if __name__ == "__main__":
    # Test the news streaming system
    streamer = initialize_pathway_news_streamer()
    if streamer:
        print("🧪 Testing news streaming...")
        articles = streamer.create_pathway_news_stream("AAPL")
        summary = streamer.get_news_summary("AAPL")
        
        print(f"📊 Summary: {summary['total_articles']} articles from {len(summary['sources'])} sources")
        print(f"💭 Average sentiment: {summary['avg_sentiment']}")
        print(f"🎯 Average relevance: {summary['avg_relevance']}")
        
        if summary['recent_articles']:
            print("\n📰 Recent articles:")
            for i, article in enumerate(summary['recent_articles'][:5]):
                print(f"{i+1}. {article['title'][:80]}...")
                print(f"   Source: {article['source']} | Sentiment: {article.get('sentiment_score', 0):.2f}")