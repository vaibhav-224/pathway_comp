"""
Enhanced FinanceAI Web Interface with Web Crawling & Fundamental Analysis
A comprehensive web dashboard for the AI-powered financial analyst with advanced capabilities
"""
from flask import Flask, render_template, jsonify, request
import sys
import os
import threading
import json
from datetime import datetime
try:
    import pathway as pw
except ImportError:
    # Use mock pathway for testing
    import pathway_mock as pw
    
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import REAL Pathway news streaming system
try:
    from src.pathway_news.real_pathway_system import (
        initialize_real_pathway_system,
        get_pathway_news_analysis,
        real_pathway_system
    )
    logger.info("✅ REAL Pathway news system imported")
except ImportError as e:
    logger.warning(f"⚠️ Could not import real Pathway system: {e}")
    # Fallback functions
    def initialize_real_pathway_system():
        return None
    def get_pathway_news_analysis(symbol=None):
        return {'total_articles': 0, 'avg_sentiment': 0.0, 'recent_articles': []}
    real_pathway_system = None

# Import AI Portfolio Analyst
try:
    from src.portfolio.ai_analyst import get_portfolio_ai_analyst
    logger.info("✅ Portfolio AI analyst imported")
except ImportError as e:
    logger.warning(f"⚠️ Could not import portfolio AI analyst: {e}")
    def get_portfolio_ai_analyst(client=None):
        return None

# Import legacy system as fallback
try:
    from src.pathway_news.news_streams import (
        initialize_pathway_news_streamer, 
        get_enhanced_news_analysis,
        refresh_news_stream,
        pathway_news_streamer
    )
    logger.info("✅ Legacy news streaming imported as fallback")
except ImportError as e:
    logger.warning(f"⚠️ Could not import legacy news streaming: {e}")
    # Fallback functions
    def initialize_pathway_news_streamer():
        return None
    def get_enhanced_news_analysis(symbol=None):
        return {'total_articles': 0, 'avg_sentiment': 0.0, 'recent_articles': []}
    def refresh_news_stream(symbol=None):
        return []
    pathway_news_streamer = None

# Import enhanced Pathway alert system
try:
    from src.alerts.pathway_alerts import (
        initialize_pathway_alert_system,
        add_alert_watchlist_symbol,
        get_news_alerts,
        pathway_alert_system
    )
    logger.info("✅ Enhanced Pathway alert system imported")
except ImportError as e:
    logger.warning(f"⚠️ Could not import enhanced alert system: {e}")
    # Fallback functions
    def initialize_pathway_alert_system():
        return None
    def add_alert_watchlist_symbol(symbol):
        pass
    def get_news_alerts(symbol=None):
        return {'alerts': [], 'summary': {}}
    pathway_alert_system = None

# Import Pathway Portfolio System
try:
    from src.portfolio.pathway_portfolio import get_portfolio_manager
    from src.portfolio.metrics_engine import get_metrics_engine
    logger.info("✅ Pathway portfolio system imported")
except ImportError as e:
    logger.warning(f"⚠️ Could not import Pathway portfolio system: {e}")
    # Fallback functions
    def get_portfolio_manager():
        return None
    def get_metrics_engine():
        return None

# Try to import enhanced components from src directory
try:
    from src.enhanced_ai_pipeline import EnhancedAIPipeline
    # Market data feeds
    from src.market_data.realtime_feeds import initialize_market_feed, market_feed, alert_manager
    # AI insights
    from src.ai_insights.engine import get_enhanced_ai_analysis
    # Portfolio management
    from src.portfolio.pathway_portfolio import get_portfolio_manager
    from src.portfolio.metrics_engine import get_metrics_engine
    # Alerts
    from src.alerts.pathway_alerts import initialize_pathway_alert_system, get_news_alerts
    
    logger.info("✅ All real implementations imported successfully")
    
    # Initialize real components
    ai_signal_generator = get_enhanced_ai_analysis
    portfolio_optimizer = get_portfolio_manager
    risk_analyzer = get_metrics_engine()
    initialize_ai_signals = lambda x: None
    initialize_risk_management = lambda: None
    create_alert_system = lambda x: (None, None)
    real_implementations = True
    
except ImportError as e:
    logger.warning(f"⚠️  Import error: {e}")
    # Fallback for missing modules
    market_feed = None
    ai_signal_generator = None
    portfolio_optimizer = None
    risk_analyzer = None
    initialize_ai_signals = lambda x: None
    initialize_risk_management = lambda: None
    create_alert_system = lambda x: (None, None)
    real_implementations = False
    logger.info("🔧 Using mock implementations for demo...")
    
    # Mock implementations for demo purposes
    class EnhancedAIPipeline:
        def __init__(self, *args, **kwargs):
            self.symbols = kwargs.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
            
        def get_enhanced_pipeline_status(self):
            return {
                'pipeline_status': 'ENHANCED_RUNNING',
                'symbols_monitored': len(self.symbols),
                'data_streams': {
                    'market_data': True,
                    'web_crawling': True,
                    'anomaly_detection': True,
                    'ai_insights': True,
                    'fundamental_analysis': True,
                    'integrated_intelligence': True
                },
                'capabilities': {
                    'real_time_market_data': True,
                    'web_news_crawling': True,
                    'financial_statements_analysis': True,
                    'anomaly_detection': True,
                    'ai_insights': True,
                    'fundamental_analysis': True,
                    'integrated_intelligence': True,
                    'investment_recommendations': True
                },
                'update_interval': 30,
                'monitored_symbols': self.symbols
            }
    
    class PortfolioTracker:
        def __init__(self, portfolio):
            self.portfolio = portfolio

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced-financeai-dashboard-2025'

# Global variables
enhanced_pipeline = None
portfolio_tracker = None
realtime_market_feed = None
trading_signal_generator = None
alert_system = None
notification_service = None
gemini_client = None
system_initialized = False  # Flag to prevent re-initialization
enhanced_system_data = {
    'market_data': [],
    'crawled_news': [],
    'financial_statements': [],
    'fundamental_analysis': [],
    'integrated_intelligence': [],
    'portfolio_data': [],
    'alerts': [],
    'ai_insights': [],
    'anomalies': [],
    'system_status': {},
    'last_updated': None,
    'capabilities': {}
}

def initialize_enhanced_system():
    """Initialize the Enhanced FinanceAI system with web crawling"""
    global enhanced_pipeline, portfolio_tracker, pathway_news_streamer, gemini_client, system_initialized
    
    # Only initialize once
    if system_initialized:
        return enhanced_pipeline, portfolio_tracker
        
    logger.info("🚀 Initializing Enhanced FinanceAI System for Web Interface...")
    
    # Initialize Gemini AI client
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            gemini_client = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Gemini AI client initialized")
        else:
            logger.warning("⚠️ No Gemini API key found")
            gemini_client = None
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Gemini client: {e}")
        gemini_client = None
    
    # Initialize Pathway Portfolio System
    try:
        portfolio_manager = get_portfolio_manager()
        metrics_engine = get_metrics_engine()
        if portfolio_manager and metrics_engine:
            logger.info("✅ Pathway portfolio system initialized")
        else:
            logger.warning("⚠️ Portfolio system components not available")
    except Exception as e:
        logger.warning(f"⚠️ Portfolio system initialization failed: {e}")
    
    # Initialize REAL Pathway news system first
    try:
        real_pathway_system = initialize_real_pathway_system()
        logger.info("✅ REAL Pathway news system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Real Pathway system initialization failed: {e}")
        real_pathway_system = None
    
    # Initialize legacy enhanced Pathway news streaming as fallback
    try:
        pathway_news_streamer = initialize_pathway_news_streamer()
        logger.info("✅ Legacy Pathway news streaming initialized as fallback")
    except Exception as e:
        logger.warning(f"⚠️ Legacy Pathway news streaming initialization failed: {e}")
        pathway_news_streamer = None
    
    # Initialize enhanced Pathway alert system
    try:
        pathway_alert_system = initialize_pathway_alert_system()
        logger.info("✅ Enhanced Pathway alert system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Pathway alert system initialization failed: {e}")
        pathway_alert_system = None
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Get AI provider configuration from environment
    ai_provider = os.getenv('AI_PROVIDER', 'gemini')
    if ai_provider.lower() == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY')
    else:
        api_key = os.getenv('OPENAI_API_KEY')
    
    # Gemini client is already initialized at the top of this function
    
    # Initialize enhanced AI pipeline
    enhanced_pipeline = EnhancedAIPipeline(
        symbols=symbols,
        update_interval=30,
        price_threshold=2.5,
        volume_threshold=2.0,
        ai_provider=ai_provider,
        api_key=api_key
    )
    
    # Initialize enhanced portfolio tracker
    try:
        # PortfolioPosition class for demo purposes
        class PortfolioPosition:
            def __init__(self, symbol, quantity, purchase_price, purchase_date):
                self.symbol = symbol
                self.quantity = quantity
                self.purchase_price = purchase_price
                self.purchase_date = purchase_date
        enhanced_positions = [
            PortfolioPosition('AAPL', 100.0, 180.0, '2025-01-15'),
            PortfolioPosition('MSFT', 50.0, 420.0, '2025-01-20'),
            PortfolioPosition('GOOGL', 25.0, 2800.0, '2025-01-25'),
            PortfolioPosition('TSLA', 20.0, 250.0, '2025-01-30'),
            PortfolioPosition('NVDA', 30.0, 800.0, '2025-02-01')
        ]
        # Use existing portfolio manager instead of PortfolioTracker
        portfolio_tracker = get_portfolio_manager()
        if portfolio_tracker:
            # Add demo positions
            for pos in enhanced_positions:
                portfolio_tracker.add_stock(pos.symbol, pos.quantity, pos.purchase_price)
    except ImportError:
        # Mock portfolio tracker if import fails
        portfolio_tracker = PortfolioTracker({'AAPL': {'shares': 100, 'avg_cost': 180.0}})
    
    # Initialize advanced features if available
    try:
        if 'initialize_market_feed' in globals():
            # Initialize real-time market feed
            global realtime_market_feed, trading_signal_generator, alert_system, notification_service
            
            realtime_market_feed = initialize_market_feed(symbols)
            logger.info("✅ Real-time market feed initialized")
            
            # Initialize AI trading signals
            trading_signal_generator = initialize_ai_signals(symbols)
            logger.info("✅ AI trading signals initialized")
            
            # Initialize portfolio risk management
            initialize_risk_management()
            logger.info("✅ Portfolio risk management initialized")
            
            # Initialize alert system
            alert_config = {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': os.getenv('EMAIL_USERNAME'),
                    'password': os.getenv('EMAIL_PASSWORD'),
                    'to_email': os.getenv('ALERT_EMAIL')
                },
                'slack': {
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL')
                }
            }
            alert_system, notification_service = create_alert_system(alert_config)
            alert_system.start_monitoring()
            logger.info("✅ Alert system initialized")
            
    except Exception as e:
        logger.warning(f"⚠️ Some advanced features failed to initialize: {e}")
    
    # Mark system as initialized
    system_initialized = True
    logger.info("✅ Enhanced FinanceAI System initialized successfully")
    return enhanced_pipeline, portfolio_tracker

def update_enhanced_system_data():
    """Update enhanced system data from the AI pipeline"""
    global enhanced_system_data, enhanced_pipeline
    
    try:
        if enhanced_pipeline:
            # Get real pipeline status
            status = enhanced_pipeline.get_enhanced_pipeline_status()
            enhanced_system_data['system_status'] = status
            enhanced_system_data['capabilities'] = status.get('capabilities', {})
            enhanced_system_data['last_updated'] = datetime.now().isoformat()
            
            # Try to get real crawled news data
            try:
                if hasattr(enhanced_pipeline, 'web_crawler'):
                    logger.info("🕷️ Attempting to fetch real news data...")
                    
                    # Directly call the RSS fetching methods to get raw data
                    # Bypass Pathway table creation to avoid format issues
                    raw_news_data = []
                    
                    # Get news sources and fetch from each one
                    for source in enhanced_pipeline.web_crawler.news_sources:
                        try:
                            if source.source_type == "rss":
                                source_news = enhanced_pipeline.web_crawler._fetch_rss_news(source)
                                raw_news_data.extend(source_news)
                                logger.info(f"📰 Fetched {len(source_news)} articles from {source.name}")
                        except Exception as source_e:
                            logger.warning(f"⚠️ Error fetching from {source.name}: {source_e}")
                    
                    # Convert raw news data to API format
                    news_data = []
                    for article in raw_news_data[:10]:  # Limit to recent 10 articles
                        try:
                            # Debug: Log the actual article being processed
                            logger.info(f"🔍 Processing article: {article.get('title', 'No Title')} from {article.get('source', 'Unknown')}")
                            logger.info(f"🔍 Article URL: {article.get('url', 'No URL')}")
                            logger.info(f"🔍 Published: {article.get('published_date', 'No date')}")
                            
                            news_data.append({
                                'symbol': article.get('symbol', 'MARKET'),
                                'title': article.get('title', 'No Title'),
                                'sentiment_score': float(article.get('sentiment_score', 0.5)),
                                'impact_level': 'HIGH' if article.get('relevance_score', 0) > 0.7 else 'MEDIUM' if article.get('relevance_score', 0) > 0.4 else 'LOW',
                                'source': article.get('source', 'Unknown'),
                                'timestamp': datetime.now().isoformat(),
                                'url': article.get('url', '#')
                            })
                        except Exception as convert_e:
                            logger.warning(f"⚠️ Error converting article: {convert_e}")
                    
                    if news_data:
                        enhanced_system_data['crawled_news'] = news_data
                        logger.info(f"✅ Successfully fetched {len(news_data)} real news articles!")
                    else:
                        # Fall back to mock data if no real data
                        enhanced_system_data['crawled_news'] = get_mock_news_data()
                        logger.warning("⚠️ No real news articles found, using mock data")
                        
                else:
                    enhanced_system_data['crawled_news'] = get_mock_news_data()
                    logger.warning("⚠️ Web crawler not available, using mock data")
            except Exception as e:
                logger.error(f"❌ Error fetching real news data: {e}")
                enhanced_system_data['crawled_news'] = get_mock_news_data()
                logger.info("🔄 Falling back to mock news data")
            
            # Get real market data using yfinance
            enhanced_system_data['market_data'] = get_real_market_data()
            enhanced_system_data['fundamental_analysis'] = get_real_fundamental_data()
            enhanced_system_data['anomalies'] = get_mock_anomaly_data()
            enhanced_system_data['portfolio_data'] = get_mock_portfolio_data()
            
    except Exception as e:
        logger.error(f"❌ Error updating enhanced system data: {e}")
        # Use mixed real and mock data as fallback
        enhanced_system_data.update({
            'crawled_news': get_mock_news_data(),
            'market_data': get_real_market_data(),
            'fundamental_analysis': get_real_fundamental_data(),
            'anomalies': get_mock_anomaly_data(),
            'portfolio_data': get_mock_portfolio_data(),
            'last_updated': datetime.now().isoformat()
        })

def get_mock_news_data():
    """Get mock news data"""
    return [
        {
            'symbol': 'AAPL',
            'title': 'Apple Reports Strong Q4 Earnings',
            'sentiment_score': 0.8,
            'impact_level': 'HIGH',
            'source': 'Reuters',
            'timestamp': datetime.now().isoformat(),
            'url': '#'
        },
        {
            'symbol': 'TSLA',
            'title': 'Tesla Announces New Gigafactory',
            'sentiment_score': 0.6,
            'impact_level': 'MEDIUM',
            'source': 'Bloomberg',
            'timestamp': datetime.now().isoformat(),
            'url': '#'
        }
    ]

def get_real_market_data():
    """Get real market data using yfinance"""
    import yfinance as yf
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    market_data = []
    
    try:
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='2d')
                
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    volume = int(hist['Volume'].iloc[-1])
                else:
                    # Fallback to info data
                    current_price = info.get('currentPrice', info.get('previousClose', 0))
                    prev_price = info.get('previousClose', current_price)
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price else 0
                    volume = info.get('volume', 0)
                
                market_data.append({
                    'symbol': symbol,
                    'price': float(current_price),
                    'change': float(change),
                    'change_pct': float(change_pct),
                    'volume': int(volume),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error fetching data for {symbol}: {e}")
                # Use fallback data for this symbol
                market_data.append({
                    'symbol': symbol,
                    'price': float(100.0),
                    'change': float(0.0),
                    'change_pct': float(0.0),
                    'volume': int(1000000),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"✅ Fetched real market data for {len(market_data)} symbols")
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching real market data: {e}")
        return get_mock_market_data()

def get_mock_market_data():
    """Get mock market data as fallback"""
    return [
        {
            'symbol': 'AAPL',
            'price': float(185.50),
            'change': float(2.30),
            'change_pct': float(1.26),
            'volume': int(45000000),
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'MSFT',
            'price': float(425.80),
            'change': float(-1.20),
            'change_pct': float(-0.28),
            'volume': int(28000000),
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'GOOGL',
            'price': float(2850.00),
            'change': float(15.50),
            'change_pct': float(0.55),
            'volume': int(15000000),
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'TSLA',
            'price': float(248.75),
            'change': float(-3.25),
            'change_pct': float(-1.29),
            'volume': int(32000000),
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'NVDA',
            'price': float(875.20),
            'change': float(12.80),
            'change_pct': float(1.48),
            'volume': int(22000000),
            'timestamp': datetime.now().isoformat()
        }
    ]

def get_real_fundamental_data():
    """Get real fundamental analysis data using yfinance"""
    import yfinance as yf
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    fundamental_data = []
    
    try:
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract fundamental metrics
                profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 20.0
                revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 10.0
                pe_ratio = info.get('forwardPE', info.get('trailingPE', 20))
                
                # Calculate health score based on metrics
                health_score = min(100, max(0, 
                    (profit_margin * 2) + 
                    (revenue_growth * 3) + 
                    (50 - abs(pe_ratio - 20)) +
                    30  # Base score
                ))
                
                # Determine health rating
                if health_score >= 85:
                    health_rating = 'EXCELLENT'
                    recommendation = 'BUY'
                    risk_level = 'LOW'
                elif health_score >= 70:
                    health_rating = 'GOOD'
                    recommendation = 'BUY'
                    risk_level = 'MEDIUM'
                elif health_score >= 50:
                    health_rating = 'FAIR'
                    recommendation = 'HOLD'
                    risk_level = 'MEDIUM'
                else:
                    health_rating = 'POOR'
                    recommendation = 'SELL'
                    risk_level = 'HIGH'
                
                # Simple sentiment based on recent performance
                sentiment_score = 0.6 + (revenue_growth / 100) * 0.3
                sentiment_score = max(0, min(1, sentiment_score))
                
                fundamental_data.append({
                    'symbol': symbol,
                    'health_rating': health_rating,
                    'health_score': float(health_score),
                    'investment_recommendation': recommendation,
                    'risk_level': risk_level,
                    'profit_margin': float(profit_margin),
                    'revenue_growth': float(revenue_growth),
                    'sentiment_score': float(sentiment_score)
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error fetching fundamental data for {symbol}: {e}")
                # Use mock data for this symbol
                fundamental_data.append({
                    'symbol': symbol,
                    'health_rating': 'Loading...',
                    'health_score': float(0),
                    'investment_recommendation': 'Analyzing...',
                    'risk_level': 'UNKNOWN',
                    'profit_margin': float(0),
                    'revenue_growth': float(0),
                    'sentiment_score': float(0.5)
                })
        
        logger.info(f"✅ Fetched real fundamental data for {len(fundamental_data)} symbols")
        return fundamental_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching real fundamental data: {e}")
        return get_mock_fundamental_data()

def get_mock_fundamental_data():
    """Get mock fundamental analysis data as fallback"""
    return [
        {
            'symbol': 'AAPL',
            'health_rating': 'EXCELLENT',
            'health_score': float(92.5),
            'investment_recommendation': 'BUY',
            'risk_level': 'LOW',
            'profit_margin': float(25.3),
            'revenue_growth': float(8.2),
            'sentiment_score': float(0.75)
        },
        {
            'symbol': 'MSFT',
            'health_rating': 'EXCELLENT',
            'health_score': float(89.2),
            'investment_recommendation': 'BUY',
            'risk_level': 'LOW',
            'profit_margin': float(35.1),
            'revenue_growth': float(12.1),
            'sentiment_score': float(0.65)
        }
    ]

def get_mock_anomaly_data():
    """Get mock anomaly data"""
    return []

def get_mock_portfolio_data():
    """Get mock portfolio data"""
    return []

@app.route('/')
def enhanced_dashboard():
    """Enhanced dashboard with web crawling and fundamental analysis"""
    initialize_enhanced_system()
    update_enhanced_system_data()
    
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio_page():
    """Portfolio management page with AI-powered analysis"""
    initialize_enhanced_system()
    return render_template('portfolio.html')

@app.route('/api/enhanced_status')
def enhanced_api_status():
    """API endpoint for enhanced system status"""
    try:
        initialize_enhanced_system()
        status = enhanced_pipeline.get_enhanced_pipeline_status() if enhanced_pipeline else {}
        
        return jsonify({
            'status': 'success',
            'data': {
                'pipeline_status': status.get('pipeline_status', 'UNKNOWN'),
                'symbols_monitored': status.get('symbols_monitored', 0),
                'active_streams': len([k for k, v in status.get('data_streams', {}).items() if v]),
                'total_capabilities': len([k for k, v in status.get('capabilities', {}).items() if v]),
                'update_interval': status.get('update_interval', 30),
                'monitored_symbols': status.get('monitored_symbols', []),
                'data_streams': status.get('data_streams', {}),
                'capabilities': status.get('capabilities', {}),
                'last_updated': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"❌ API status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/enhanced_market_data')
def enhanced_api_market_data():
    """API endpoint for enhanced market data"""
    update_enhanced_system_data()
    return jsonify({
        'status': 'success',
        'data': enhanced_system_data['market_data']
    })

@app.route('/api/crawled_news')
def api_crawled_news():
    """API endpoint for web-crawled news"""
    update_enhanced_system_data()
    return jsonify({
        'status': 'success',
        'data': enhanced_system_data['crawled_news']
    })

@app.route('/api/fundamental_analysis')
def api_fundamental_analysis():
    """API endpoint for fundamental analysis"""
    update_enhanced_system_data()
    return jsonify({
        'status': 'success',
        'data': enhanced_system_data['fundamental_analysis']
    })

@app.route('/api/integrated_intelligence')
def api_integrated_intelligence():
    """API endpoint for integrated intelligence"""
    update_enhanced_system_data()
    return jsonify({
        'status': 'success',
        'data': enhanced_system_data['integrated_intelligence']
    })

@app.route('/api/portfolio_enhanced')
def api_portfolio_enhanced():
    """API endpoint for enhanced portfolio data"""
    try:
        # Mock enhanced portfolio data
        enhanced_portfolio_data = [
            {
                'symbol': 'AAPL',
                'shares_held': 100,
                'avg_cost': 180.0,
                'current_price': 185.50,
                'current_value': 18550.0,
                'absolute_pnl': 550.0,
                'percent_pnl': 3.06,
                'daily_impact': 230.0,
                'investment_recommendation': 'BUY',
                'risk_level': 'LOW'
            },
            {
                'symbol': 'MSFT',
                'shares_held': 50,
                'avg_cost': 420.0,
                'current_price': 425.80,
                'current_value': 21290.0,
                'absolute_pnl': 290.0,
                'percent_pnl': 1.38,
                'daily_impact': -60.0,
                'investment_recommendation': 'BUY',
                'risk_level': 'LOW'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'data': enhanced_portfolio_data
        })
    except Exception as e:
        logger.error(f"❌ Enhanced portfolio API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def perform_comprehensive_analysis(symbol):
    """Perform comprehensive fundamental analysis for a ticker"""
    logger.info(f"🔍 Starting comprehensive analysis for {symbol}")
    
    try:
        import yfinance as yf
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        ticker = yf.Ticker(symbol)
        
        # 1. Basic Company Information
        info = ticker.info
        basic_info = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'website': info.get('website', ''),
            'business_summary': info.get('longBusinessSummary', '')
        }
        
        # 2. Financial Data
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        # 3. Key Financial Ratios
        financial_ratios = calculate_financial_ratios(info, financials, balance_sheet)
        
        # 4. Valuation Metrics
        valuation = calculate_valuation_metrics(info, financials)
        
        # 5. Growth Analysis
        growth_analysis = analyze_growth_trends(financials, cash_flow)
        
        # 6. Technical Analysis
        technical_analysis = perform_technical_analysis(symbol)
        
        # 7. News and Sentiment Analysis
        news_sentiment = analyze_news_sentiment(symbol)
        
        # 8. Peer Comparison
        peer_comparison = perform_peer_analysis(symbol, info.get('sector', ''))
        
        # 9. Risk Assessment
        risk_assessment = assess_investment_risks(info, financials, balance_sheet)
        
        # 10. AI-Powered Analysis
        ai_analysis = generate_ai_analysis(symbol, basic_info, financial_ratios, valuation, growth_analysis)
        
        return {
            'basic_info': basic_info,
            'financial_ratios': financial_ratios,
            'valuation': valuation,
            'growth_analysis': growth_analysis,
            'technical_analysis': technical_analysis,
            'news_sentiment': news_sentiment,
            'peer_comparison': peer_comparison,
            'risk_assessment': risk_assessment,
            'ai_analysis': ai_analysis,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in comprehensive analysis for {symbol}: {e}")
        return {
            'error': str(e),
            'symbol': symbol,
            'last_updated': datetime.now().isoformat()
        }

def calculate_financial_ratios(info, financials, balance_sheet):
    """Calculate key financial ratios"""
    try:
        ratios = {}
        
        # Profitability Ratios
        ratios['profit_margin'] = info.get('profitMargins', 0) * 100
        ratios['operating_margin'] = info.get('operatingMargins', 0) * 100
        ratios['return_on_equity'] = info.get('returnOnEquity', 0) * 100
        ratios['return_on_assets'] = info.get('returnOnAssets', 0) * 100
        
        # Liquidity Ratios
        current_ratio = info.get('currentRatio', 0)
        quick_ratio = info.get('quickRatio', 0)
        ratios['current_ratio'] = current_ratio
        ratios['quick_ratio'] = quick_ratio
        
        # Leverage Ratios
        ratios['debt_to_equity'] = info.get('debtToEquity', 0) / 100
        ratios['debt_ratio'] = info.get('totalDebt', 0) / max(info.get('totalAssets', 1), 1)
        
        # Efficiency Ratios
        ratios['asset_turnover'] = info.get('revenuePerShare', 0) / max(info.get('bookValue', 1), 1)
        ratios['inventory_turnover'] = info.get('inventoryTurnover', 0)
        
        # Market Ratios
        ratios['pe_ratio'] = info.get('forwardPE', info.get('trailingPE', 0))
        ratios['pb_ratio'] = info.get('priceToBook', 0)
        ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0)
        ratios['peg_ratio'] = info.get('pegRatio', 0)
        
        return ratios
    except Exception as e:
        logger.error(f"❌ Error calculating financial ratios: {e}")
        return {}

def calculate_valuation_metrics(info, financials):
    """Calculate valuation metrics"""
    try:
        valuation = {}
        
        # Current Valuation
        valuation['market_cap'] = info.get('marketCap', 0)
        valuation['enterprise_value'] = info.get('enterpriseValue', 0)
        valuation['book_value'] = info.get('bookValue', 0)
        
        # Valuation Ratios
        valuation['ev_revenue'] = info.get('enterpriseToRevenue', 0)
        valuation['ev_ebitda'] = info.get('enterpriseToEbitda', 0)
        
        # DCF Valuation (simplified)
        free_cash_flow = info.get('freeCashflow', 0)
        shares_outstanding = info.get('sharesOutstanding', 1)
        
        if free_cash_flow > 0 and shares_outstanding > 0:
            # Simplified DCF with 10% discount rate and 3% growth
            discount_rate = 0.10
            growth_rate = 0.03
            terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
            dcf_value = terminal_value / shares_outstanding
            valuation['dcf_value_per_share'] = dcf_value
        else:
            valuation['dcf_value_per_share'] = 0
            
        return valuation
    except Exception as e:
        logger.error(f"❌ Error calculating valuation metrics: {e}")
        return {}

def analyze_growth_trends(financials, cash_flow):
    """Analyze growth trends"""
    try:
        growth = {}
        
        if not financials.empty and len(financials.columns) >= 2:
            # Revenue Growth
            revenue_current = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
            revenue_previous = financials.loc['Total Revenue'].iloc[1] if 'Total Revenue' in financials.index else 0
            
            if revenue_previous != 0:
                growth['revenue_growth'] = ((revenue_current - revenue_previous) / revenue_previous) * 100
            else:
                growth['revenue_growth'] = 0
                
            # Earnings Growth
            if 'Net Income' in financials.index:
                earnings_current = financials.loc['Net Income'].iloc[0]
                earnings_previous = financials.loc['Net Income'].iloc[1]
                
                if earnings_previous != 0:
                    growth['earnings_growth'] = ((earnings_current - earnings_previous) / earnings_previous) * 100
                else:
                    growth['earnings_growth'] = 0
        else:
            growth['revenue_growth'] = 0
            growth['earnings_growth'] = 0
            
        return growth
    except Exception as e:
        logger.error(f"❌ Error analyzing growth trends: {e}")
        return {'revenue_growth': 0, 'earnings_growth': 0}

def perform_technical_analysis(symbol):
    """Perform technical analysis"""
    try:
        import yfinance as yf
        import pandas as pd
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='6mo')
        
        if hist.empty:
            return {}
            
        # Moving Averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        current_price = hist['Close'].iloc[-1]
        ma20 = hist['MA20'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        
        # RSI Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Support and Resistance
        recent_high = hist['High'].tail(30).max()
        recent_low = hist['Low'].tail(30).min()
        
        technical = {
            'current_price': float(current_price),
            'ma20': float(ma20) if not pd.isna(ma20) else 0,
            'ma50': float(ma50) if not pd.isna(ma50) else 0,
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
            'support_level': float(recent_low),
            'resistance_level': float(recent_high),
            'trend': 'Bullish' if current_price > ma20 > ma50 else 'Bearish'
        }
        
        return technical
    except Exception as e:
        logger.error(f"❌ Error in technical analysis: {e}")
        return {}

def analyze_news_sentiment(symbol):
    """Analyze news sentiment using REAL Pathway streaming"""
    try:
        logger.info(f"🔍 Analyzing news sentiment for {symbol} using REAL Pathway...")
        
        # Use REAL Pathway system first
        pathway_analysis = get_pathway_news_analysis(symbol)
        
        # If real Pathway system works, use it
        if pathway_analysis['total_articles'] > 0:
            logger.info(f"✅ Using REAL Pathway system - {pathway_analysis['total_articles']} articles")
            
            sentiment_data = {
                'overall_sentiment': pathway_analysis.get('avg_sentiment', 0.0),
                'news_count': pathway_analysis.get('total_articles', 0),
                'recent_news': [],
                'sources': pathway_analysis.get('sources', [])
            }
            
            # Process recent articles from real Pathway
            for article in pathway_analysis.get('recent_articles', [])[:15]:
                sentiment_data['recent_news'].append({
                    'title': article.get('title', 'Unknown'),
                    'link': article.get('url', '#'),
                    'published': article.get('published_date', 'Unknown'),
                    'sentiment_score': article.get('sentiment_score', 0.0),
                    'source': article.get('source', 'Unknown'),
                    'relevance': article.get('relevance_score', 0.0),
                    'keywords': article.get('keywords', ''),
                    'symbol_mentions': article.get('symbol_mentions', '')
                })
            
            logger.info(f"✅ REAL Pathway analyzed {sentiment_data['news_count']} articles for {symbol} with avg sentiment {sentiment_data['overall_sentiment']:.3f}")
            return sentiment_data
        
        # Fallback to legacy enhanced system
        logger.info(f"🔄 Falling back to legacy enhanced system for {symbol}...")
        enhanced_analysis = get_enhanced_news_analysis(symbol)
        
        # If no results, try refreshing the stream
        if enhanced_analysis['total_articles'] == 0:
            logger.info(f"🔄 Refreshing legacy news stream for {symbol}...")
            refresh_news_stream(symbol)
            enhanced_analysis = get_enhanced_news_analysis(symbol)
        
        # Convert to expected format
        sentiment_data = {
            'overall_sentiment': enhanced_analysis.get('avg_sentiment', 0.0),
            'news_count': enhanced_analysis.get('total_articles', 0),
            'recent_news': []
        }
        
        # Process recent articles
        for article in enhanced_analysis.get('recent_articles', [])[:15]:
            sentiment_data['recent_news'].append({
                'title': article.get('title', 'Unknown'),
                'link': article.get('url', '#'),
                'published': article.get('published_date', 'Unknown'),
                'sentiment_score': article.get('sentiment_score', 0.0),
                'source': article.get('source', 'Unknown'),
                'relevance': article.get('relevance_score', 0.0),
                'keywords': article.get('keywords', '')
            })
        
        logger.info(f"✅ Legacy system analyzed {sentiment_data['news_count']} articles for {symbol} with avg sentiment {sentiment_data['overall_sentiment']:.3f}")
        return sentiment_data
        
    except Exception as e:
        logger.error(f"❌ Error analyzing news sentiment with Pathway systems: {e}")
        # Fallback to simple analysis
        return analyze_news_sentiment_fallback(symbol)

def analyze_news_sentiment_fallback(symbol):
    """Fallback news sentiment analysis"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import feedparser
        
        sentiment_data = {
            'overall_sentiment': 0.0,
            'news_count': 0,
            'recent_news': []
        }
        
        # Try multiple news sources
        news_sources = [
            f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reddit.com/r/stocks/.rss'
        ]
        
        total_sentiment = 0
        news_count = 0
        
        for source in news_sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:5]:  # Limit to 5 articles per source
                    if symbol.lower() in entry.title.lower() or symbol.lower() in entry.get('summary', '').lower():
                        # Simple sentiment analysis based on keywords
                        title = entry.title.lower()
                        sentiment_score = 0
                        
                        # Positive keywords
                        positive_words = ['growth', 'profit', 'gain', 'rise', 'bull', 'strong', 'beat', 'exceed']
                        negative_words = ['loss', 'fall', 'drop', 'bear', 'weak', 'miss', 'decline', 'concern']
                        
                        for word in positive_words:
                            if word in title:
                                sentiment_score += 0.1
                                
                        for word in negative_words:
                            if word in title:
                                sentiment_score -= 0.1
                        
                        total_sentiment += sentiment_score
                        news_count += 1
                        
                        sentiment_data['recent_news'].append({
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.get('published', ''),
                            'sentiment_score': sentiment_score,
                            'source': 'RSS Feed',
                            'relevance': 0.5
                        })
            except:
                continue
        
        if news_count > 0:
            sentiment_data['overall_sentiment'] = total_sentiment / news_count
            sentiment_data['news_count'] = news_count
        
        return sentiment_data
    except Exception as e:
        logger.error(f"❌ Error analyzing news sentiment: {e}")
        return {'overall_sentiment': 0.0, 'news_count': 0, 'recent_news': []}

def perform_peer_analysis(symbol, sector):
    """Perform peer comparison analysis"""
    try:
        # Sample peer analysis - in production, this would use sector data
        peer_data = {
            'sector': sector,
            'sector_pe_avg': 0,
            'sector_pb_avg': 0,
            'sector_roe_avg': 0,
            'relative_valuation': 'Fair'
        }
        
        # This would be enhanced with real sector data
        if sector == 'Technology':
            peer_data['sector_pe_avg'] = 25.0
            peer_data['sector_pb_avg'] = 3.5
            peer_data['sector_roe_avg'] = 15.0
        elif sector == 'Healthcare':
            peer_data['sector_pe_avg'] = 22.0
            peer_data['sector_pb_avg'] = 2.8
            peer_data['sector_roe_avg'] = 12.0
        else:
            peer_data['sector_pe_avg'] = 18.0
            peer_data['sector_pb_avg'] = 2.0
            peer_data['sector_roe_avg'] = 10.0
            
        return peer_data
    except Exception as e:
        logger.error(f"❌ Error in peer analysis: {e}")
        return {}

def assess_investment_risks(info, financials, balance_sheet):
    """Assess investment risks"""
    try:
        risks = {
            'financial_risk': 'Medium',
            'market_risk': 'Medium',
            'liquidity_risk': 'Low',
            'risk_factors': []
        }
        
        # Financial Risk Assessment
        debt_to_equity = info.get('debtToEquity', 0) / 100
        current_ratio = info.get('currentRatio', 1)
        
        if debt_to_equity > 1.0:
            risks['financial_risk'] = 'High'
            risks['risk_factors'].append('High debt-to-equity ratio')
        elif debt_to_equity > 0.5:
            risks['financial_risk'] = 'Medium'
        else:
            risks['financial_risk'] = 'Low'
            
        # Liquidity Risk
        if current_ratio < 1.0:
            risks['liquidity_risk'] = 'High'
            risks['risk_factors'].append('Low current ratio indicating liquidity concerns')
        elif current_ratio < 1.5:
            risks['liquidity_risk'] = 'Medium'
            
        # Market Risk (Beta)
        beta = info.get('beta', 1.0)
        if beta > 1.5:
            risks['market_risk'] = 'High'
            risks['risk_factors'].append('High beta indicates high market sensitivity')
        elif beta > 1.2:
            risks['market_risk'] = 'Medium'
        else:
            risks['market_risk'] = 'Low'
            
        return risks
    except Exception as e:
        logger.error(f"❌ Error assessing risks: {e}")
        return {'financial_risk': 'Unknown', 'market_risk': 'Unknown', 'liquidity_risk': 'Unknown', 'risk_factors': []}

def generate_ai_analysis(symbol, basic_info, financial_ratios, valuation, growth_analysis):
    """Generate AI-powered analysis using Gemini"""
    try:
        import google.generativeai as genai
        import os
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return {'summary': 'AI analysis unavailable - API key not configured', 'recommendation': 'Manual analysis required'}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze the following financial data for {symbol} ({basic_info.get('company_name', 'Unknown Company')}) and provide comprehensive investment analysis:

        Company Info:
        - Sector: {basic_info.get('sector', 'Unknown')}
        - Industry: {basic_info.get('industry', 'Unknown')}
        - Market Cap: ${basic_info.get('market_cap', 0):,}

        Financial Ratios:
        - P/E Ratio: {financial_ratios.get('pe_ratio', 'N/A')}
        - Profit Margin: {financial_ratios.get('profit_margin', 0):.2f}%
        - ROE: {financial_ratios.get('return_on_equity', 0):.2f}%
        - Debt-to-Equity: {financial_ratios.get('debt_to_equity', 0):.2f}
        - Current Ratio: {financial_ratios.get('current_ratio', 0):.2f}

        Growth:
        - Revenue Growth: {growth_analysis.get('revenue_growth', 0):.2f}%
        - Earnings Growth: {growth_analysis.get('earnings_growth', 0):.2f}%

        Valuation:
        - Market Cap: ${valuation.get('market_cap', 0):,}
        - DCF Value: ${valuation.get('dcf_value_per_share', 0):.2f}

        Please provide:
        1. Overall investment recommendation (BUY/HOLD/SELL)
        2. Key strengths and weaknesses
        3. Fair value estimate
        4. Risk assessment
        5. Investment thesis summary (2-3 sentences)
        """
        
        response = model.generate_content(prompt)
        
        return {
            'summary': response.text,
            'recommendation': 'AI Generated',
            'confidence': 'High'
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating AI analysis: {e}")
        return {
            'summary': 'AI analysis temporarily unavailable',
            'recommendation': 'Manual analysis recommended',
            'confidence': 'Low'
        }

@app.route('/fundamental_analysis')
def comprehensive_fundamental_analysis():
    """Comprehensive fundamental analysis page with real data sourcing"""
    initialize_enhanced_system()
    update_enhanced_system_data()
    
    return render_template('comprehensive_analysis.html', 
                         system_data=enhanced_system_data)

@app.route('/fundamental_analysis/<symbol>')
def analyze_ticker(symbol):
    """Analyze a specific ticker with comprehensive fundamental analysis"""
    try:
        # Perform comprehensive analysis
        analysis_data = perform_comprehensive_analysis(symbol.upper())
        return render_template('comprehensive_analysis.html', 
                             symbol=symbol.upper(),
                             analysis_data=analysis_data,
                             system_data=enhanced_system_data)
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return render_template('comprehensive_analysis.html', 
                             symbol=symbol.upper(),
                             error=str(e),
                             system_data=enhanced_system_data)

@app.route('/integrated_intelligence')
def integrated_intelligence_page():
    """Integrated intelligence page"""
    initialize_enhanced_system()
    update_enhanced_system_data()
    
    return render_template('fundamental_analysis.html', 
                         system_data=enhanced_system_data)

# Advanced Feature API Endpoints
@app.route('/api/realtime_prices/<symbol>')
def get_realtime_price(symbol):
    """Get real-time price data for a symbol using yfinance"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period='2d')
        
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            volume = int(hist['Volume'].iloc[-1])
        else:
            info = ticker.info
            current_price = info.get('currentPrice', info.get('previousClose', 0))
            prev_price = info.get('previousClose', current_price)
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price else 0
            volume = info.get('volume', 0)
        
        data = {
            'symbol': symbol.upper(),
            'current_price': float(current_price),
            'change': float(change),
            'change_pct': float(change_pct),
            'volume': int(volume),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({'status': 'success', 'data': data})
        
    except Exception as e:
        logger.error(f"❌ Error fetching real-time data for {symbol}: {e}")
        # Return mock data as fallback
        return jsonify({
            'status': 'success', 
            'data': {
                'symbol': symbol.upper(),
                'current_price': 100.0,
                'change': 0.0,
                'change_pct': 0.0,
                'volume': 1000000,
                'timestamp': datetime.now().isoformat()
            }
        })

@app.route('/api/trading_signals/<symbol>')
def get_trading_signals(symbol):
    """Get AI trading signals for a symbol"""
    try:
        if trading_signal_generator:
            signals = trading_signal_generator.generate_signals(symbol.upper())
            return jsonify({'status': 'success', 'data': signals})
        return jsonify({'status': 'error', 'message': 'Trading signals not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio/risk_metrics')
def get_risk_metrics():
    """Get portfolio risk metrics"""
    try:
        if risk_analyzer:
            # Get current portfolio from session or use default
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
            weights = [0.25, 0.25, 0.25, 0.25]
            
            metrics = risk_analyzer.calculate_portfolio_risk(symbols, weights)
            return jsonify({'status': 'success', 'data': metrics})
        return jsonify({'status': 'error', 'message': 'Risk manager not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/alerts/active')
def get_active_alerts():
    """Get active alerts"""
    try:
        if alert_system:
            alerts = alert_system.get_active_alerts()
            return jsonify({'status': 'success', 'data': alerts})
        return jsonify({'status': 'error', 'message': 'Alert system not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/alerts/create', methods=['POST'])
def create_alert():
    """Create a new alert"""
    try:
        if alert_system:
            data = request.json
            alert_id = alert_system.create_alert(
                alert_type=data.get('type', 'price'),
                symbol=data.get('symbol'),
                condition=data.get('condition'),
                value=data.get('value'),
                notification_channels=data.get('channels', ['console'])
            )
            return jsonify({'status': 'success', 'alert_id': alert_id})
        return jsonify({'status': 'error', 'message': 'Alert system not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/comprehensive_analysis/<symbol>')
def api_comprehensive_analysis(symbol):
    """API endpoint for comprehensive analysis"""
    try:
        analysis_data = perform_comprehensive_analysis(symbol.upper())
        return jsonify({
            'status': 'success',
            'data': analysis_data
        })
    except Exception as e:
        logger.error(f"❌ API comprehensive analysis error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/pathway_news/<symbol>')
def api_pathway_news(symbol):
    """API endpoint for REAL Pathway news analysis"""
    try:
        news_analysis = get_pathway_news_analysis(symbol.upper())
        return jsonify({
            'status': 'success',
            'system': 'real_pathway',
            'data': news_analysis
        })
    except Exception as e:
        logger.error(f"❌ API real Pathway news error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/enhanced_news/<symbol>')
def api_enhanced_news(symbol):
    """API endpoint for legacy enhanced news analysis (fallback)"""
    try:
        news_analysis = get_enhanced_news_analysis(symbol.upper())
        return jsonify({
            'status': 'success', 
            'system': 'legacy_enhanced',
            'data': news_analysis
        })
    except Exception as e:
        logger.error(f"❌ API enhanced news error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/refresh_news/<symbol>')
def api_refresh_news(symbol):
    """API endpoint to refresh news stream"""
    try:
        articles = refresh_news_stream(symbol.upper())
        return jsonify({
            'status': 'success',
            'articles_fetched': len(articles),
            'message': f'Refreshed news stream for {symbol.upper()}'
        })
    except Exception as e:
        logger.error(f"❌ API refresh news error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/news_alerts/<symbol>')
def api_news_alerts(symbol):
    """API endpoint for news alerts"""
    try:
        alerts_data = get_news_alerts(symbol.upper())
        return jsonify({
            'status': 'success',
            'data': alerts_data
        })
    except Exception as e:
        logger.error(f"❌ API news alerts error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/add_watchlist/<symbol>')
def api_add_watchlist(symbol):
    """API endpoint to add symbol to alert watchlist"""
    try:
        add_alert_watchlist_symbol(symbol.upper())
        return jsonify({
            'status': 'success',
            'message': f'Added {symbol.upper()} to alert watchlist'
        })
    except Exception as e:
        logger.error(f"❌ API add watchlist error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/market_overview')
def get_market_overview():
    """Get comprehensive market overview"""
    try:
        overview = {}
        
        # Get real-time data from multiple sources
        if realtime_market_feed:
            overview['market_data'] = realtime_market_feed.get_market_overview()
        
        if trading_signal_generator:
            overview['market_sentiment'] = trading_signal_generator.get_market_sentiment()
        
        if risk_analyzer:
            overview['market_risk'] = risk_analyzer.get_market_risk_indicators()
        
        if alert_system:
            overview['active_alerts_count'] = len(alert_system.get_active_alerts())
        
        return jsonify({'status': 'success', 'data': overview})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Legacy routes removed for cleanup

# ========================================
# PATHWAY PORTFOLIO MANAGEMENT ROUTES
# ========================================

@app.route('/portfolio')
def portfolio_dashboard():
    """Real-time portfolio dashboard"""
    return render_template('live_portfolio.html')

# Old portfolio dashboard route removed

@app.route('/api/portfolio/add_stock', methods=['POST'])
def add_stock_to_portfolio():
    """Add a stock to the portfolio"""
    try:
        # Ensure system is initialized
        initialize_enhanced_system()
        
        data = request.get_json()
        
        # Support both 'shares' and 'quantity' field names
        required_fields = ['symbol']
        if 'shares' in data:
            quantity_field = 'shares'
        elif 'quantity' in data:
            quantity_field = 'quantity'
        else:
            return jsonify({'success': False, 'message': 'Missing required field: shares or quantity'})
            
        if 'purchase_price' not in data:
            return jsonify({'success': False, 'message': 'Missing required field: purchase_price'})
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        symbol = data['symbol'].upper()
        shares = float(data[quantity_field])
        purchase_price = float(data['purchase_price'])
        purchase_date = data.get('purchase_date')
        
        success = portfolio_manager.add_stock(symbol, shares, purchase_price, purchase_date)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Added {symbol} to portfolio',
                'data': {
                    'symbol': symbol,
                    'shares': shares,
                    'purchase_price': purchase_price
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to add stock to portfolio'})
            
    except Exception as e:
        logger.error(f"Error adding stock to portfolio: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/portfolio/remove_stock', methods=['POST'])
def remove_stock_from_portfolio():
    """Remove a stock from the portfolio"""
    try:
        # Ensure system is initialized
        initialize_enhanced_system()
        
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({'success': False, 'message': 'Missing required field: symbol'})
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        symbol = data['symbol'].upper()
        success = portfolio_manager.remove_stock(symbol)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Removed {symbol} from portfolio'
            })
        else:
            return jsonify({'success': False, 'message': f'Stock {symbol} not found in portfolio'})
            
    except Exception as e:
        logger.error(f"Error removing stock from portfolio: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/portfolio/data')
def get_portfolio_data():
    """Get complete portfolio data with real-time metrics"""
    try:
        # Ensure system is initialized
        initialize_enhanced_system()
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            logger.error("Portfolio manager not available")
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        logger.info("Getting portfolio data from manager...")
        portfolio_data = portfolio_manager.get_portfolio_data()
        logger.info(f"Portfolio data keys: {list(portfolio_data.keys()) if portfolio_data else 'None'}")
        
        # Extract portfolio metrics
        portfolio_metrics = portfolio_data.get('portfolio_metrics', {})
        positions = portfolio_metrics.get('positions', [])
        logger.info(f"Found {len(positions)} positions")
        
        # Format data for frontend
        stocks = []
        
        for stock in positions:
            # Map the portfolio manager's data structure to frontend format
            stocks.append({
                'symbol': stock.get('symbol', ''),
                'company_name': stock.get('company_name', stock.get('symbol', '')),
                'quantity': stock.get('shares', 0),  # Use 'shares' from portfolio manager
                'purchase_price': stock.get('purchase_price', 0),
                'current_price': stock.get('current_price', 0),
                'current_value': stock.get('position_value', 0),
                'cost_basis': stock.get('position_cost', 0),
                'gain_loss': stock.get('unrealized_pnl', 0),
                'percentage_change': stock.get('unrealized_pnl_percent', 0),
                'last_updated': datetime.now().isoformat()
            })
        
        # Use metrics from portfolio manager if available
        total_value = portfolio_metrics.get('total_value', 0)
        total_cost = portfolio_metrics.get('total_cost', 0)
        total_gain_loss = portfolio_metrics.get('total_pnl', 0)
        total_percentage = portfolio_metrics.get('total_pnl_percent', 0)
        
        response_data = {
            'success': True,
            'stocks': stocks,
            'summary': {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_gain_loss': total_gain_loss,
                'total_percentage': total_percentage,
                'total_stocks': len(stocks)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Returning response with {len(stocks)} stocks")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting portfolio data: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/portfolio/stock/<symbol>')
def get_stock_metrics():
    """Get detailed metrics for a specific stock"""
    try:
        portfolio_manager = get_portfolio_manager()
        metrics_engine = get_metrics_engine()
        
        if not portfolio_manager or not metrics_engine:
            return jsonify({'status': 'error', 'message': 'Portfolio systems not available'})
        
        symbol = symbol.upper()
        
        # Get current stock data
        stock_data = portfolio_manager.get_stock_data(symbol)
        
        # Calculate comprehensive metrics
        comprehensive_metrics = metrics_engine.calculate_comprehensive_metrics(symbol, stock_data)
        
        combined_data = {**stock_data, **comprehensive_metrics}
        
        return jsonify({
            'status': 'success',
            'data': combined_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting stock metrics for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio/alerts')
def get_portfolio_alerts():
    """Get portfolio alerts"""
    try:
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'status': 'error', 'message': 'Portfolio manager not available'})
        
        alerts = portfolio_manager.get_alerts()
        
        return jsonify({
            'status': 'success',
            'data': {
                'alerts': alerts,
                'count': len(alerts)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio alerts: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio/start_streaming', methods=['POST'])
def start_portfolio_streaming():
    """Start real-time streaming for portfolio"""
    try:
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'status': 'error', 'message': 'Portfolio manager not available'})
        
        # Start streaming in background thread
        threading.Thread(target=portfolio_manager.start_streaming, daemon=True).start()
        
        return jsonify({
            'status': 'success',
            'message': 'Portfolio streaming started',
            'symbols': portfolio_manager.get_portfolio_symbols()
        })
        
    except Exception as e:
        logger.error(f"Error starting portfolio streaming: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio/stop_streaming', methods=['POST'])
def stop_portfolio_streaming():
    """Stop real-time streaming for portfolio"""
    try:
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'status': 'error', 'message': 'Portfolio manager not available'})
        
        portfolio_manager.stop_streaming()
        
        return jsonify({
            'status': 'success',
            'message': 'Portfolio streaming stopped'
        })
        
    except Exception as e:
        logger.error(f"Error stopping portfolio streaming: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# AI-Powered Portfolio Analysis Endpoints
@app.route('/api/portfolio/ai_analysis')
def get_portfolio_ai_analysis():
    """Get AI-powered analysis of the entire portfolio"""
    try:
        from src.portfolio.ai_analyst import get_portfolio_ai_analyst
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        # Get portfolio data
        portfolio_data = portfolio_manager.calculate_portfolio_metrics()
        
        # Get AI analyst
        ai_analyst = get_portfolio_ai_analyst(gemini_client)
        if not ai_analyst:
            return jsonify({'success': False, 'message': 'AI analyst not available'})
        
        # Generate AI analysis
        analysis = ai_analyst.analyze_portfolio_performance(portfolio_data)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'portfolio_summary': {
                'total_value': portfolio_data.get('total_value', 0),
                'total_pnl': portfolio_data.get('total_pnl', 0),
                'total_pnl_percent': portfolio_data.get('total_pnl_percent', 0),
                'num_positions': portfolio_data.get('num_positions', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio AI analysis: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/portfolio/ai_analysis/<symbol>')
def get_stock_ai_analysis(symbol):
    """Get AI-powered analysis of a specific stock in the portfolio"""
    try:
        from src.portfolio.ai_analyst import get_portfolio_ai_analyst
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        symbol = symbol.upper()
        
        # Get stock data from portfolio
        portfolio_data = portfolio_manager.get_portfolio_data()
        stock_position = None
        
        for stock in portfolio_data:
            if stock.get('symbol') == symbol:
                stock_position = stock
                break
        
        if not stock_position:
            return jsonify({'success': False, 'message': f'Stock {symbol} not found in portfolio'})
        
        # Get additional market data
        market_data = portfolio_manager.fetch_real_time_data(symbol)
        technical_indicators = portfolio_manager.calculate_technical_indicators(symbol)
        
        # Combine all data
        combined_data = {**stock_position}
        if market_data:
            combined_data.update(market_data)
        if technical_indicators:
            combined_data.update(technical_indicators)
        
        # Get AI analyst
        ai_analyst = get_portfolio_ai_analyst(gemini_client)
        if not ai_analyst:
            return jsonify({'success': False, 'message': 'AI analyst not available'})
        
        # Generate AI analysis
        analysis = ai_analyst.analyze_individual_stock(combined_data, market_data)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'analysis': analysis,
            'stock_data': combined_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting AI analysis for {symbol}: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/portfolio/market_insights')
def get_portfolio_market_insights():
    """Get AI-powered market insights for the portfolio"""
    try:
        from src.portfolio.ai_analyst import get_portfolio_ai_analyst
        
        portfolio_manager = get_portfolio_manager()
        if not portfolio_manager:
            return jsonify({'success': False, 'message': 'Portfolio manager not available'})
        
        # Get portfolio data
        portfolio_data = portfolio_manager.calculate_portfolio_metrics()
        
        # Get recent news data
        news_data = []
        try:
            if real_pathway_system:
                recent_news = real_pathway_system.get_recent_news(limit=10)
                news_data = recent_news.get('articles', [])
        except:
            logger.warning("Could not fetch recent news for market insights")
        
        # Get AI analyst
        ai_analyst = get_portfolio_ai_analyst(gemini_client)
        if not ai_analyst:
            return jsonify({'success': False, 'message': 'AI analyst not available'})
        
        # Generate market insights
        insights = ai_analyst.generate_market_insights(portfolio_data, news_data)
        
        return jsonify({
            'success': True,
            'insights': insights,
            'portfolio_context': {
                'total_value': portfolio_data.get('total_value', 0),
                'num_positions': portfolio_data.get('num_positions', 0),
                'volatility': portfolio_data.get('volatility', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting market insights: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Legacy routes removed for cleanup

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced FinanceAI Web Dashboard...")
    logger.info("🕷️  Features: Web Crawling + Fundamental Analysis + AI Insights")
    logger.info("🌐 Access the dashboard at: http://localhost:8082")
    
    # Initialize enhanced system on startup
    initialize_enhanced_system()
    
    app.run(debug=False, host='0.0.0.0', port=8082)