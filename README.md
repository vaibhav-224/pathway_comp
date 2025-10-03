# FinanceAI - Real-time Portfolio Management & AI Analytics

A sophisticated financial technology platform that combines real-time data processing with AI-powered insights for comprehensive portfolio management and market analysis.

## 🏗️ Architecture Overview

### Core Technology Stack
- **Backend**: Python Flask web framework
- **Real-time Processing**: Pathway streaming engine
- **AI/ML**: Google Gemini API, OpenAI API, scikit-learn
- **Data Sources**: yfinance, RSS feeds, web scraping
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap, Chart.js

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Flask Backend   │    │  Pathway Engine │
│   (Dashboard)   │◄──►│  (enhanced_app)  │◄──►│  (Streaming)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                ┌──────▼──────┐  ┌──────▼──────┐
                │ AI Analysis │  │ Market Data │
                │ (Gemini/    │  │ (yfinance/  │
                │  OpenAI)    │  │  RSS Feeds) │
                └─────────────┘  └─────────────┘
```

## 🚀 Key Features

### Real-time Portfolio Management
- Live stock price tracking with yfinance integration
- Real-time P&L calculations and performance metrics
- Technical indicator computation (RSI, MACD, Bollinger Bands)
- Portfolio optimization using Modern Portfolio Theory

### AI-Powered Analytics
- **Sentiment Analysis**: Real-time news sentiment scoring
- **Fundamental Analysis**: Comprehensive financial health assessment
- **Trading Signals**: ML-based buy/sell recommendations
- **Risk Assessment**: Multi-dimensional risk analysis

### News & Market Intelligence
- Real-time RSS feed processing from 8+ sources
- Automated news sentiment analysis
- Smart alerting system for market events
- Keyword extraction and symbol detection

### Advanced Analytics
- **Technical Analysis**: 20+ indicators and pattern recognition
- **Fundamental Analysis**: P/E, ROE, debt ratios, growth metrics
- **Risk Management**: VaR, Sharpe ratio, portfolio optimization
- **AI Insights**: LLM-powered explanations and recommendations

## 📊 Pathway Integration

The system leverages Pathway for real-time data processing:

- **Streaming UDFs**: Real-time data transformation and enrichment
- **Table Operations**: Joins, aggregations, and schema management
- **Data Pipelines**: End-to-end streaming from sources to insights
- **Schema Evolution**: Flexible data structures for different data types

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/vaibhav-224/pathway_comp
cd pathbhay
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with:
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

4. **Run the application**
```bash
python enhanced_app.py
```

5. **Access the dashboard**
Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
pathbhay/
├── enhanced_app.py              # Main Flask application
├── src/
│   ├── pathway_news/            # Real-time news processing
│   ├── portfolio/               # Portfolio management system
│   ├── ai_insights/             # AI analysis components
│   ├── market_data/             # Market data feeds
│   └── alerts/                  # Alert and notification system
├── templates/                   # Web interface templates
└── requirements.txt             # Python dependencies
```

## 🎯 Core Components

### 1. Enhanced AI Pipeline
- Orchestrates data ingestion, anomaly detection, and AI insights
- Integrates multiple analytical modules within Pathway streaming context
- Generates comprehensive intelligence summaries

### 2. Real-time News System
- Processes RSS feeds from major financial news sources
- Applies sentiment analysis and relevance scoring
- Extracts keywords and identifies stock symbol mentions

### 3. Portfolio Manager
- Real-time portfolio tracking with live market data
- Advanced risk metrics and performance calculations
- AI-powered portfolio analysis and recommendations

### 4. Alert System
- Intelligent news monitoring and alerting
- Price-based alerts and technical indicator triggers
- Customizable watchlists and notification preferences

## 🔧 API Endpoints

- `GET /` - Main dashboard
- `GET /portfolio` - Portfolio management interface
- `GET /fundamental_analysis` - Stock analysis tools
- `POST /api/portfolio/add_stock` - Add stocks to portfolio
- `GET /api/crawled_news` - Real-time news data
- `GET /api/fundamental_analysis` - AI-powered analysis

## 🚀 Performance Features

- **Real-time Updates**: 30-second data refresh cycles
- **Streaming Control**: Start/stop live data feeds
- **Fallback Systems**: Graceful degradation when services unavailable
- **Responsive Design**: Mobile-friendly interface
- **Auto-refresh**: Continuous data updates without page reload

## 📈 Use Cases

- **Individual Investors**: Portfolio tracking and analysis
- **Financial Analysts**: Market research and sentiment analysis
- **Traders**: Real-time signals and technical analysis
- **Researchers**: Financial data analysis and backtesting

## 🔒 Security & Reliability

- Environment variable management for API keys
- Error handling with fallback mechanisms
- Input validation and sanitization
- Graceful service degradation

---

**FinanceAI** - Empowering financial decisions with real-time data and AI insights.
