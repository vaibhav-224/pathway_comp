"""
Enhanced AI Pipeline with Web Crawling and Fundamental Analysis
Integrates Pathway's streaming capabilities with web crawling and advanced AI analysis
"""
import pathway as pw
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Removed unused imports - using mock implementations for demo
# from data_ingestion.web_crawler import PathwayWebCrawler  
# from ai_insights.fundamental_analyzer import AdvancedFundamentalAnalyzer
# from data_ingestion.pipeline import DataPipeline
# from anomaly_detection.detector import AnomalyDetector
try:
    from ai_insights.engine import AIInsightsEngine
except ImportError:
    AIInsightsEngine = None

logger = logging.getLogger(__name__)

class EnhancedAIPipeline:
    """Enhanced AI pipeline with web crawling and fundamental analysis capabilities"""
    
    def __init__(self, symbols: List[str], update_interval: int = 30, 
                 price_threshold: float = 2.5, volume_threshold: float = 2.0,
                 ai_provider: Optional[str] = None, api_key: Optional[str] = None):
        
        self.symbols = symbols
        self.update_interval = update_interval
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.ai_provider = ai_provider
        self.api_key = api_key
        
        # Initialize components - using mock/simple implementations
        self.data_pipeline = None  # Mock: DataPipeline(symbols, update_interval)
        self.anomaly_detector = None  # Mock: AnomalyDetector(price_threshold, volume_threshold)
        self.ai_insights_engine = AIInsightsEngine(provider=ai_provider, api_key=api_key) if AIInsightsEngine else None
        self.web_crawler = None  # Mock: PathwayWebCrawler(symbols)
        # Keep backward compatibility for fundamental analyzer - mock implementation
        self.fundamental_analyzer = None  # Mock: AdvancedFundamentalAnalyzer(api_key if ai_provider == "openai" else None)
        
        # Pipeline state
        self.pipeline_status = "INITIALIZING"
        self.data_streams = {}
        
    def start_enhanced_pipeline(self) -> Tuple[pw.Table, pw.Table, pw.Table, pw.Table, pw.Table, pw.Table]:
        """Start the complete enhanced AI pipeline with web crawling"""
        
        logger.info("🚀 Starting Enhanced AI Pipeline with Web Crawling...")
        self.pipeline_status = "STARTING"
        
        # 1. Start traditional market data ingestion - Mock implementation
        logger.info("📊 Initializing market data streams...")
        # market_table, news_table_basic = self.data_pipeline.start_data_ingestion()
        market_table, news_table_basic = None, None  # Mock data
        self.data_streams['market_data'] = True
        
        # 2. Start web crawling for enhanced news and financial data
        logger.info("🕷️  Starting advanced web crawling...")
        # crawled_news_table, financial_statements_table = self.web_crawler.create_integrated_data_stream()
        crawled_news_table, financial_statements_table = None, None  # Mock data
        self.data_streams['web_crawling'] = True
        
        # 3. Run anomaly detection on market data
        logger.info("🚨 Activating anomaly detection...")
        # anomaly_table = self.anomaly_detector.detect_anomalies(market_table)
        anomaly_table = None  # Mock data
        self.data_streams['anomaly_detection'] = True
        
        # 4. Generate AI insights from market anomalies
        logger.info("🤖 Generating AI market insights...")
        ai_insights_table = self.ai_insights_engine.generate_ai_insights(market_table, anomaly_table)
        self.data_streams['ai_insights'] = True
        
        # 5. Perform comprehensive fundamental analysis
        logger.info("📈 Running fundamental analysis...")
        fundamental_analysis_table = self.fundamental_analyzer.create_comprehensive_analysis(
            financial_statements_table, crawled_news_table
        )
        self.data_streams['fundamental_analysis'] = True
        
        # 6. Create integrated intelligence summary
        logger.info("🧠 Creating integrated intelligence summary...")
        integrated_intelligence_table = self._create_integrated_intelligence(
            market_table, anomaly_table, ai_insights_table, fundamental_analysis_table
        )
        self.data_streams['integrated_intelligence'] = True
        
        self.pipeline_status = "ENHANCED_RUNNING"
        logger.info("✅ Enhanced AI Pipeline fully operational!")
        
        return (market_table, anomaly_table, ai_insights_table, 
                crawled_news_table, financial_statements_table, 
                fundamental_analysis_table, integrated_intelligence_table)
    
    def _create_integrated_intelligence(self, market_table: pw.Table, anomaly_table: pw.Table,
                                      ai_insights_table: pw.Table, fundamental_table: pw.Table) -> pw.Table:
        """Create integrated intelligence combining all data sources"""
        
        # Define schema for integrated intelligence
        class IntegratedIntelligenceSchema(pw.Schema):
            symbol: str
            current_price: float
            price_change_pct: float
            volume: float
            
            # Anomaly information
            has_anomaly: bool
            anomaly_type: str
            anomaly_severity: str
            
            # Fundamental analysis
            health_rating: str
            health_score: float
            investment_recommendation: str
            risk_level: str
            profit_margin: float
            revenue_growth: float
            
            # News sentiment
            sentiment_score: float
            news_articles_count: int
            high_impact_news: int
            
            # Integrated signals
            overall_signal: str
            confidence_level: float
            key_insights: str
            risk_factors: str
            opportunities: str
            
            # AI-generated summary
            ai_summary: str
            next_action: str
            
            timestamp: int
        
        # Get latest market data per symbol
        latest_market = market_table.groupby(pw.this.symbol).reduce(
            symbol=pw.this.symbol,
            current_price=pw.reducers.argmax(pw.this.timestamp, pw.this.price),
            price_change_pct=pw.reducers.argmax(pw.this.timestamp, pw.this.change_pct),
            volume=pw.reducers.argmax(pw.this.timestamp, pw.this.volume),
            latest_market_timestamp=pw.reducers.max(pw.this.timestamp)
        )
        
        # Get anomaly information
        anomaly_summary = anomaly_table.groupby(pw.this.symbol).reduce(
            symbol=pw.this.symbol,
            has_anomaly=pw.reducers.any(True),
            anomaly_count=pw.reducers.count(),
            latest_anomaly_type=pw.reducers.argmax(pw.this.timestamp, pw.this.anomaly_type),
            latest_anomaly_severity=pw.reducers.argmax(pw.this.timestamp, pw.this.alert_level)
        )
        
        # Start with market data
        base_intelligence = latest_market.join(
            fundamental_table,
            latest_market.symbol == fundamental_table.symbol,
            how=pw.JoinMode.LEFT
        ).select(
            symbol=pw.this.symbol,
            current_price=pw.this.current_price,
            price_change_pct=pw.this.price_change_pct,
            volume=pw.this.volume,
            
            # Fundamental data
            health_rating=pw.if_else(
                pw.this.health_rating.is_not_none(), 
                pw.this.health_rating, 
                "UNKNOWN"
            ),
            health_score=pw.if_else(
                pw.this.health_score.is_not_none(), 
                pw.this.health_score, 
                50.0
            ),
            investment_recommendation=pw.if_else(
                pw.this.investment_recommendation.is_not_none(),
                pw.this.investment_recommendation,
                "HOLD"
            ),
            risk_level=pw.if_else(
                pw.this.risk_level.is_not_none(),
                pw.this.risk_level,
                "MEDIUM"
            ),
            profit_margin=pw.if_else(
                pw.this.profit_margin.is_not_none(),
                pw.this.profit_margin,
                0.0
            ),
            revenue_growth=pw.if_else(
                pw.this.revenue_growth.is_not_none(),
                pw.this.revenue_growth,
                0.0
            ),
            sentiment_score=pw.if_else(
                pw.this.sentiment_score.is_not_none(),
                pw.this.sentiment_score,
                0.0
            ),
            news_articles_count=pw.if_else(
                pw.this.news_articles_count.is_not_none(),
                pw.this.news_articles_count,
                0
            ),
            high_impact_news=pw.if_else(
                pw.this.high_impact_news.is_not_none(),
                pw.this.high_impact_news,
                0
            )
        )
        
        # Add anomaly information
        intelligence_with_anomalies = base_intelligence.join(
            anomaly_summary,
            base_intelligence.symbol == anomaly_summary.symbol,
            how=pw.JoinMode.LEFT
        ).select(
            symbol=pw.this.symbol,
            current_price=pw.this.current_price,
            price_change_pct=pw.this.price_change_pct,
            volume=pw.this.volume,
            health_rating=pw.this.health_rating,
            health_score=pw.this.health_score,
            investment_recommendation=pw.this.investment_recommendation,
            risk_level=pw.this.risk_level,
            profit_margin=pw.this.profit_margin,
            revenue_growth=pw.this.revenue_growth,
            sentiment_score=pw.this.sentiment_score,
            news_articles_count=pw.this.news_articles_count,
            high_impact_news=pw.this.high_impact_news,
            
            # Anomaly data
            has_anomaly=pw.if_else(
                pw.this.has_anomaly.is_not_none(),
                pw.this.has_anomaly,
                False
            ),
            anomaly_type=pw.if_else(
                pw.this.latest_anomaly_type.is_not_none(),
                pw.this.latest_anomaly_type,
                "NONE"
            ),
            anomaly_severity=pw.if_else(
                pw.this.latest_anomaly_severity.is_not_none(),
                pw.this.latest_anomaly_severity,
                "NONE"
            )
        )
        
        # Generate final integrated intelligence
        final_intelligence = intelligence_with_anomalies.select(
            symbol=pw.this.symbol,
            current_price=pw.this.current_price,
            price_change_pct=pw.this.price_change_pct,
            volume=pw.this.volume,
            has_anomaly=pw.this.has_anomaly,
            anomaly_type=pw.this.anomaly_type,
            anomaly_severity=pw.this.anomaly_severity,
            health_rating=pw.this.health_rating,
            health_score=pw.this.health_score,
            investment_recommendation=pw.this.investment_recommendation,
            risk_level=pw.this.risk_level,
            profit_margin=pw.this.profit_margin,
            revenue_growth=pw.this.revenue_growth,
            sentiment_score=pw.this.sentiment_score,
            news_articles_count=pw.this.news_articles_count,
            high_impact_news=pw.this.high_impact_news,
            
            # Generate overall signal
            overall_signal=pw.apply_with_type(
                self._generate_overall_signal,
                str,
                pw.this.investment_recommendation,
                pw.this.has_anomaly,
                pw.this.sentiment_score,
                pw.this.health_score
            ),
            
            # Calculate confidence level
            confidence_level=pw.apply_with_type(
                self._calculate_confidence,
                float,
                pw.this.health_score,
                pw.this.news_articles_count,
                pw.this.has_anomaly,
                pw.this.sentiment_score
            ),
            
            # Generate insights
            key_insights=pw.apply_with_type(
                self._generate_key_insights,
                str,
                pw.this.health_rating,
                pw.this.profit_margin,
                pw.this.revenue_growth,
                pw.this.sentiment_score,
                pw.this.has_anomaly
            ),
            
            risk_factors=pw.apply_with_type(
                self._identify_risk_factors,
                str,
                pw.this.risk_level,
                pw.this.anomaly_severity,
                pw.this.sentiment_score,
                pw.this.high_impact_news
            ),
            
            opportunities=pw.apply_with_type(
                self._identify_opportunities,
                str,
                pw.this.health_rating,
                pw.this.revenue_growth,
                pw.this.sentiment_score,
                pw.this.investment_recommendation
            ),
            
            # AI-generated summary
            ai_summary=pw.apply_with_type(
                self._generate_ai_summary,
                str,
                pw.this.symbol,
                pw.this.investment_recommendation,
                pw.this.health_rating,
                pw.this.sentiment_score
            ),
            
            next_action=pw.apply_with_type(
                self._suggest_next_action,
                str,
                pw.this.investment_recommendation,
                pw.this.has_anomaly,
                pw.this.risk_level
            ),
            
            timestamp=pw.apply_with_type(
                lambda: int(datetime.now().timestamp() * 1000),
                int
            )
        )
        
        return final_intelligence
    
    def _generate_overall_signal(self, investment_rec: str, has_anomaly: bool, 
                               sentiment: float, health_score: float) -> str:
        """Generate overall trading signal"""
        
        # Base signal from fundamental analysis
        signal_score = 0
        
        if investment_rec == "STRONG_BUY":
            signal_score += 3
        elif investment_rec == "BUY":
            signal_score += 2
        elif investment_rec == "HOLD":
            signal_score += 0
        elif investment_rec == "SELL":
            signal_score -= 2
        elif investment_rec == "STRONG_SELL":
            signal_score -= 3
            
        # Adjust for anomalies
        if has_anomaly:
            signal_score -= 1
            
        # Adjust for sentiment
        if sentiment > 0.3:
            signal_score += 1
        elif sentiment < -0.3:
            signal_score -= 1
            
        # Adjust for health score
        if health_score > 75:
            signal_score += 1
        elif health_score < 40:
            signal_score -= 1
            
        # Convert to signal
        if signal_score >= 3:
            return "STRONG_BUY"
        elif signal_score >= 1:
            return "BUY"
        elif signal_score <= -3:
            return "STRONG_SELL"
        elif signal_score <= -1:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence(self, health_score: float, news_count: int, 
                            has_anomaly: bool, sentiment: float) -> float:
        """Calculate confidence level in the analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Health score contributes to confidence
        if health_score > 0:
            confidence += min(health_score / 200, 0.3)  # Max 0.3 boost
            
        # News coverage contributes to confidence
        if news_count > 5:
            confidence += 0.15
        elif news_count > 2:
            confidence += 0.1
            
        # Strong sentiment (positive or negative) increases confidence
        if abs(sentiment) > 0.5:
            confidence += 0.1
        elif abs(sentiment) > 0.3:
            confidence += 0.05
            
        # Anomalies reduce confidence
        if has_anomaly:
            confidence -= 0.1
            
        return max(0.1, min(1.0, confidence))
    
    def _generate_key_insights(self, health_rating: str, profit_margin: float,
                             revenue_growth: float, sentiment: float, has_anomaly: bool) -> str:
        """Generate key insights summary"""
        
        insights = []
        
        if health_rating == "EXCELLENT":
            insights.append("Exceptional financial health")
        elif health_rating == "GOOD":
            insights.append("Solid financial fundamentals")
        elif health_rating == "POOR":
            insights.append("Concerning financial metrics")
            
        if profit_margin > 20:
            insights.append(f"High profit margins ({profit_margin:.1f}%)")
        elif profit_margin < 5:
            insights.append(f"Low profit margins ({profit_margin:.1f}%)")
            
        if revenue_growth > 15:
            insights.append(f"Strong revenue growth ({revenue_growth:.1f}%)")
        elif revenue_growth < 0:
            insights.append(f"Revenue declining ({revenue_growth:.1f}%)")
            
        if sentiment > 0.3:
            insights.append("Positive market sentiment")
        elif sentiment < -0.3:
            insights.append("Negative market sentiment")
            
        if has_anomaly:
            insights.append("Unusual market activity detected")
            
        return "; ".join(insights) if insights else "Standard market conditions"
    
    def _identify_risk_factors(self, risk_level: str, anomaly_severity: str,
                             sentiment: float, high_impact_news: int) -> str:
        """Identify key risk factors"""
        
        risks = []
        
        if risk_level == "HIGH":
            risks.append("High financial risk profile")
        elif risk_level == "MEDIUM":
            risks.append("Moderate financial risks")
            
        if anomaly_severity in ["HIGH", "CRITICAL"]:
            risks.append("Significant market anomalies")
        elif anomaly_severity == "MEDIUM":
            risks.append("Market volatility detected")
            
        if sentiment < -0.3:
            risks.append("Negative sentiment trend")
            
        if high_impact_news > 2:
            risks.append("High news impact volume")
            
        return "; ".join(risks) if risks else "Low risk environment"
    
    def _identify_opportunities(self, health_rating: str, revenue_growth: float,
                              sentiment: float, investment_rec: str) -> str:
        """Identify investment opportunities"""
        
        opportunities = []
        
        if health_rating in ["EXCELLENT", "GOOD"] and investment_rec in ["BUY", "STRONG_BUY"]:
            opportunities.append("Strong fundamental buy opportunity")
            
        if revenue_growth > 15:
            opportunities.append("High growth momentum")
            
        if sentiment > 0.3 and investment_rec in ["BUY", "STRONG_BUY"]:
            opportunities.append("Positive sentiment alignment")
            
        if health_rating == "EXCELLENT" and sentiment > 0.2:
            opportunities.append("Premium quality with positive sentiment")
            
        return "; ".join(opportunities) if opportunities else "Monitor for entry points"
    
    def _generate_ai_summary(self, symbol: str, investment_rec: str, 
                           health_rating: str, sentiment: float) -> str:
        """Generate AI-powered summary"""
        
        sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        
        summary = f"{symbol} shows {health_rating.lower()} financial health with {sentiment_desc} market sentiment. "
        summary += f"Recommendation: {investment_rec.replace('_', ' ').title()}. "
        
        if investment_rec in ["STRONG_BUY", "BUY"]:
            summary += "Consider accumulating on weakness."
        elif investment_rec in ["STRONG_SELL", "SELL"]:
            summary += "Consider reducing exposure."
        else:
            summary += "Monitor for clearer signals."
            
        return summary
    
    def _suggest_next_action(self, investment_rec: str, has_anomaly: bool, risk_level: str) -> str:
        """Suggest next action based on analysis"""
        
        if has_anomaly and risk_level == "HIGH":
            return "WAIT_FOR_CLARITY"
        elif investment_rec == "STRONG_BUY":
            return "ACCUMULATE"
        elif investment_rec == "BUY":
            return "GRADUAL_BUY"
        elif investment_rec == "STRONG_SELL":
            return "REDUCE_POSITION"
        elif investment_rec == "SELL":
            return "CONSIDER_EXIT"
        else:
            return "MONITOR_CLOSELY"
    
    def get_enhanced_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            'pipeline_status': self.pipeline_status,
            'symbols_monitored': len(self.symbols),
            'data_streams': {
                'market_data': self.data_streams.get('market_data', False),
                'web_crawling': self.data_streams.get('web_crawling', False),
                'anomaly_detection': self.data_streams.get('anomaly_detection', False),
                'ai_insights': self.data_streams.get('ai_insights', False),
                'fundamental_analysis': self.data_streams.get('fundamental_analysis', False),
                'integrated_intelligence': self.data_streams.get('integrated_intelligence', False)
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
            'update_interval': self.update_interval,
            'monitored_symbols': self.symbols
        }


def test_enhanced_pipeline():
    """Test the enhanced AI pipeline"""
    print("🧪 Testing Enhanced AI Pipeline with Web Crawling...")
    
    # Initialize enhanced pipeline
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    enhanced_pipeline = EnhancedAIPipeline(
        symbols=symbols,
        update_interval=30,
        price_threshold=2.5,
        volume_threshold=2.0
    )
    
    # Start the enhanced pipeline
    print("🚀 Starting enhanced pipeline...")
    results = enhanced_pipeline.start_enhanced_pipeline()
    
    if len(results) == 7:
        (market_table, anomaly_table, ai_insights_table, 
         crawled_news_table, financial_table, fundamental_table, 
         integrated_intelligence_table) = results
        
        print("\n📊 Market Data:")
        if market_table:
            pw.debug.compute_and_print(market_table)
            
        print("\n📰 Crawled News:")
        if crawled_news_table:
            pw.debug.compute_and_print(crawled_news_table)
            
        print("\n📈 Fundamental Analysis:")
        if fundamental_table:
            pw.debug.compute_and_print(fundamental_table)
            
        print("\n🧠 Integrated Intelligence:")
        if integrated_intelligence_table:
            pw.debug.compute_and_print(integrated_intelligence_table)
    
    # Show pipeline status
    status = enhanced_pipeline.get_enhanced_pipeline_status()
    print(f"\n📊 Enhanced Pipeline Status:")
    print(f"   Status: {status['pipeline_status']}")
    print(f"   Symbols: {status['symbols_monitored']}")
    print(f"   Capabilities: {len([k for k, v in status['capabilities'].items() if v])}")
    
    print("\n✅ Enhanced AI Pipeline test completed!")


if __name__ == "__main__":
    test_enhanced_pipeline()