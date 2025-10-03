"""
Advanced AI-Powered Fundamental Analysis Engine
Leverages Pathway's streaming capabilities with crawled news and financial data for comprehensive analysis
"""
import pathway as pw
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics"""
    pe_ratio: float
    pb_ratio: float
    debt_to_equity: float
    roe: float
    roa: float
    profit_margin: float
    revenue_growth: float
    eps_growth: float
    current_ratio: float
    quick_ratio: float
    asset_turnover: float
    inventory_turnover: float

@dataclass
class NewsImpact:
    """News impact analysis"""
    sentiment_score: float
    relevance_score: float
    impact_magnitude: float
    category_weights: Dict[str, float]
    trend_direction: str
    confidence_level: float

class AdvancedFundamentalAnalyzer:
    """Advanced fundamental analysis using Pathway's streaming capabilities"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.use_ai = openai_api_key is not None
        
        # Define category weights for news impact
        self.news_category_weights = {
            'earnings': 0.9,
            'regulatory': 0.8,
            'corporate_action': 0.85,
            'product_news': 0.7,
            'management': 0.75,
            'general': 0.4
        }
        
    def analyze_financial_health(self, financial_table: pw.Table) -> pw.Table:
        """Comprehensive financial health analysis using Pathway operations"""
        
        # Calculate advanced financial ratios
        financial_analysis = financial_table.select(
            symbol=pw.this.symbol,
            filing_type=pw.this.filing_type,
            filing_date=pw.this.filing_date,
            
            # Profitability ratios
            profit_margin=pw.if_else(
                pw.this.revenue > 0,
                (pw.this.net_income / pw.this.revenue) * 100,
                0.0
            ),
            
            roe=pw.this.roe,  # Already calculated
            roa=pw.this.roa,  # Already calculated
            
            # Liquidity ratios (simplified calculation)
            current_ratio=pw.if_else(
                pw.this.total_debt > 0,
                pw.this.cash_and_equivalents / (pw.this.total_debt * 0.3),  # Approximate current liabilities
                10.0  # High liquidity if no debt
            ),
            
            # Leverage ratios
            debt_to_equity=pw.this.debt_to_equity,
            debt_to_assets=pw.if_else(
                pw.this.total_assets > 0,
                pw.this.total_debt / pw.this.total_assets,
                0.0
            ),
            
            # Efficiency ratios
            asset_turnover=pw.if_else(
                pw.this.total_assets > 0,
                pw.this.revenue / pw.this.total_assets,
                0.0
            ),
            
            # Valuation metrics
            revenue_growth=pw.this.revenue_growth,
            eps=pw.this.eps,
            
            # Market cap approximation (price * shares outstanding would be added from market data)
            book_value_per_share=pw.if_else(
                pw.this.shares_outstanding > 0,
                (pw.this.total_assets - pw.this.total_debt) / pw.this.shares_outstanding,
                0.0
            ),
            
            timestamp=pw.this.timestamp
        )
        
        # Add financial health scoring
        health_scored = financial_analysis.select(
            symbol=pw.this.symbol,
            filing_type=pw.this.filing_type,
            filing_date=pw.this.filing_date,
            profit_margin=pw.this.profit_margin,
            roe=pw.this.roe,
            roa=pw.this.roa,
            current_ratio=pw.this.current_ratio,
            debt_to_equity=pw.this.debt_to_equity,
            debt_to_assets=pw.this.debt_to_assets,
            asset_turnover=pw.this.asset_turnover,
            revenue_growth=pw.this.revenue_growth,
            eps=pw.this.eps,
            book_value_per_share=pw.this.book_value_per_share,
            
            # Financial health score (0-100)
            profitability_score=pw.if_else(
                pw.this.profit_margin > 20, 25,
                pw.if_else(
                    pw.this.profit_margin > 10, 20,
                    pw.if_else(
                        pw.this.profit_margin > 5, 15,
                        pw.if_else(pw.this.profit_margin > 0, 10, 0)
                    )
                )
            ),
            
            liquidity_score=pw.if_else(
                pw.this.current_ratio > 2.0, 20,
                pw.if_else(
                    pw.this.current_ratio > 1.5, 15,
                    pw.if_else(
                        pw.this.current_ratio > 1.0, 10,
                        5
                    )
                )
            ),
            
            leverage_score=pw.if_else(
                pw.this.debt_to_equity < 0.3, 25,
                pw.if_else(
                    pw.this.debt_to_equity < 0.5, 20,
                    pw.if_else(
                        pw.this.debt_to_equity < 1.0, 15,
                        pw.if_else(pw.this.debt_to_equity < 2.0, 10, 5)
                    )
                )
            ),
            
            growth_score=pw.if_else(
                pw.this.revenue_growth > 20, 20,
                pw.if_else(
                    pw.this.revenue_growth > 10, 15,
                    pw.if_else(
                        pw.this.revenue_growth > 5, 10,
                        pw.if_else(pw.this.revenue_growth > 0, 5, 0)
                    )
                )
            ),
            
            efficiency_score=pw.if_else(
                pw.this.asset_turnover > 1.0, 10,
                pw.if_else(
                    pw.this.asset_turnover > 0.5, 8,
                    pw.if_else(pw.this.asset_turnover > 0.2, 5, 2)
                )
            ),
            
            timestamp=pw.this.timestamp
        )
        
        # Calculate overall financial health score
        final_health_analysis = health_scored.select(
            symbol=pw.this.symbol,
            filing_type=pw.this.filing_type,
            filing_date=pw.this.filing_date,
            profit_margin=pw.this.profit_margin,
            roe=pw.this.roe,
            roa=pw.this.roa,
            current_ratio=pw.this.current_ratio,
            debt_to_equity=pw.this.debt_to_equity,
            revenue_growth=pw.this.revenue_growth,
            eps=pw.this.eps,
            
            overall_health_score=pw.this.profitability_score + pw.this.liquidity_score + 
                               pw.this.leverage_score + pw.this.growth_score + pw.this.efficiency_score,
            
            health_rating=pw.if_else(
                pw.this.profitability_score + pw.this.liquidity_score + pw.this.leverage_score + 
                pw.this.growth_score + pw.this.efficiency_score > 80,
                "EXCELLENT",
                pw.if_else(
                    pw.this.profitability_score + pw.this.liquidity_score + pw.this.leverage_score + 
                    pw.this.growth_score + pw.this.efficiency_score > 60,
                    "GOOD",
                    pw.if_else(
                        pw.this.profitability_score + pw.this.liquidity_score + pw.this.leverage_score + 
                        pw.this.growth_score + pw.this.efficiency_score > 40,
                        "FAIR",
                        "POOR"
                    )
                )
            ),
            
            key_strengths=pw.apply_with_type(
                self._identify_strengths,
                str,
                pw.this.profit_margin, pw.this.roe, pw.this.current_ratio, 
                pw.this.debt_to_equity, pw.this.revenue_growth
            ),
            
            key_concerns=pw.apply_with_type(
                self._identify_concerns,
                str,
                pw.this.profit_margin, pw.this.roe, pw.this.current_ratio, 
                pw.this.debt_to_equity, pw.this.revenue_growth
            ),
            
            timestamp=pw.this.timestamp
        )
        
        return final_health_analysis
    
    def analyze_news_sentiment_impact(self, news_table: pw.Table) -> pw.Table:
        """Analyze news sentiment and market impact using Pathway operations"""
        
        # Calculate weighted sentiment impact
        news_impact_analysis = news_table.select(
            symbol=pw.this.symbol,
            source=pw.this.source,
            title=pw.this.title,
            category=pw.this.category,
            sentiment_score=pw.this.sentiment_score,
            relevance_score=pw.this.relevance_score,
            published_date=pw.this.published_date,
            
            # Calculate category weight
            category_weight=pw.if_else(
                pw.this.category == "earnings", 0.9,
                pw.if_else(
                    pw.this.category == "regulatory", 0.8,
                    pw.if_else(
                        pw.this.category == "corporate_action", 0.85,
                        pw.if_else(
                            pw.this.category == "product_news", 0.7,
                            pw.if_else(
                                pw.this.category == "management", 0.75,
                                0.4  # general
                            )
                        )
                    )
                )
            ),
            
            # Calculate weighted impact score
            weighted_impact=pw.this.sentiment_score * pw.this.relevance_score * 
                          pw.if_else(
                              pw.this.category == "earnings", 0.9,
                              pw.if_else(
                                  pw.this.category == "regulatory", 0.8,
                                  pw.if_else(
                                      pw.this.category == "corporate_action", 0.85,
                                      pw.if_else(
                                          pw.this.category == "product_news", 0.7,
                                          pw.if_else(
                                              pw.this.category == "management", 0.75,
                                              0.4  # general
                                          )
                                      )
                                  )
                              )
                          ),
            
            # Impact magnitude classification
            impact_magnitude=pw.if_else(
                pw.apply_with_type(
                    lambda sent, rel: abs(sent * rel),
                    float,
                    pw.this.sentiment_score, pw.this.relevance_score
                ) > 0.7,
                "HIGH",
                pw.if_else(
                    pw.apply_with_type(
                        lambda sent, rel: abs(sent * rel),
                        float,
                        pw.this.sentiment_score, pw.this.relevance_score
                    ) > 0.4,
                    "MEDIUM",
                    "LOW"
                )
            ),
            
            # Trend direction
            trend_direction=pw.if_else(
                pw.this.sentiment_score > 0.2, "POSITIVE",
                pw.if_else(
                    pw.this.sentiment_score < -0.2, "NEGATIVE",
                    "NEUTRAL"
                )
            ),
            
            timestamp=pw.this.timestamp
        )
        
        return news_impact_analysis
    
    def create_comprehensive_analysis(self, financial_table: pw.Table, news_table: pw.Table) -> pw.Table:
        """Create comprehensive fundamental analysis combining financial and news data"""
        
        # Analyze financial health
        financial_analysis = self.analyze_financial_health(financial_table)
        
        # Analyze news sentiment
        news_analysis = self.analyze_news_sentiment_impact(news_table)
        
        # Aggregate news sentiment by symbol
        news_aggregated = news_analysis.groupby(pw.this.symbol).reduce(
            symbol=pw.this.symbol,
            total_articles=pw.reducers.count(),
            avg_sentiment=pw.reducers.avg(pw.this.sentiment_score),
            avg_relevance=pw.reducers.avg(pw.this.relevance_score),
            avg_weighted_impact=pw.reducers.avg(pw.this.weighted_impact),
            high_impact_count=pw.reducers.sum(
                pw.if_else(pw.this.impact_magnitude == "HIGH", 1, 0)
            ),
            positive_news_count=pw.reducers.sum(
                pw.if_else(pw.this.trend_direction == "POSITIVE", 1, 0)
            ),
            negative_news_count=pw.reducers.sum(
                pw.if_else(pw.this.trend_direction == "NEGATIVE", 1, 0)
            ),
            latest_news_timestamp=pw.reducers.max(pw.this.timestamp)
        )
        
        # Get latest financial data per symbol
        latest_financials = financial_analysis.groupby(pw.this.symbol).reduce(
            symbol=pw.this.symbol,
            latest_filing_date=pw.reducers.max(pw.this.filing_date),
            latest_health_score=pw.reducers.max(pw.this.overall_health_score),
            latest_health_rating=pw.reducers.argmax(pw.this.filing_date, pw.this.health_rating),
            latest_profit_margin=pw.reducers.argmax(pw.this.filing_date, pw.this.profit_margin),
            latest_roe=pw.reducers.argmax(pw.this.filing_date, pw.this.roe),
            latest_debt_to_equity=pw.reducers.argmax(pw.this.filing_date, pw.this.debt_to_equity),
            latest_revenue_growth=pw.reducers.argmax(pw.this.filing_date, pw.this.revenue_growth),
            latest_key_strengths=pw.reducers.argmax(pw.this.filing_date, pw.this.key_strengths),
            latest_key_concerns=pw.reducers.argmax(pw.this.filing_date, pw.this.key_concerns),
            latest_financial_timestamp=pw.reducers.max(pw.this.timestamp)
        )
        
        # Combine financial and news analysis
        comprehensive_analysis = latest_financials.join(
            news_aggregated,
            latest_financials.symbol == news_aggregated.symbol,
            how=pw.JoinMode.LEFT
        ).select(
            symbol=pw.this.symbol,
            
            # Financial metrics
            filing_date=pw.this.latest_filing_date,
            health_score=pw.this.latest_health_score,
            health_rating=pw.this.latest_health_rating,
            profit_margin=pw.this.latest_profit_margin,
            roe=pw.this.latest_roe,
            debt_to_equity=pw.this.latest_debt_to_equity,
            revenue_growth=pw.this.latest_revenue_growth,
            key_strengths=pw.this.latest_key_strengths,
            key_concerns=pw.this.latest_key_concerns,
            
            # News sentiment metrics
            news_articles_count=pw.if_else(pw.this.total_articles.is_not_none(), pw.this.total_articles, 0),
            sentiment_score=pw.if_else(pw.this.avg_sentiment.is_not_none(), pw.this.avg_sentiment, 0.0),
            news_relevance=pw.if_else(pw.this.avg_relevance.is_not_none(), pw.this.avg_relevance, 0.0),
            weighted_news_impact=pw.if_else(pw.this.avg_weighted_impact.is_not_none(), pw.this.avg_weighted_impact, 0.0),
            high_impact_news=pw.if_else(pw.this.high_impact_count.is_not_none(), pw.this.high_impact_count, 0),
            positive_news=pw.if_else(pw.this.positive_news_count.is_not_none(), pw.this.positive_news_count, 0),
            negative_news=pw.if_else(pw.this.negative_news_count.is_not_none(), pw.this.negative_news_count, 0),
            
            # Combined analysis
            overall_sentiment_trend=pw.if_else(
                pw.if_else(pw.this.avg_sentiment.is_not_none(), pw.this.avg_sentiment, 0.0) > 0.2,
                "POSITIVE",
                pw.if_else(
                    pw.if_else(pw.this.avg_sentiment.is_not_none(), pw.this.avg_sentiment, 0.0) < -0.2,
                    "NEGATIVE",
                    "NEUTRAL"
                )
            ),
            
            # Investment recommendation
            investment_recommendation=pw.apply_with_type(
                self._generate_investment_recommendation,
                str,
                pw.this.latest_health_score,
                pw.if_else(pw.this.avg_sentiment.is_not_none(), pw.this.avg_sentiment, 0.0),
                pw.if_else(pw.this.avg_weighted_impact.is_not_none(), pw.this.avg_weighted_impact, 0.0)
            ),
            
            # Risk assessment
            risk_level=pw.apply_with_type(
                self._assess_risk_level,
                str,
                pw.this.latest_debt_to_equity,
                pw.this.latest_revenue_growth,
                pw.if_else(pw.this.avg_sentiment.is_not_none(), pw.this.avg_sentiment, 0.0),
                pw.if_else(pw.this.high_impact_count.is_not_none(), pw.this.high_impact_count, 0)
            ),
            
            # Analysis timestamp
            analysis_timestamp=pw.apply_with_type(
                lambda: int(datetime.now().timestamp() * 1000),
                int
            )
        )
        
        return comprehensive_analysis
    
    def _identify_strengths(self, profit_margin: float, roe: float, current_ratio: float, 
                          debt_to_equity: float, revenue_growth: float) -> str:
        """Identify key financial strengths"""
        strengths = []
        
        if profit_margin > 15:
            strengths.append("High profitability")
        if roe > 15:
            strengths.append("Strong ROE")
        if current_ratio > 2:
            strengths.append("Excellent liquidity")
        if debt_to_equity < 0.5:
            strengths.append("Conservative leverage")
        if revenue_growth > 10:
            strengths.append("Strong growth")
            
        return ", ".join(strengths) if strengths else "Stable operations"
    
    def _identify_concerns(self, profit_margin: float, roe: float, current_ratio: float, 
                          debt_to_equity: float, revenue_growth: float) -> str:
        """Identify key financial concerns"""
        concerns = []
        
        if profit_margin < 5:
            concerns.append("Low margins")
        if roe < 8:
            concerns.append("Weak ROE")
        if current_ratio < 1:
            concerns.append("Liquidity concerns")
        if debt_to_equity > 1.5:
            concerns.append("High leverage")
        if revenue_growth < 0:
            concerns.append("Revenue decline")
            
        return ", ".join(concerns) if concerns else "No major concerns"
    
    def _generate_investment_recommendation(self, health_score: float, sentiment: float, 
                                         news_impact: float) -> str:
        """Generate investment recommendation based on combined analysis"""
        
        # Weighted scoring
        financial_weight = 0.6
        sentiment_weight = 0.3
        news_impact_weight = 0.1
        
        combined_score = (health_score * financial_weight + 
                         (sentiment + 1) * 50 * sentiment_weight +  # Normalize sentiment to 0-100
                         (news_impact + 1) * 50 * news_impact_weight)  # Normalize news impact to 0-100
        
        if combined_score > 75:
            return "STRONG_BUY"
        elif combined_score > 60:
            return "BUY"
        elif combined_score > 40:
            return "HOLD"
        elif combined_score > 25:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _assess_risk_level(self, debt_to_equity: float, revenue_growth: float, 
                          sentiment: float, high_impact_news: int) -> str:
        """Assess overall risk level"""
        
        risk_factors = 0
        
        # Financial risk factors
        if debt_to_equity > 1.0:
            risk_factors += 2
        elif debt_to_equity > 0.5:
            risk_factors += 1
            
        if revenue_growth < 0:
            risk_factors += 2
        elif revenue_growth < 5:
            risk_factors += 1
            
        # Sentiment risk factors
        if sentiment < -0.3:
            risk_factors += 2
        elif sentiment < -0.1:
            risk_factors += 1
            
        # News impact risk
        if high_impact_news > 2:
            risk_factors += 1
            
        if risk_factors >= 5:
            return "HIGH"
        elif risk_factors >= 3:
            return "MEDIUM"
        else:
            return "LOW"


def test_advanced_analyzer():
    """Test the advanced fundamental analyzer"""
    print("🧪 Testing Advanced Fundamental Analyzer...")
    
    # Create mock data for testing
    from src.data_ingestion.web_crawler import PathwayWebCrawler
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    crawler = PathwayWebCrawler(symbols)
    
    # Get data streams
    news_table, financial_table = crawler.create_integrated_data_stream()
    
    # Initialize analyzer
    analyzer = AdvancedFundamentalAnalyzer()
    
    # Perform comprehensive analysis
    print("📊 Running comprehensive fundamental analysis...")
    comprehensive_analysis = analyzer.create_comprehensive_analysis(financial_table, news_table)
    
    # Display results
    print("\n📈 Comprehensive Fundamental Analysis Results:")
    if comprehensive_analysis:
        pw.debug.compute_and_print(comprehensive_analysis)
    else:
        print("   No analysis data available")
    
    print("\n✅ Advanced Fundamental Analyzer test completed!")


if __name__ == "__main__":
    test_advanced_analyzer()