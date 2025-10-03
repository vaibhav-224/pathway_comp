"""
AI-Powered Portfolio Analysis using Gemini LLM
Provides intelligent insights and interpretations for portfolio data
"""

import google.generativeai as genai
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PortfolioAIAnalyst:
    """AI-powered portfolio analysis using Gemini"""
    
    def __init__(self, gemini_client):
        self.client = gemini_client
        self.model = genai.GenerativeModel('gemini-pro')
        
    def analyze_portfolio_performance(self, portfolio_data: Dict) -> Dict:
        """
        Analyze overall portfolio performance and provide insights
        """
        try:
            positions = portfolio_data.get('positions', [])
            summary = {
                'total_value': portfolio_data.get('total_value', 0),
                'total_cost': portfolio_data.get('total_cost', 0),
                'total_pnl': portfolio_data.get('total_pnl', 0),
                'total_pnl_percent': portfolio_data.get('total_pnl_percent', 0),
                'volatility': portfolio_data.get('volatility', 0),
                'num_positions': len(positions)
            }
            
            # Prepare data for AI analysis
            performance_context = f"""
Portfolio Performance Analysis:

Summary Metrics:
- Total Portfolio Value: ${summary['total_value']:,.2f}
- Total Cost Basis: ${summary['total_cost']:,.2f}
- Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:.2f}%)
- Portfolio Volatility: {summary['volatility']:.2f}%
- Number of Positions: {summary['num_positions']}

Individual Positions:
"""
            
            for pos in positions[:10]:  # Limit to top 10 positions
                performance_context += f"""
- {pos['symbol']}: {pos['shares']:.2f} shares
  Current Price: ${pos['current_price']:.2f}
  Purchase Price: ${pos['purchase_price']:.2f}
  Position Value: ${pos['position_value']:,.2f}
  P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_percent']:.2f}%)
"""
            
            prompt = f"""
{performance_context}

As a professional financial advisor, provide a comprehensive analysis of this portfolio:

1. **Overall Performance Assessment**: How is the portfolio performing? What are the key strengths and weaknesses?

2. **Risk Analysis**: Analyze the risk profile based on volatility and position concentration.

3. **Position Analysis**: Which positions are performing well and which need attention?

4. **Market Context**: Consider current market conditions and how they might affect this portfolio.

5. **Recommendations**: Provide 3-5 specific actionable recommendations for improving the portfolio.

6. **Risk Management**: Suggest risk management strategies based on the current portfolio composition.

Provide insights in a structured JSON format with sections for:
- summary (brief overall assessment)
- performance_grade (A+ to F)
- key_insights (array of 3-5 key insights)
- top_performers (best performing stocks)
- underperformers (stocks needing attention)  
- risk_assessment (low/medium/high with explanation)
- recommendations (array of specific actions)
- market_outlook (brief market context)
"""
            
            response = self.model.generate_content(prompt)
            
            # Try to parse JSON response, fallback to structured text
            try:
                analysis = json.loads(response.text)
            except:
                # If JSON parsing fails, create structured response
                analysis = {
                    'summary': response.text[:200] + "...",
                    'performance_grade': self._calculate_performance_grade(summary['total_pnl_percent']),
                    'key_insights': self._extract_insights_from_text(response.text),
                    'risk_assessment': self._assess_risk_level(summary['volatility']),
                    'full_analysis': response.text
                }
            
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['analysis_type'] = 'portfolio_performance'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in portfolio performance analysis: {e}")
            return {
                'error': str(e),
                'summary': 'Unable to generate AI analysis at this time',
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_individual_stock(self, stock_data: Dict, market_data: Dict = None) -> Dict:
        """
        Analyze individual stock performance and provide insights
        """
        try:
            symbol = stock_data.get('symbol', 'Unknown')
            
            stock_context = f"""
Stock Analysis for {symbol}:

Position Details:
- Symbol: {symbol}
- Shares Owned: {stock_data.get('shares', 0)}
- Purchase Price: ${stock_data.get('purchase_price', 0):.2f}
- Current Price: ${stock_data.get('current_price', 0):.2f}
- Position Value: ${stock_data.get('position_value', 0):,.2f}
- Unrealized P&L: ${stock_data.get('unrealized_pnl', 0):,.2f} ({stock_data.get('unrealized_pnl_percent', 0):.2f}%)

Technical Indicators (if available):
"""
            
            if market_data:
                for key, value in market_data.items():
                    if key not in ['symbol', 'timestamp']:
                        stock_context += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            prompt = f"""
{stock_context}

As a financial analyst, provide a detailed analysis of this stock position:

1. **Position Assessment**: How is this specific position performing?

2. **Technical Analysis**: Based on the available indicators, what's the technical outlook?

3. **Risk Factors**: What are the key risks for this position?

4. **Action Recommendation**: Should I hold, buy more, or consider selling? Why?

5. **Price Targets**: What are reasonable price targets for this stock?

Provide analysis in JSON format with:
- position_grade (A+ to F)
- recommendation (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
- target_price (estimated fair value)
- risk_level (LOW/MEDIUM/HIGH)
- key_points (array of 3-4 key analysis points)
- action_items (specific recommendations)
"""
            
            response = self.model.generate_content(prompt)
            
            try:
                analysis = json.loads(response.text)
            except:
                analysis = {
                    'position_grade': self._calculate_position_grade(stock_data.get('unrealized_pnl_percent', 0)),
                    'recommendation': self._generate_recommendation(stock_data.get('unrealized_pnl_percent', 0)),
                    'key_points': self._extract_insights_from_text(response.text),
                    'full_analysis': response.text
                }
            
            analysis['symbol'] = symbol
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['analysis_type'] = 'individual_stock'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in individual stock analysis for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'summary': 'Unable to generate AI analysis for this stock',
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_market_insights(self, portfolio_data: Dict, news_data: List = None) -> Dict:
        """
        Generate market insights and their impact on the portfolio
        """
        try:
            sectors = self._analyze_sector_exposure(portfolio_data.get('positions', []))
            
            context = f"""
Portfolio Market Context Analysis:

Portfolio Composition:
- Total Value: ${portfolio_data.get('total_value', 0):,.2f}
- Number of Positions: {len(portfolio_data.get('positions', []))}
- Sector Exposure: {json.dumps(sectors, indent=2)}

Recent News Headlines:
"""
            
            if news_data:
                for article in news_data[:5]:  # Top 5 news items
                    context += f"- {article.get('title', 'No title')}\n"
            
            prompt = f"""
{context}

As a market strategist, analyze the current market environment and its impact on this portfolio:

1. **Market Environment**: What's the current market sentiment and key drivers?

2. **Sector Impact**: How might current market conditions affect the sectors in this portfolio?

3. **Opportunities**: What market opportunities should be considered?

4. **Threats**: What are the key market risks to watch?

5. **Strategic Recommendations**: How should the portfolio be positioned for current market conditions?

Provide insights in JSON format with:
- market_sentiment (BULLISH/NEUTRAL/BEARISH)
- key_market_drivers (array of main market factors)
- sector_outlook (analysis of relevant sectors)
- opportunities (potential opportunities)
- threats (key risks to monitor)
- strategic_actions (recommended portfolio adjustments)
"""
            
            response = self.model.generate_content(prompt)
            
            try:
                analysis = json.loads(response.text)
            except:
                analysis = {
                    'market_sentiment': 'NEUTRAL',
                    'key_market_drivers': self._extract_insights_from_text(response.text),
                    'full_analysis': response.text
                }
            
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['analysis_type'] = 'market_insights'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market insights analysis: {e}")
            return {
                'error': str(e),
                'summary': 'Unable to generate market insights at this time',
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_performance_grade(self, pnl_percent: float) -> str:
        """Calculate performance grade based on P&L percentage"""
        if pnl_percent >= 20:
            return 'A+'
        elif pnl_percent >= 15:
            return 'A'
        elif pnl_percent >= 10:
            return 'B+'
        elif pnl_percent >= 5:
            return 'B'
        elif pnl_percent >= 0:
            return 'C'
        elif pnl_percent >= -5:
            return 'D'
        else:
            return 'F'
    
    def _calculate_position_grade(self, pnl_percent: float) -> str:
        """Calculate individual position grade"""
        return self._calculate_performance_grade(pnl_percent)
    
    def _generate_recommendation(self, pnl_percent: float) -> str:
        """Generate basic recommendation based on performance"""
        if pnl_percent >= 15:
            return 'HOLD'  # Strong performer, hold
        elif pnl_percent >= 5:
            return 'HOLD'  # Good performer, hold
        elif pnl_percent >= -5:
            return 'HOLD'  # Neutral, hold
        elif pnl_percent >= -15:
            return 'REVIEW'  # Underperforming, review
        else:
            return 'CONSIDER_SELL'  # Poor performer, consider selling
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess risk level based on volatility"""
        if volatility < 10:
            return 'LOW'
        elif volatility < 20:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _extract_insights_from_text(self, text: str) -> List[str]:
        """Extract key insights from text response"""
        sentences = text.split('.')
        insights = []
        for sentence in sentences[:5]:  # Top 5 sentences
            if len(sentence.strip()) > 20:
                insights.append(sentence.strip())
        return insights
    
    def _analyze_sector_exposure(self, positions: List[Dict]) -> Dict:
        """Analyze sector exposure of the portfolio"""
        # This is a simplified sector mapping - in production, you'd use a proper sector database
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'NVDA': 'Technology',
            'META': 'Technology',
            'BRK.B': 'Financial Services',
            'JPM': 'Financial Services',
            'JNJ': 'Healthcare'
        }
        
        sectors = {}
        total_value = sum(pos.get('position_value', 0) for pos in positions)
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            sector = sector_map.get(symbol, 'Other')
            value = pos.get('position_value', 0)
            
            if sector not in sectors:
                sectors[sector] = {'value': 0, 'percentage': 0}
            
            sectors[sector]['value'] += value
        
        # Calculate percentages
        for sector in sectors:
            if total_value > 0:
                sectors[sector]['percentage'] = (sectors[sector]['value'] / total_value) * 100
        
        return sectors

# Global instance
_portfolio_ai_analyst = None

def get_portfolio_ai_analyst(gemini_client=None):
    """Get the global portfolio AI analyst instance"""
    global _portfolio_ai_analyst
    if _portfolio_ai_analyst is None and gemini_client:
        _portfolio_ai_analyst = PortfolioAIAnalyst(gemini_client)
    return _portfolio_ai_analyst