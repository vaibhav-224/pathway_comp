"""
AI Insights Engine
Generates natural language explanations for market anomalies using LLM
Integrates with Pathway tables for real-time context and analysis
Supports both OpenAI and Google Gemini APIs
"""
import pathway as pw
import openai
import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIInsightsEngine:
    """AI-powered insights generation using Pathway and LLM (OpenAI or Gemini)"""
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        # Determine which AI provider to use
        self.provider = provider or os.getenv("AI_PROVIDER", "gemini")
        self.client = None
        self.gemini_model = None
        
        if self.provider.lower() == "gemini":
            # Initialize Gemini
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-pro")
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel(self.model)
                print("✅ Gemini AI client initialized")
            else:
                print("⚠️  No Gemini API key found - using mock insights")
                
        elif self.provider.lower() == "openai":
            # Initialize OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            
            if self.api_key:
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
                print("✅ OpenAI client initialized")
            else:
                print("⚠️  No OpenAI API key found - using mock insights")
        else:
            print(f"⚠️  Unknown AI provider '{self.provider}' - using mock insights")
            self.api_key = None
    
    def create_context_table(self, market_table: pw.Table, news_table: pw.Table, 
                           anomaly_table: pw.Table) -> pw.Table:
        """Create a comprehensive context table using Pathway operations"""
        
        # Extract key market context using Pathway
        market_context = market_table.select(
            symbol=pw.this.symbol,
            current_price=pw.this.price,
            change_pct=pw.this.change_pct,
            volume=pw.this.volume,
            market_cap=pw.this.market_cap,
            timestamp=pw.this.timestamp
        )
        
        # Add anomaly insights
        enhanced_context = anomaly_table.select(
            symbol=pw.this.symbol,
            current_price=pw.this.price,
            change_pct=pw.this.change_pct,
            volume=pw.this.volume,
            timestamp=pw.this.timestamp,
            is_anomaly=pw.this.is_price_anomaly,
            anomaly_type=pw.this.anomaly_type,
            alert_level=pw.this.alert_level,
            priority_score=pw.this.priority_score,
            # Create context summary using Pathway operations
            market_status=pw.if_else(
                pw.this.is_price_anomaly,
                "ANOMALOUS_MOVEMENT",
                "NORMAL_TRADING"
            ),
            movement_direction=pw.if_else(
                pw.this.change_pct > 0,
                "UPWARD",
                pw.if_else(
                    pw.this.change_pct < 0,
                    "DOWNWARD", 
                    "FLAT"
                )
            ),
            movement_magnitude=pw.if_else(
                pw.this.change_pct > 5.0,
                "LARGE",
                pw.if_else(
                    pw.this.change_pct > 2.0,
                    "MODERATE",
                    "SMALL"
                )
            )
        )
        
        return enhanced_context
    
    def generate_anomaly_insights(self, anomaly_table: pw.Table) -> pw.Table:
        """Generate AI insights for anomalies using Pathway and LLM integration"""
        
        # Filter for anomalies only using the correct column name
        anomalies_only = anomaly_table.filter(pw.this.is_price_anomaly == True)
        
        # Add AI-generated insights using Pathway operations
        insights_table = anomalies_only.select(
            symbol=pw.this.symbol,
            current_price=pw.this.price,
            change_pct=pw.this.change_pct,
            anomaly_type=pw.this.anomaly_type,
            alert_level=pw.this.alert_level,
            priority_score=pw.this.priority_score,
            timestamp=pw.this.timestamp,
            # Generate insight categories using Pathway logic
            insight_category=pw.if_else(
                pw.this.anomaly_type == "PRICE_SPIKE",
                "BULLISH_MOMENTUM",
                pw.if_else(
                    pw.this.anomaly_type == "PRICE_DROP",
                    "BEARISH_PRESSURE",
                    "VOLUME_ACTIVITY"
                )
            ),
            # Risk assessment using Pathway calculations
            risk_level=pw.if_else(
                pw.this.priority_score > 25.0,
                "HIGH_RISK",
                pw.if_else(
                    pw.this.priority_score > 15.0,
                    "MEDIUM_RISK",
                    "LOW_RISK"
                )
            ),
            # Market impact assessment
            market_impact=pw.if_else(
                (pw.this.change_pct > 5.0) | (pw.this.change_pct < -5.0),
                "SIGNIFICANT_IMPACT",
                pw.if_else(
                    (pw.this.change_pct > 2.0) | (pw.this.change_pct < -2.0),
                    "MODERATE_IMPACT",
                    "MINOR_IMPACT"
                )
            ),
            # Generate basic explanation using Pathway string operations
            basic_explanation=pw.this.symbol + "_" + pw.this.anomaly_type + "_detected_at_" + pw.this.timestamp,
            # AI insight status
            ai_insight_ready=False
        )
        
        return insights_table
    
    def enhance_insights_with_llm(self, insights_data: List[Dict]) -> List[Dict]:
        """Enhance basic insights with LLM-generated explanations"""
        
        enhanced_insights = []
        
        for insight in insights_data:
            # Create context for LLM
            context = {
                "symbol": insight.get("symbol"),
                "price": insight.get("current_price"),
                "change_pct": insight.get("change_pct"),
                "anomaly_type": insight.get("anomaly_type"),
                "alert_level": insight.get("alert_level"),
                "risk_level": insight.get("risk_level"),
                "market_impact": insight.get("market_impact")
            }
            
            # Generate LLM insight
            llm_explanation = self._generate_llm_explanation(context)
            
            # Add LLM insights to the data
            enhanced_insight = insight.copy()
            enhanced_insight.update({
                "ai_explanation": llm_explanation["explanation"],
                "trading_implications": llm_explanation["implications"],
                "suggested_actions": llm_explanation["actions"],
                "confidence_score": llm_explanation["confidence"]
            })
            
            enhanced_insights.append(enhanced_insight)
        
        return enhanced_insights
    
    def _generate_llm_explanation(self, context: Dict) -> Dict:
        """Generate LLM explanation for market anomaly using selected provider"""
        
        if self.provider.lower() == "gemini" and self.gemini_model:
            return self._generate_gemini_explanation(context)
        elif self.provider.lower() == "openai" and self.client:
            return self._generate_openai_explanation(context)
        else:
            # Return mock insights if no API key or invalid provider
            return self._generate_mock_insight(context)
    
    def _generate_gemini_explanation(self, context: Dict) -> Dict:
        """Generate explanation using Google Gemini API"""
        
        try:
            prompt = self._create_anomaly_prompt(context)
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse the response
            content = response.text
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return self._generate_mock_insight(context)
    
    def _generate_openai_explanation(self, context: Dict) -> Dict:
        """Generate explanation using OpenAI API"""
        
        try:
            prompt = self._create_anomaly_prompt(context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a expert financial analyst providing real-time market insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse the response
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"⚠️ OpenAI API error: {e}")
            return self._generate_mock_insight(context)
    
    def _create_anomaly_prompt(self, context: Dict) -> str:
        """Create structured prompt for LLM analysis"""
        
        prompt = f"""
        MARKET ANOMALY DETECTED:
        
        Stock: {context['symbol']}
        Current Price: ${context['price']:.2f}
        Price Change: {context['change_pct']:+.2f}%
        Anomaly Type: {context['anomaly_type']}
        Alert Level: {context['alert_level']}
        Risk Assessment: {context['risk_level']}
        Market Impact: {context['market_impact']}
        
        Please provide:
        1. EXPLANATION: Why is this movement significant? (2-3 sentences)
        2. IMPLICATIONS: What does this mean for traders/investors? (1-2 sentences)  
        3. ACTIONS: Suggested next steps or considerations (1-2 bullet points)
        4. CONFIDENCE: Your confidence in this analysis (1-100)
        
        Format as JSON with keys: explanation, implications, actions, confidence
        """
        
        return prompt
    
    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response into structured format"""
        
        try:
            # Try to parse as JSON first
            return json.loads(content)
        except:
            # Fallback parsing
            return {
                "explanation": content[:200] + "..." if len(content) > 200 else content,
                "implications": "Market conditions require careful monitoring.",
                "actions": ["Monitor price action", "Check news catalysts"],
                "confidence": 75
            }
    
    def _generate_mock_insight(self, context: Dict) -> Dict:
        """Generate mock insights when LLM is not available"""
        
        symbol = context.get("symbol", "STOCK")
        change = context.get("change_pct", 0)
        anomaly_type = context.get("anomaly_type", "UNKNOWN")
        
        if anomaly_type == "PRICE_SPIKE":
            explanation = f"{symbol} shows strong bullish momentum with {change:+.2f}% move. This indicates significant buying interest."
            implications = "Potential breakout or positive catalyst. Consider momentum strategies."
            actions = ["Monitor volume confirmation", "Check for news catalysts"]
        elif anomaly_type == "PRICE_DROP":
            explanation = f"{symbol} experiencing selling pressure with {change:+.2f}% decline. This suggests negative sentiment."
            implications = "Risk-off environment. Consider defensive positioning."
            actions = ["Assess support levels", "Review fundamentals"]
        else:
            explanation = f"{symbol} showing unusual activity. Requires closer monitoring."
            implications = "Market conditions uncertain. Proceed with caution."
            actions = ["Monitor closely", "Wait for confirmation"]
        
        return {
            "explanation": explanation,
            "implications": implications,
            "actions": actions,
            "confidence": 80
        }
    
    def create_ai_enhanced_table(self, insights_table: pw.Table, 
                               enhanced_insights: List[Dict]) -> pw.Table:
        """Create Pathway table with AI-enhanced insights"""
        
        if not enhanced_insights:
            return insights_table
        
        # Convert enhanced insights back to Pathway table format
        ai_schema = pw.schema_from_dict({
            'symbol': str,
            'current_price': float,
            'change_pct': float,
            'anomaly_type': str,
            'alert_level': str,
            'ai_explanation': str,
            'trading_implications': str,
            'confidence_score': int,
            'timestamp': str
        })
        
        # Convert to rows for Pathway
        ai_rows = []
        for insight in enhanced_insights:
            row = (
                insight.get('symbol', ''),
                insight.get('current_price', 0.0),
                insight.get('change_pct', 0.0),
                insight.get('anomaly_type', ''),
                insight.get('alert_level', ''),
                insight.get('ai_explanation', ''),
                insight.get('trading_implications', ''),
                insight.get('confidence_score', 0),
                insight.get('timestamp', '')
            )
            ai_rows.append(row)
        
        if ai_rows:
            ai_table = pw.debug.table_from_rows(schema=ai_schema, rows=ai_rows)
            return ai_table
        
        return None

def test_ai_insights():
    """Test the AI insights engine"""
    print("🧪 Testing AI Insights Engine...")
    
    # Create test anomaly data
    test_anomalies = [
        ("AAPL", "2025-09-21T15:00:00", 250.0, 0, 245.0, 252.0, 243.0, 4.2, 3600000000000, 50000000, True, 1.68, "PRICE_SPIKE", "MEDIUM", 16.8),
        ("TSLA", "2025-09-21T15:00:00", 400.0, 150000000, 430.0, 430.0, 400.0, -6.5, 1400000000000, 90000000, True, 2.6, "PRICE_DROP", "MEDIUM", 26.0),
    ]
    
    # Create Pathway table for anomalies
    anomaly_schema = pw.schema_from_dict({
        'symbol': str, 'timestamp': str, 'price': float, 'volume': int,
        'open': float, 'high': float, 'low': float, 'change_pct': float,
        'market_cap': int, 'avg_volume': int, 'is_price_anomaly': bool,
        'anomaly_severity': float, 'anomaly_type': str, 'alert_level': str,
        'priority_score': float
    })
    
    anomaly_table = pw.debug.table_from_rows(schema=anomaly_schema, rows=test_anomalies)
    
    print("📊 Test Anomaly Data:")
    pw.debug.compute_and_print(anomaly_table)
    
    # Initialize AI engine with Gemini (default from .env)
    ai_engine = AIInsightsEngine()
    print(f"🤖 Using AI Provider: {ai_engine.provider}")
    
    # Generate insights using Pathway
    print("\n🤖 Generating AI Insights...")
    insights_table = ai_engine.generate_anomaly_insights(anomaly_table)
    
    print("🔍 Pathway-Generated Insights:")
    pw.debug.compute_and_print(insights_table)
    
    # Test LLM integration with real API call
    print("\n🧠 Testing LLM Integration...")
    test_context = {
        "symbol": "AAPL",
        "price": 250.0,
        "change_pct": 4.2,
        "anomaly_type": "PRICE_SPIKE",
        "alert_level": "MEDIUM",
        "risk_level": "MEDIUM_RISK",
        "market_impact": "MODERATE_IMPACT"
    }
    
    llm_result = ai_engine._generate_llm_explanation(test_context)
    print("💬 LLM Generated Insight:")
    print(f"  Provider: {ai_engine.provider}")
    print(f"  Explanation: {llm_result.get('explanation', 'No explanation')}")
    print(f"  Confidence: {llm_result.get('confidence', 'No confidence score')}")
    
    return ai_engine

# Global function for easy access
def get_enhanced_ai_analysis(context: Dict) -> Dict:
    """Get enhanced AI analysis for a given context"""
    try:
        ai_engine = AIInsightsEngine()
        return ai_engine._generate_llm_explanation(context)
    except Exception as e:
        return {
            "explanation": f"AI analysis temporarily unavailable: {str(e)}",
            "confidence": 0.0,
            "risk_assessment": "UNKNOWN",
            "market_implications": "Unable to assess"
        }

if __name__ == "__main__":
    test_ai_insights()