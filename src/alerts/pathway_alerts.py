"""
Real-time News Alerts using Pathway
Monitors news streams and triggers alerts for significant market events
"""

import pathway as pw
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    SENTIMENT_SPIKE = "sentiment_spike"
    VOLUME_SURGE = "volume_surge"
    BREAKING_NEWS = "breaking_news"
    EARNINGS_ALERT = "earnings_alert"
    REGULATORY_NEWS = "regulatory_news"
    MERGER_ACQUISITION = "merger_acquisition"

@dataclass
class NewsAlert:
    """News alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    symbol: str
    title: str
    description: str
    source: str
    url: str
    timestamp: datetime
    sentiment_score: float
    relevance_score: float
    keywords: List[str]
    metadata: Dict[str, Any]

class PathwayNewsAlertSystem:
    """Real-time news monitoring and alert system using Pathway"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.alert_rules = self._initialize_alert_rules()
        self.symbol_watchlist = set()
        
    def _initialize_alert_rules(self) -> Dict[str, Dict]:
        """Initialize smart alert rules"""
        return {
            'sentiment_spike': {
                'threshold': 0.7,  # Sentiment score threshold
                'severity_mapping': {0.8: AlertSeverity.MEDIUM, 0.9: AlertSeverity.HIGH},
                'keywords': ['surge', 'rally', 'crash', 'plummet', 'breakthrough']
            },
            'breaking_news': {
                'keywords': ['breaking', 'urgent', 'alert', 'developing', 'just in'],
                'severity': AlertSeverity.HIGH
            },
            'earnings_alert': {
                'keywords': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'guidance', 'forecast'],
                'severity': AlertSeverity.MEDIUM
            },
            'regulatory_news': {
                'keywords': ['sec', 'fda', 'investigation', 'regulatory', 'compliance', 'lawsuit'],
                'severity': AlertSeverity.HIGH
            },
            'merger_acquisition': {
                'keywords': ['merger', 'acquisition', 'buyout', 'takeover', 'deal', 'acquire'],
                'severity': AlertSeverity.HIGH
            }
        }
    
    def add_symbol_to_watchlist(self, symbol: str):
        """Add symbol to watchlist for monitoring"""
        self.symbol_watchlist.add(symbol.upper())
        logger.info(f"📊 Added {symbol} to news alert watchlist")
    
    def remove_symbol_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        self.symbol_watchlist.discard(symbol.upper())
        logger.info(f"📊 Removed {symbol} from news alert watchlist")
    
    def analyze_article_for_alerts(self, article: Dict) -> List[NewsAlert]:
        """Analyze a single article for potential alerts"""
        alerts = []
        
        try:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {content}"
            
            symbol = self._extract_symbol_from_text(full_text)
            if not symbol and len(self.symbol_watchlist) > 0:
                # Check if any watchlist symbol is mentioned
                for watch_symbol in self.symbol_watchlist:
                    if watch_symbol.lower() in full_text:
                        symbol = watch_symbol
                        break
            
            # Skip if no relevant symbol found
            if not symbol:
                return alerts
            
            # Check for sentiment spikes
            sentiment_score = article.get('sentiment_score', 0.0)
            if abs(sentiment_score) >= self.alert_rules['sentiment_spike']['threshold']:
                severity = AlertSeverity.MEDIUM
                if abs(sentiment_score) >= 0.9:
                    severity = AlertSeverity.HIGH
                
                alert = self._create_alert(
                    alert_type=AlertType.SENTIMENT_SPIKE,
                    severity=severity,
                    symbol=symbol,
                    article=article,
                    description=f"Sentiment spike detected: {sentiment_score:.2f}"
                )
                alerts.append(alert)
            
            # Check for breaking news
            if any(keyword in full_text for keyword in self.alert_rules['breaking_news']['keywords']):
                alert = self._create_alert(
                    alert_type=AlertType.BREAKING_NEWS,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    article=article,
                    description="Breaking news detected"
                )
                alerts.append(alert)
            
            # Check for earnings alerts
            if any(keyword in full_text for keyword in self.alert_rules['earnings_alert']['keywords']):
                alert = self._create_alert(
                    alert_type=AlertType.EARNINGS_ALERT,
                    severity=AlertSeverity.MEDIUM,
                    symbol=symbol,
                    article=article,
                    description="Earnings-related news detected"
                )
                alerts.append(alert)
            
            # Check for regulatory news
            if any(keyword in full_text for keyword in self.alert_rules['regulatory_news']['keywords']):
                alert = self._create_alert(
                    alert_type=AlertType.REGULATORY_NEWS,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    article=article,
                    description="Regulatory news detected"
                )
                alerts.append(alert)
            
            # Check for M&A news
            if any(keyword in full_text for keyword in self.alert_rules['merger_acquisition']['keywords']):
                alert = self._create_alert(
                    alert_type=AlertType.MERGER_ACQUISITION,
                    severity=AlertSeverity.HIGH,
                    symbol=symbol,
                    article=article,
                    description="Merger/Acquisition news detected"
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error analyzing article for alerts: {e}")
            return []
    
    def _extract_symbol_from_text(self, text: str) -> Optional[str]:
        """Extract stock symbol from text"""
        try:
            # Common patterns for stock symbols
            symbol_patterns = [
                r'\$([A-Z]{1,5})',  # $AAPL format
                r'\b([A-Z]{1,5})\s+stock',  # AAPL stock
                r'\b([A-Z]{1,5})\s+shares',  # AAPL shares
            ]
            
            for pattern in symbol_patterns:
                matches = re.findall(pattern, text.upper())
                if matches:
                    return matches[0]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting symbol: {e}")
            return None
    
    def _create_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                     symbol: str, article: Dict, description: str) -> NewsAlert:
        """Create a news alert"""
        alert_id = f"{alert_type.value}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return NewsAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            symbol=symbol,
            title=article.get('title', 'Unknown'),
            description=description,
            source=article.get('source', 'Unknown'),
            url=article.get('url', '#'),
            timestamp=datetime.now(),
            sentiment_score=article.get('sentiment_score', 0.0),
            relevance_score=article.get('relevance_score', 0.0),
            keywords=article.get('keywords', '').split(', ') if article.get('keywords') else [],
            metadata={
                'published_date': article.get('published_date'),
                'article_id': article.get('article_id')
            }
        )
    
    def process_news_batch(self, articles: List[Dict]) -> List[NewsAlert]:
        """Process a batch of news articles for alerts"""
        all_alerts = []
        
        try:
            logger.info(f"🔍 Processing {len(articles)} articles for alerts...")
            
            for article in articles:
                alerts = self.analyze_article_for_alerts(article)
                all_alerts.extend(alerts)
            
            # Store alerts
            for alert in all_alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
            
            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alert_history = [
                alert for alert in self.alert_history 
                if alert.timestamp > cutoff_time
            ]
            
            logger.info(f"✅ Generated {len(all_alerts)} alerts from news batch")
            return all_alerts
            
        except Exception as e:
            logger.error(f"❌ Error processing news batch for alerts: {e}")
            return []
    
    def get_active_alerts(self, symbol: str = None, severity: AlertSeverity = None) -> List[NewsAlert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if symbol:
            alerts = [alert for alert in alerts if alert.symbol.upper() == symbol.upper()]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        active_alerts = list(self.active_alerts.values())
        
        summary = {
            'total_active_alerts': len(active_alerts),
            'alerts_by_severity': {
                'critical': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            },
            'alerts_by_type': {},
            'watchlist_symbols': list(self.symbol_watchlist),
            'recent_alerts': []
        }
        
        # Count by type
        for alert_type in AlertType:
            summary['alerts_by_type'][alert_type.value] = len([
                a for a in active_alerts if a.alert_type == alert_type
            ])
        
        # Get recent alerts
        recent_alerts = sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
        summary['recent_alerts'] = [
            {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'severity': alert.severity.value,
                'symbol': alert.symbol,
                'title': alert.title,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in recent_alerts
        ]
        
        return summary
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Remove from active alerts
        to_remove = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.timestamp < cutoff_time
        ]
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
        
        logger.info(f"📧 Cleared {len(to_remove)} old alerts")

# Global alert system instance
pathway_alert_system = None

def initialize_pathway_alert_system():
    """Initialize the global alert system"""
    global pathway_alert_system
    try:
        pathway_alert_system = PathwayNewsAlertSystem()
        logger.info("✅ Pathway news alert system initialized")
        return pathway_alert_system
    except Exception as e:
        logger.error(f"❌ Failed to initialize Pathway alert system: {e}")
        return None

def add_alert_watchlist_symbol(symbol: str):
    """Add symbol to alert watchlist"""
    global pathway_alert_system
    if pathway_alert_system:
        pathway_alert_system.add_symbol_to_watchlist(symbol)

def get_news_alerts(symbol: str = None) -> Dict[str, Any]:
    """Get news alerts for symbol"""
    global pathway_alert_system
    
    if not pathway_alert_system:
        pathway_alert_system = initialize_pathway_alert_system()
    
    if pathway_alert_system:
        return {
            'alerts': [
                {
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'symbol': alert.symbol,
                    'title': alert.title,
                    'description': alert.description,
                    'source': alert.source,
                    'url': alert.url,
                    'timestamp': alert.timestamp.isoformat(),
                    'sentiment_score': alert.sentiment_score,
                    'relevance_score': alert.relevance_score
                }
                for alert in pathway_alert_system.get_active_alerts(symbol)
            ],
            'summary': pathway_alert_system.get_alert_summary()
        }
    
    return {'alerts': [], 'summary': {}}

if __name__ == "__main__":
    # Test the alert system
    alert_system = initialize_pathway_alert_system()
    if alert_system:
        alert_system.add_symbol_to_watchlist("AAPL")
        
        # Mock article for testing
        test_article = {
            'title': 'BREAKING: Apple Reports Exceptional Q4 Earnings',
            'content': 'Apple Inc. exceeded expectations with record revenue...',
            'sentiment_score': 0.8,
            'relevance_score': 0.9,
            'source': 'Financial Times',
            'url': 'https://example.com/news',
            'keywords': 'apple, earnings, revenue, exceptional'
        }
        
        alerts = alert_system.analyze_article_for_alerts(test_article)
        print(f"🚨 Generated {len(alerts)} alerts")
        for alert in alerts:
            print(f"   - {alert.alert_type.value} ({alert.severity.value}): {alert.description}")