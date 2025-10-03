"""
Real-time Alerts & Notifications System
Configurable alerts for price movements, news sentiment, earnings, and custom triggers
"""

import smtplib
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
import os
from enum import Enum

logger = logging.getLogger(__name__)

class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE = "price_change"
    VOLUME_SPIKE = "volume_spike"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_SIGNAL = "technical_signal"
    EARNINGS_ANNOUNCEMENT = "earnings_announcement"
    CUSTOM = "custom"

class NotificationMethod(Enum):
    EMAIL = "email"
    SMS = "sms"  # Would require Twilio integration
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    symbol: str
    alert_type: AlertType
    condition: Dict  # Specific conditions for the alert
    notification_methods: List[NotificationMethod]
    priority: AlertPriority
    message_template: str
    is_active: bool
    created_at: datetime
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 60  # Minimum time between notifications
    max_triggers: int = 10  # Maximum number of times to trigger
    expires_at: Optional[datetime] = None

@dataclass
class TriggeredAlert:
    alert_id: str
    symbol: str
    alert_type: AlertType
    trigger_value: float
    trigger_data: Dict
    message: str
    priority: AlertPriority
    triggered_at: datetime
    notification_sent: bool = False

class NotificationService:
    """Handle different notification methods"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.email_config = self.config.get('email', {})
        self.slack_config = self.config.get('slack', {})
        self.sms_config = self.config.get('sms', {})
    
    def send_notification(self, alert: TriggeredAlert, methods: List[NotificationMethod]):
        """Send notification via specified methods"""
        for method in methods:
            try:
                if method == NotificationMethod.EMAIL:
                    self._send_email(alert)
                elif method == NotificationMethod.SLACK:
                    self._send_slack(alert)
                elif method == NotificationMethod.SMS:
                    self._send_sms(alert)
                elif method == NotificationMethod.WEBHOOK:
                    self._send_webhook(alert)
                elif method == NotificationMethod.LOG:
                    self._log_alert(alert)
                
                logger.info(f"✅ Sent {method.value} notification for {alert.symbol} alert")
                
            except Exception as e:
                logger.error(f"❌ Failed to send {method.value} notification: {e}")
    
    def _send_email(self, alert: TriggeredAlert):
        """Send email notification"""
        if not self.email_config:
            logger.warning("Email configuration not found")
            return
        
        smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = self.email_config.get('smtp_port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        to_email = self.email_config.get('to_email')
        
        if not all([username, password, to_email]):
            logger.warning("Incomplete email configuration")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to_email
        msg['Subject'] = f"🚨 {alert.priority.value.upper()} Alert: {alert.symbol}"
        
        # Email body
        body = f"""
        Financial Alert Triggered
        
        Symbol: {alert.symbol}
        Type: {alert.alert_type.value}
        Priority: {alert.priority.value.upper()}
        Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
        Trigger Value: {alert.trigger_value}
        
        Message: {alert.message}
        
        Trigger Data:
        {json.dumps(alert.trigger_data, indent=2)}
        
        ---
        FinanceAI Alert System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_slack(self, alert: TriggeredAlert):
        """Send Slack notification"""
        webhook_url = self.slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        # Priority emoji mapping
        emoji_map = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔥"
        }
        
        # Create Slack message
        emoji = emoji_map.get(alert.priority, "📊")
        color_map = {
            AlertPriority.LOW: "#36a64f",
            AlertPriority.MEDIUM: "#ff9500",
            AlertPriority.HIGH: "#ff0000",
            AlertPriority.CRITICAL: "#8b0000"
        }
        
        color = color_map.get(alert.priority, "#36a64f")
        
        slack_data = {
            "text": f"{emoji} Financial Alert: {alert.symbol}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {
                            "title": "Symbol",
                            "value": alert.symbol,
                            "short": True
                        },
                        {
                            "title": "Alert Type",
                            "value": alert.alert_type.value.replace('_', ' ').title(),
                            "short": True
                        },
                        {
                            "title": "Priority",
                            "value": alert.priority.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Trigger Value",
                            "value": f"{alert.trigger_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        }
                    ],
                    "footer": "FinanceAI Alert System",
                    "ts": int(alert.triggered_at.timestamp())
                }
            ]
        }
        
        response = requests.post(webhook_url, json=slack_data)
        response.raise_for_status()
    
    def _send_sms(self, alert: TriggeredAlert):
        """Send SMS notification (requires Twilio)"""
        # This would require Twilio integration
        logger.info(f"SMS notification would be sent: {alert.message}")
    
    def _send_webhook(self, alert: TriggeredAlert):
        """Send webhook notification"""
        webhook_url = self.config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return
        
        data = {
            'alert': asdict(alert),
            'timestamp': alert.triggered_at.isoformat()
        }
        
        response = requests.post(webhook_url, json=data, timeout=10)
        response.raise_for_status()
    
    def _log_alert(self, alert: TriggeredAlert):
        """Log alert to console/file"""
        priority_emoji = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️", 
            AlertPriority.HIGH: "🚨",
            AlertPriority.CRITICAL: "🔥"
        }
        
        emoji = priority_emoji.get(alert.priority, "📊")
        logger.info(f"{emoji} ALERT [{alert.priority.value.upper()}] {alert.symbol}: {alert.message}")

class AlertManager:
    """Main alert management system"""
    
    def __init__(self, notification_service: NotificationService):
        self.alerts: Dict[str, Alert] = {}
        self.triggered_alerts: List[TriggeredAlert] = []
        self.notification_service = notification_service
        self.monitoring_thread = None
        self.is_monitoring = False
        self.data_callbacks: Dict[str, Callable] = {}
    
    def add_alert(self, symbol: str, alert_type: AlertType, condition: Dict,
                  notification_methods: List[NotificationMethod], 
                  priority: AlertPriority = AlertPriority.MEDIUM,
                  message_template: str = None,
                  cooldown_minutes: int = 60,
                  max_triggers: int = 10,
                  expires_hours: int = None) -> str:
        """Add new alert"""
        
        alert_id = f"{symbol}_{alert_type.value}_{int(datetime.now().timestamp())}"
        
        # Default message template
        if not message_template:
            message_template = f"Alert triggered for {symbol}: {alert_type.value}"
        
        # Set expiration
        expires_at = None
        if expires_hours:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        alert = Alert(
            id=alert_id,
            symbol=symbol,
            alert_type=alert_type,
            condition=condition,
            notification_methods=notification_methods,
            priority=priority,
            message_template=message_template,
            is_active=True,
            created_at=datetime.now(),
            cooldown_minutes=cooldown_minutes,
            max_triggers=max_triggers,
            expires_at=expires_at
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"✅ Created alert {alert_id} for {symbol}")
        
        return alert_id
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"🗑️ Removed alert {alert_id}")
            return True
        return False
    
    def update_alert(self, alert_id: str, **kwargs) -> bool:
        """Update existing alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        for key, value in kwargs.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
        
        logger.info(f"📝 Updated alert {alert_id}")
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if alert.is_active]
    
    def get_triggered_alerts(self, hours: int = 24) -> List[TriggeredAlert]:
        """Get recently triggered alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.triggered_alerts 
                if alert.triggered_at >= cutoff_time]
    
    def register_data_callback(self, data_type: str, callback: Callable):
        """Register callback for data updates"""
        self.data_callbacks[data_type] = callback
    
    def check_alert(self, alert: Alert, current_data: Dict) -> bool:
        """Check if alert should be triggered"""
        try:
            # Check if alert is expired
            if alert.expires_at and datetime.now() > alert.expires_at:
                alert.is_active = False
                return False
            
            # Check if alert has exceeded max triggers
            if alert.triggered_count >= alert.max_triggers:
                alert.is_active = False
                return False
            
            # Check cooldown period
            if (alert.last_triggered and 
                datetime.now() - alert.last_triggered < timedelta(minutes=alert.cooldown_minutes)):
                return False
            
            # Check specific alert conditions
            should_trigger = False
            trigger_value = 0.0
            
            if alert.alert_type == AlertType.PRICE_ABOVE:
                current_price = current_data.get('price', 0)
                target_price = alert.condition.get('target_price', float('inf'))
                should_trigger = current_price >= target_price
                trigger_value = current_price
                
            elif alert.alert_type == AlertType.PRICE_BELOW:
                current_price = current_data.get('price', 0)
                target_price = alert.condition.get('target_price', 0)
                should_trigger = current_price <= target_price
                trigger_value = current_price
                
            elif alert.alert_type == AlertType.PRICE_CHANGE:
                price_change = current_data.get('change_percent', 0)
                threshold = alert.condition.get('threshold_percent', 5)
                direction = alert.condition.get('direction', 'both')  # 'up', 'down', 'both'
                
                if direction == 'up':
                    should_trigger = price_change >= threshold
                elif direction == 'down':
                    should_trigger = price_change <= -threshold
                else:  # both
                    should_trigger = abs(price_change) >= threshold
                
                trigger_value = price_change
                
            elif alert.alert_type == AlertType.VOLUME_SPIKE:
                current_volume = current_data.get('volume', 0)
                avg_volume = current_data.get('avg_volume', current_volume)
                spike_ratio = alert.condition.get('spike_ratio', 2.0)
                
                should_trigger = current_volume >= avg_volume * spike_ratio
                trigger_value = current_volume / avg_volume if avg_volume > 0 else 0
                
            elif alert.alert_type == AlertType.NEWS_SENTIMENT:
                sentiment_score = current_data.get('sentiment_score', 0.5)
                threshold = alert.condition.get('sentiment_threshold', 0.7)
                direction = alert.condition.get('direction', 'positive')
                
                if direction == 'positive':
                    should_trigger = sentiment_score >= threshold
                else:  # negative
                    should_trigger = sentiment_score <= (1 - threshold)
                
                trigger_value = sentiment_score
                
            elif alert.alert_type == AlertType.TECHNICAL_SIGNAL:
                signal_type = current_data.get('technical_signal')
                target_signal = alert.condition.get('signal_type')
                min_confidence = alert.condition.get('min_confidence', 0.7)
                signal_confidence = current_data.get('signal_confidence', 0)
                
                should_trigger = (signal_type == target_signal and 
                                signal_confidence >= min_confidence)
                trigger_value = signal_confidence
            
            return should_trigger, trigger_value
            
        except Exception as e:
            logger.error(f"Error checking alert {alert.id}: {e}")
            return False, 0.0
    
    def trigger_alert(self, alert: Alert, trigger_value: float, trigger_data: Dict):
        """Trigger an alert and send notifications"""
        try:
            # Create message from template
            message = alert.message_template.format(
                symbol=alert.symbol,
                trigger_value=trigger_value,
                **trigger_data
            )
            
            # Create triggered alert record
            triggered_alert = TriggeredAlert(
                alert_id=alert.id,
                symbol=alert.symbol,
                alert_type=alert.alert_type,
                trigger_value=trigger_value,
                trigger_data=trigger_data,
                message=message,
                priority=alert.priority,
                triggered_at=datetime.now()
            )
            
            # Update alert counters
            alert.triggered_count += 1
            alert.last_triggered = datetime.now()
            
            # Store triggered alert
            self.triggered_alerts.append(triggered_alert)
            
            # Send notifications
            self.notification_service.send_notification(
                triggered_alert, 
                alert.notification_methods
            )
            
            triggered_alert.notification_sent = True
            
            logger.info(f"🚨 Triggered alert {alert.id} for {alert.symbol}")
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.id}: {e}")
    
    def process_market_data(self, symbol: str, market_data: Dict):
        """Process incoming market data and check alerts"""
        symbol_alerts = [alert for alert in self.get_active_alerts() 
                        if alert.symbol == symbol or alert.symbol == '*']
        
        for alert in symbol_alerts:
            try:
                should_trigger, trigger_value = self.check_alert(alert, market_data)
                
                if should_trigger:
                    self.trigger_alert(alert, trigger_value, market_data)
                    
            except Exception as e:
                logger.error(f"Error processing alert {alert.id}: {e}")
    
    def start_monitoring(self):
        """Start the alert monitoring system"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("🚀 Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert monitoring system"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("⏹️ Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Cleanup expired alerts
                self._cleanup_expired_alerts()
                
                # Trim old triggered alerts (keep last 1000)
                if len(self.triggered_alerts) > 1000:
                    self.triggered_alerts = self.triggered_alerts[-1000:]
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(30)
    
    def _cleanup_expired_alerts(self):
        """Remove expired or inactive alerts"""
        expired_alerts = []
        
        for alert_id, alert in self.alerts.items():
            if not alert.is_active:
                expired_alerts.append(alert_id)
            elif alert.expires_at and datetime.now() > alert.expires_at:
                expired_alerts.append(alert_id)
                alert.is_active = False
            elif alert.triggered_count >= alert.max_triggers:
                expired_alerts.append(alert_id)
                alert.is_active = False
        
        for alert_id in expired_alerts:
            if alert_id in self.alerts:
                logger.info(f"🧹 Cleaning up expired alert {alert_id}")
                del self.alerts[alert_id]

class AlertPresets:
    """Predefined alert configurations"""
    
    @staticmethod
    def create_price_breakout_alert(symbol: str, support_price: float, resistance_price: float,
                                   notification_methods: List[NotificationMethod]) -> List[Dict]:
        """Create alerts for support/resistance breakouts"""
        alerts = []
        
        # Support breakout (bearish)
        alerts.append({
            'symbol': symbol,
            'alert_type': AlertType.PRICE_BELOW,
            'condition': {'target_price': support_price},
            'notification_methods': notification_methods,
            'priority': AlertPriority.HIGH,
            'message_template': f"{symbol} broke below support at ${support_price:.2f} (Current: ${{trigger_value:.2f}})"
        })
        
        # Resistance breakout (bullish)
        alerts.append({
            'symbol': symbol,
            'alert_type': AlertType.PRICE_ABOVE,
            'condition': {'target_price': resistance_price},
            'notification_methods': notification_methods,
            'priority': AlertPriority.HIGH,
            'message_template': f"{symbol} broke above resistance at ${resistance_price:.2f} (Current: ${{trigger_value:.2f}})"
        })
        
        return alerts
    
    @staticmethod
    def create_volatility_alert(symbol: str, volatility_threshold: float,
                               notification_methods: List[NotificationMethod]) -> Dict:
        """Create alert for high volatility"""
        return {
            'symbol': symbol,
            'alert_type': AlertType.PRICE_CHANGE,
            'condition': {
                'threshold_percent': volatility_threshold,
                'direction': 'both'
            },
            'notification_methods': notification_methods,
            'priority': AlertPriority.MEDIUM,
            'message_template': f"{symbol} high volatility: {{trigger_value:.1f}}% change"
        }
    
    @staticmethod
    def create_earnings_alert(symbol: str, notification_methods: List[NotificationMethod]) -> Dict:
        """Create alert for earnings announcements"""
        return {
            'symbol': symbol,
            'alert_type': AlertType.EARNINGS_ANNOUNCEMENT,
            'condition': {},
            'notification_methods': notification_methods,
            'priority': AlertPriority.HIGH,
            'message_template': f"{symbol} earnings announcement detected"
        }

# Global instances
def create_alert_system(config: Dict = None) -> Tuple[AlertManager, NotificationService]:
    """Create and initialize the alert system"""
    notification_service = NotificationService(config)
    alert_manager = AlertManager(notification_service)
    
    logger.info("✅ Alert system initialized")
    return alert_manager, notification_service