"""
Configuration settings for FinanceAI LiveStream
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Data Sources
    YAHOO_FINANCE_ENABLED: bool = True
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    
    # AI/LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Market Data Settings
    DEFAULT_SYMBOLS: List[str] = None
    UPDATE_INTERVAL_SECONDS: int = 5
    
    # Anomaly Detection
    ANOMALY_THRESHOLD_SIGMA: float = 2.5
    VOLUME_SPIKE_THRESHOLD: float = 3.0
    PRICE_CHANGE_THRESHOLD: float = 0.05  # 5%
    
    # Dashboard Settings
    DASHBOARD_PORT: int = 8501
    DASHBOARD_HOST: str = "localhost"
    
    # Alerts
    ALERT_EMAIL_ENABLED: bool = False
    ALERT_EMAIL_RECIPIENT: str = os.getenv("ALERT_EMAIL", "")
    
    def __post_init__(self):
        if self.DEFAULT_SYMBOLS is None:
            self.DEFAULT_SYMBOLS = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Tech
                "JPM", "BAC", "GS",                        # Finance
                "SPY", "QQQ", "IWM"                        # ETFs
            ]

# Global config instance
config = Config()