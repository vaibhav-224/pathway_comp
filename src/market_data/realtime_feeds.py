"""
Real-time Market Data Feed Integration
Provides live stock prices, technical indicators, and WebSocket connections
"""

import asyncio
import websockets
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class StockPrice:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    
@dataclass
class TechnicalIndicators:
    symbol: str
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    sma_20: float
    sma_50: float
    volume_sma: float
    timestamp: datetime

class RealTimeMarketFeed:
    """Real-time market data feed with multiple data sources"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY']
        self.price_data = {}
        self.technical_data = {}
        self.subscribers = []
        self.running = False
        self.update_thread = None
        
    def subscribe(self, callback: Callable):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, data_type: str, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def start_feed(self):
        """Start the real-time data feed"""
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("🚀 Real-time market feed started")
    
    def stop_feed(self):
        """Stop the real-time data feed"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("⏹️ Real-time market feed stopped")
    
    def _update_loop(self):
        """Main update loop for market data"""
        while self.running:
            try:
                # Update stock prices
                self._update_stock_prices()
                
                # Update technical indicators (less frequently)
                if int(time.time()) % 60 == 0:  # Every minute
                    self._update_technical_indicators()
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)
    
    def _update_stock_prices(self):
        """Update real-time stock prices"""
        try:
            # Use yfinance for real-time data
            tickers = yf.Tickers(' '.join(self.symbols))
            
            for symbol in self.symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period='2d', interval='1m')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose', current_price)
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        stock_price = StockPrice(
                            symbol=symbol,
                            price=float(current_price),
                            change=float(change),
                            change_percent=float(change_percent),
                            volume=int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                            timestamp=datetime.now(),
                            market_cap=info.get('marketCap'),
                            pe_ratio=info.get('trailingPE')
                        )
                        
                        self.price_data[symbol] = stock_price
                        self._notify_subscribers('price_update', {symbol: stock_price})
                        
                except Exception as e:
                    logger.warning(f"Error updating price for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating stock prices: {e}")
    
    def _update_technical_indicators(self):
        """Update technical indicators"""
        try:
            for symbol in self.symbols:
                try:
                    # Get historical data for calculations
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='60d', interval='1d')
                    
                    if len(hist) < 50:
                        continue
                    
                    # Calculate technical indicators
                    indicators = self._calculate_technical_indicators(hist)
                    indicators.symbol = symbol
                    indicators.timestamp = datetime.now()
                    
                    self.technical_data[symbol] = indicators
                    self._notify_subscribers('technical_update', {symbol: indicators})
                    
                except Exception as e:
                    logger.warning(f"Error calculating indicators for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating technical indicators: {e}")
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate technical indicators from price data"""
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        volumes = data['Volume']
        
        # RSI calculation
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = closes.ewm(span=12).mean()
        exp2 = closes.ewm(span=26).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = closes.rolling(window=20).mean()
        std_20 = closes.rolling(window=20).std()
        bollinger_upper = sma_20 + (std_20 * 2)
        bollinger_lower = sma_20 - (std_20 * 2)
        
        # Simple Moving Averages
        sma_50 = closes.rolling(window=50).mean()
        volume_sma = volumes.rolling(window=20).mean()
        
        return TechnicalIndicators(
            symbol="",  # Will be set by caller
            rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            macd=float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            macd_signal=float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            bollinger_upper=float(bollinger_upper.iloc[-1]) if not pd.isna(bollinger_upper.iloc[-1]) else 0.0,
            bollinger_lower=float(bollinger_lower.iloc[-1]) if not pd.isna(bollinger_lower.iloc[-1]) else 0.0,
            sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0.0,
            sma_50=float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else 0.0,
            volume_sma=float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else 0.0,
            timestamp=datetime.now()
        )
    
    def get_current_prices(self) -> Dict[str, StockPrice]:
        """Get current price data for all symbols"""
        return self.price_data.copy()
    
    def get_technical_indicators(self) -> Dict[str, TechnicalIndicators]:
        """Get current technical indicators for all symbols"""
        return self.technical_data.copy()
    
    def get_symbol_data(self, symbol: str) -> Dict:
        """Get complete data for a specific symbol"""
        return {
            'price': self.price_data.get(symbol),
            'technical': self.technical_data.get(symbol)
        }

class PriceAlertManager:
    """Manage price alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.triggered_alerts = []
    
    def add_alert(self, symbol: str, condition: str, target_price: float, 
                  alert_type: str = 'price', notification_method: str = 'log'):
        """Add a price alert"""
        alert = {
            'id': len(self.alerts),
            'symbol': symbol,
            'condition': condition,  # 'above', 'below'
            'target_price': target_price,
            'alert_type': alert_type,
            'notification_method': notification_method,
            'created_at': datetime.now(),
            'triggered': False
        }
        self.alerts.append(alert)
        return alert['id']
    
    def check_alerts(self, symbol: str, current_price: float):
        """Check if any alerts should be triggered"""
        for alert in self.alerts:
            if alert['symbol'] == symbol and not alert['triggered']:
                should_trigger = False
                
                if alert['condition'] == 'above' and current_price >= alert['target_price']:
                    should_trigger = True
                elif alert['condition'] == 'below' and current_price <= alert['target_price']:
                    should_trigger = True
                
                if should_trigger:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now()
                    alert['triggered_price'] = current_price
                    self.triggered_alerts.append(alert)
                    self._send_notification(alert)
    
    def _send_notification(self, alert: Dict):
        """Send notification for triggered alert"""
        message = f"🚨 PRICE ALERT: {alert['symbol']} is {alert['condition']} ${alert['target_price']:.2f} (Current: ${alert['triggered_price']:.2f})"
        logger.info(message)
        # Here you could add email, SMS, Slack notifications
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert for alert in self.alerts if not alert['triggered']]
    
    def get_triggered_alerts(self) -> List[Dict]:
        """Get all triggered alerts"""
        return self.triggered_alerts

# Global instances
market_feed = RealTimeMarketFeed()
alert_manager = PriceAlertManager()

def initialize_market_feed(symbols: List[str] = None):
    """Initialize and start the market feed"""
    global market_feed, alert_manager
    
    if symbols:
        market_feed.symbols = symbols
    
    # Subscribe alert manager to price updates
    def alert_callback(data_type: str, data: Dict):
        if data_type == 'price_update':
            for symbol, price_data in data.items():
                alert_manager.check_alerts(symbol, price_data.price)
    
    market_feed.subscribe(alert_callback)
    market_feed.start_feed()
    
    return market_feed, alert_manager