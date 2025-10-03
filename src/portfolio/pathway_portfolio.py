"""
Real-time Portfolio Management using Pathway
Streams live market data and calculates metrics continuously
"""

import pathway as pw
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioStock:
    symbol: str
    shares: float
    purchase_price: float
    purchase_date: str

class PathwayPortfolioManager:
    """Real-time portfolio management using Pathway streaming"""
    
    def __init__(self):
        self.portfolio: Dict[str, PortfolioStock] = {}
        self.running = False
        self.data_cache = {}
        self.metrics_cache = {}
        self.last_update = {}
        
        # Initialize Pathway tables
        self.setup_pathway_tables()
        
    def setup_pathway_tables(self):
        """Setup Pathway tables and schemas"""
        try:
            # Define schemas for different data types
            class StockPriceSchema(pw.Schema):
                symbol: str
                price: float
                volume: int
                timestamp: int
                change: float
                change_percent: float
                
            class MetricsSchema(pw.Schema):
                symbol: str
                rsi: float
                macd: float
                bollinger_upper: float
                bollinger_lower: float
                pe_ratio: float
                portfolio_value: float
                timestamp: int
                
            self.price_schema = StockPriceSchema
            self.metrics_schema = MetricsSchema
            
            logger.info("Pathway schemas initialized")
            
        except Exception as e:
            logger.error(f"Error setting up Pathway tables: {e}")
            
    def add_stock(self, symbol: str, shares: float, purchase_price: float, purchase_date: str = None):
        """Add a stock to the portfolio"""
        if purchase_date is None:
            purchase_date = datetime.now().strftime('%Y-%m-%d')
            
        self.portfolio[symbol] = PortfolioStock(
            symbol=symbol,
            shares=shares,
            purchase_price=purchase_price,
            purchase_date=purchase_date
        )
        
        # Fetch initial real-time data for this stock
        self.fetch_real_time_data(symbol)
        
        logger.info(f"Added {symbol} to portfolio: {shares} shares at ${purchase_price}")
        return True
        
    def remove_stock(self, symbol: str):
        """Remove a stock from the portfolio"""
        if symbol in self.portfolio:
            del self.portfolio[symbol]
            if symbol in self.data_cache:
                del self.data_cache[symbol]
            logger.info(f"Removed {symbol} from portfolio")
            return True
        return False
        
    def get_portfolio_symbols(self) -> List[str]:
        """Get list of all portfolio symbols"""
        return list(self.portfolio.keys())
        
    def fetch_real_time_data(self, symbol: str) -> Dict:
        """Fetch real-time data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price data
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            volume = int(hist['Volume'].iloc[-1])
            
            change = current_price - prev_price
            change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # Get additional info
            info = ticker.info
            
            data = {
                'symbol': symbol,
                'price': current_price,
                'volume': volume,
                'change': change,
                'change_percent': change_percent,
                'timestamp': int(time.time()),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            self.data_cache[symbol] = data
            self.last_update[symbol] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """Calculate technical indicators using historical data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo", interval="1d")
            
            if len(hist) < 50:
                return {}
                
            # RSI calculation
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD calculation
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            bollinger_upper = sma + (std * 2)
            bollinger_lower = sma - (std * 2)
            
            # Moving averages
            sma_50 = hist['Close'].rolling(window=50).mean()
            sma_200 = hist['Close'].rolling(window=200).mean()
            
            indicators = {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 0,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
                'macd_signal': float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0,
                'bollinger_upper': float(bollinger_upper.iloc[-1]) if not pd.isna(bollinger_upper.iloc[-1]) else 0,
                'bollinger_lower': float(bollinger_lower.iloc[-1]) if not pd.isna(bollinger_lower.iloc[-1]) else 0,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else 0,
                'sma_200': float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else 0,
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}
            
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate overall portfolio metrics"""
        try:
            total_value = 0
            total_cost = 0
            portfolio_data = []
            
            for symbol, stock in self.portfolio.items():
                if symbol in self.data_cache:
                    current_price = self.data_cache[symbol]['price']
                    position_value = stock.shares * current_price
                    position_cost = stock.shares * stock.purchase_price
                    
                    total_value += position_value
                    total_cost += position_cost
                    
                    portfolio_data.append({
                        'symbol': symbol,
                        'shares': stock.shares,
                        'current_price': current_price,
                        'purchase_price': stock.purchase_price,
                        'position_value': position_value,
                        'position_cost': position_cost,
                        'unrealized_pnl': position_value - position_cost,
                        'unrealized_pnl_percent': ((position_value - position_cost) / position_cost) * 100 if position_cost > 0 else 0
                    })
            
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Calculate portfolio volatility (simplified)
            returns = []
            for stock_data in portfolio_data:
                if stock_data['position_cost'] > 0:
                    returns.append(stock_data['unrealized_pnl_percent'])
            
            volatility = np.std(returns) if returns else 0
            
            metrics = {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'volatility': volatility,
                'num_positions': len(self.portfolio),
                'positions': portfolio_data,
                'timestamp': int(time.time())
            }
            
            self.metrics_cache['portfolio'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
            
    def stream_data_for_symbol(self, symbol: str):
        """Stream data for a single symbol (simulated real-time)"""
        while self.running and symbol in self.portfolio:
            try:
                # Fetch real-time data
                data = self.fetch_real_time_data(symbol)
                if data:
                    # Calculate technical indicators
                    indicators = self.calculate_technical_indicators(symbol)
                    
                    # Combine data
                    combined_data = {**data, **indicators}
                    
                    # Store in cache
                    self.metrics_cache[symbol] = combined_data
                    
                    logger.info(f"Updated data for {symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")
                
                # Wait before next update (simulate real-time streaming)
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in streaming data for {symbol}: {e}")
                time.sleep(60)  # Wait longer on error
                
    def start_streaming(self):
        """Start real-time data streaming for all portfolio stocks"""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting portfolio streaming...")
        
        # Start streaming threads for each symbol
        with ThreadPoolExecutor(max_workers=10) as executor:
            for symbol in self.portfolio.keys():
                executor.submit(self.stream_data_for_symbol, symbol)
                
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.running = False
        logger.info("Stopped portfolio streaming")
        
    def get_stock_data(self, symbol: str) -> Dict:
        """Get latest data for a specific stock"""
        if symbol in self.metrics_cache:
            return self.metrics_cache[symbol]
        return {}
        
    def get_portfolio_data(self) -> Dict:
        """Get complete portfolio data"""
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'stocks': {symbol: self.get_stock_data(symbol) for symbol in self.portfolio.keys()},
            'last_update': max(self.last_update.values()) if self.last_update else datetime.now()
        }
        
    def get_alerts(self) -> List[Dict]:
        """Check for alerts based on current data"""
        alerts = []
        
        for symbol, stock in self.portfolio.items():
            if symbol in self.data_cache:
                current_data = self.data_cache[symbol]
                price = current_data['price']
                change_percent = current_data['change_percent']
                
                # Price movement alerts
                if abs(change_percent) > 5:
                    alerts.append({
                        'type': 'price_movement',
                        'symbol': symbol,
                        'message': f"{symbol} moved {change_percent:+.2f}% to ${price:.2f}",
                        'severity': 'high' if abs(change_percent) > 10 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Technical indicator alerts
                if symbol in self.metrics_cache:
                    metrics = self.metrics_cache[symbol]
                    rsi = metrics.get('rsi', 0)
                    
                    if rsi > 70:
                        alerts.append({
                            'type': 'technical',
                            'symbol': symbol,
                            'message': f"{symbol} RSI is overbought at {rsi:.1f}",
                            'severity': 'medium',
                            'timestamp': datetime.now().isoformat()
                        })
                    elif rsi < 30:
                        alerts.append({
                            'type': 'technical',
                            'symbol': symbol,
                            'message': f"{symbol} RSI is oversold at {rsi:.1f}",
                            'severity': 'medium',
                            'timestamp': datetime.now().isoformat()
                        })
        
        return alerts

# Global portfolio manager instance
portfolio_manager = PathwayPortfolioManager()

def get_portfolio_manager():
    """Get the global portfolio manager instance"""
    return portfolio_manager