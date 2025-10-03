"""
Advanced Metrics Engine with Pathway UDFs
Calculates technical indicators, fundamental ratios, and portfolio metrics
"""

import pathway as pw
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathwayMetricsEngine:
    """Advanced metrics calculation using Pathway UDFs"""
    
    def __init__(self):
        self.setup_pathway_udfs()
        
    def setup_pathway_udfs(self):
        """Setup Pathway User Defined Functions for metrics"""
        
        # Technical Indicators UDFs
        @pw.udf
        def calculate_rsi(prices: pw.ColumnExpression, period: int = 14) -> float:
            """Calculate RSI using Pathway UDF"""
            try:
                if len(prices) < period + 1:
                    return 50.0
                    
                delta = np.diff(prices)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-period:])
                avg_loss = np.mean(loss[-period:])
                
                if avg_loss == 0:
                    return 100.0
                    
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return float(rsi)
            except:
                return 50.0
                
        @pw.udf
        def calculate_macd(prices: pw.ColumnExpression) -> Tuple[float, float, float]:
            """Calculate MACD, Signal, and Histogram"""
            try:
                if len(prices) < 35:
                    return (0.0, 0.0, 0.0)
                    
                exp1 = pd.Series(prices).ewm(span=12).mean()
                exp2 = pd.Series(prices).ewm(span=26).mean()
                macd_line = exp1 - exp2
                signal_line = macd_line.ewm(span=9).mean()
                histogram = macd_line - signal_line
                
                return (
                    float(macd_line.iloc[-1]),
                    float(signal_line.iloc[-1]),
                    float(histogram.iloc[-1])
                )
            except:
                return (0.0, 0.0, 0.0)
                
        @pw.udf
        def calculate_bollinger_bands(prices: pw.ColumnExpression, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
            """Calculate Bollinger Bands (Upper, Middle, Lower)"""
            try:
                if len(prices) < period:
                    current_price = prices[-1] if len(prices) > 0 else 0
                    return (current_price, current_price, current_price)
                    
                sma = np.mean(prices[-period:])
                std = np.std(prices[-period:])
                
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                
                return (float(upper), float(sma), float(lower))
            except:
                return (0.0, 0.0, 0.0)
                
        @pw.udf
        def calculate_stochastic(high: pw.ColumnExpression, low: pw.ColumnExpression, close: pw.ColumnExpression, k_period: int = 14) -> Tuple[float, float]:
            """Calculate Stochastic %K and %D"""
            try:
                if len(close) < k_period:
                    return (50.0, 50.0)
                    
                lowest_low = np.min(low[-k_period:])
                highest_high = np.max(high[-k_period:])
                current_close = close[-1]
                
                if highest_high == lowest_low:
                    k_percent = 50.0
                else:
                    k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                
                # Simple moving average for %D
                k_values = []
                for i in range(max(1, len(close) - 3), len(close)):
                    if i >= k_period:
                        ll = np.min(low[i-k_period:i])
                        hh = np.max(high[i-k_period:i])
                        cc = close[i-1]
                        if hh != ll:
                            k_val = ((cc - ll) / (hh - ll)) * 100
                            k_values.append(k_val)
                
                d_percent = np.mean(k_values) if k_values else k_percent
                
                return (float(k_percent), float(d_percent))
            except:
                return (50.0, 50.0)
        
        # Fundamental Analysis UDFs
        @pw.udf
        def calculate_pe_ratio(price: float, eps: float) -> float:
            """Calculate Price-to-Earnings ratio"""
            try:
                if eps <= 0:
                    return 0.0
                return float(price / eps)
            except:
                return 0.0
                
        @pw.udf
        def calculate_pb_ratio(price: float, book_value_per_share: float) -> float:
            """Calculate Price-to-Book ratio"""
            try:
                if book_value_per_share <= 0:
                    return 0.0
                return float(price / book_value_per_share)
            except:
                return 0.0
                
        @pw.udf
        def calculate_roe(net_income: float, shareholders_equity: float) -> float:
            """Calculate Return on Equity"""
            try:
                if shareholders_equity <= 0:
                    return 0.0
                return float((net_income / shareholders_equity) * 100)
            except:
                return 0.0
                
        @pw.udf
        def calculate_debt_to_equity(total_debt: float, shareholders_equity: float) -> float:
            """Calculate Debt-to-Equity ratio"""
            try:
                if shareholders_equity <= 0:
                    return 0.0
                return float(total_debt / shareholders_equity)
            except:
                return 0.0
        
        # Portfolio Metrics UDFs
        @pw.udf
        def calculate_portfolio_return(current_value: float, initial_value: float) -> float:
            """Calculate portfolio return percentage"""
            try:
                if initial_value <= 0:
                    return 0.0
                return float(((current_value - initial_value) / initial_value) * 100)
            except:
                return 0.0
                
        @pw.udf
        def calculate_sharpe_ratio(returns: pw.ColumnExpression, risk_free_rate: float = 0.02) -> float:
            """Calculate Sharpe ratio"""
            try:
                if len(returns) < 2:
                    return 0.0
                    
                excess_returns = np.array(returns) - risk_free_rate
                return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) != 0 else 0.0
            except:
                return 0.0
                
        @pw.udf
        def calculate_volatility(returns: pw.ColumnExpression) -> float:
            """Calculate volatility (standard deviation of returns)"""
            try:
                if len(returns) < 2:
                    return 0.0
                return float(np.std(returns) * np.sqrt(252))  # Annualized volatility
            except:
                return 0.0
                
        @pw.udf
        def calculate_max_drawdown(values: pw.ColumnExpression) -> float:
            """Calculate maximum drawdown"""
            try:
                if len(values) < 2:
                    return 0.0
                    
                peak = np.maximum.accumulate(values)
                drawdown = (values - peak) / peak
                return float(np.min(drawdown) * 100)
            except:
                return 0.0
        
        # Store UDFs for use
        self.udfs = {
            'rsi': calculate_rsi,
            'macd': calculate_macd,
            'bollinger': calculate_bollinger_bands,
            'stochastic': calculate_stochastic,
            'pe_ratio': calculate_pe_ratio,
            'pb_ratio': calculate_pb_ratio,
            'roe': calculate_roe,
            'debt_to_equity': calculate_debt_to_equity,
            'portfolio_return': calculate_portfolio_return,
            'sharpe_ratio': calculate_sharpe_ratio,
            'volatility': calculate_volatility,
            'max_drawdown': calculate_max_drawdown
        }
        
        logger.info("Pathway UDFs initialized")
        
    def calculate_comprehensive_metrics(self, symbol: str, data: Dict) -> Dict:
        """Calculate comprehensive metrics for a stock"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            info = ticker.info
            
            if hist.empty:
                return {}
            
            # Prepare data arrays
            prices = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            volumes = hist['Volume'].values
            
            metrics = {}
            
            # Technical Indicators
            if len(prices) >= 14:
                # RSI
                rsi = self.calculate_rsi_manual(prices)
                metrics['rsi'] = rsi
                
                # MACD
                macd_data = self.calculate_macd_manual(prices)
                metrics.update(macd_data)
                
                # Bollinger Bands
                bb_data = self.calculate_bollinger_manual(prices)
                metrics.update(bb_data)
                
                # Stochastic
                stoch_data = self.calculate_stochastic_manual(highs, lows, prices)
                metrics.update(stoch_data)
                
                # Moving Averages
                metrics['sma_20'] = float(np.mean(prices[-20:])) if len(prices) >= 20 else 0
                metrics['sma_50'] = float(np.mean(prices[-50:])) if len(prices) >= 50 else 0
                metrics['ema_12'] = float(pd.Series(prices).ewm(span=12).mean().iloc[-1])
                metrics['ema_26'] = float(pd.Series(prices).ewm(span=26).mean().iloc[-1])
            
            # Fundamental Ratios
            current_price = data.get('price', 0)
            
            # P/E Ratio
            trailing_pe = info.get('trailingPE', 0)
            forward_pe = info.get('forwardPE', 0)
            metrics['pe_ratio'] = trailing_pe if trailing_pe else forward_pe
            
            # P/B Ratio
            metrics['pb_ratio'] = info.get('priceToBook', 0)
            
            # Other fundamental metrics
            metrics['market_cap'] = info.get('marketCap', 0)
            metrics['enterprise_value'] = info.get('enterpriseValue', 0)
            metrics['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            metrics['beta'] = info.get('beta', 0)
            metrics['roe'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            metrics['debt_to_equity'] = info.get('debtToEquity', 0)
            metrics['current_ratio'] = info.get('currentRatio', 0)
            metrics['quick_ratio'] = info.get('quickRatio', 0)
            
            # Price metrics
            metrics['52_week_high'] = info.get('fiftyTwoWeekHigh', 0)
            metrics['52_week_low'] = info.get('fiftyTwoWeekLow', 0)
            metrics['price_to_52w_high'] = (current_price / metrics['52_week_high']) * 100 if metrics['52_week_high'] else 0
            metrics['price_to_52w_low'] = (current_price / metrics['52_week_low']) * 100 if metrics['52_week_low'] else 0
            
            # Volatility metrics
            returns = np.diff(prices) / prices[:-1]
            metrics['volatility_daily'] = float(np.std(returns)) if len(returns) > 1 else 0
            metrics['volatility_annualized'] = metrics['volatility_daily'] * np.sqrt(252)
            
            # Volume metrics
            metrics['avg_volume_10d'] = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else 0
            metrics['avg_volume_50d'] = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else 0
            metrics['volume_ratio'] = data.get('volume', 0) / metrics['avg_volume_50d'] if metrics['avg_volume_50d'] > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics for {symbol}: {e}")
            return {}
    
    def calculate_rsi_manual(self, prices: np.array, period: int = 14) -> float:
        """Manual RSI calculation"""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-period:])
            avg_loss = np.mean(loss[-period:])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except:
            return 50.0
    
    def calculate_macd_manual(self, prices: np.array) -> Dict:
        """Manual MACD calculation"""
        try:
            if len(prices) < 35:
                return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
                
            exp1 = pd.Series(prices).ewm(span=12).mean()
            exp2 = pd.Series(prices).ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]),
                'macd_signal': float(signal_line.iloc[-1]),
                'macd_histogram': float(histogram.iloc[-1])
            }
        except:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
    
    def calculate_bollinger_manual(self, prices: np.array, period: int = 20, std_dev: int = 2) -> Dict:
        """Manual Bollinger Bands calculation"""
        try:
            if len(prices) < period:
                current_price = prices[-1] if len(prices) > 0 else 0
                return {
                    'bollinger_upper': current_price,
                    'bollinger_middle': current_price,
                    'bollinger_lower': current_price
                }
                
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return {
                'bollinger_upper': float(upper),
                'bollinger_middle': float(sma),
                'bollinger_lower': float(lower)
            }
        except:
            return {'bollinger_upper': 0.0, 'bollinger_middle': 0.0, 'bollinger_lower': 0.0}
    
    def calculate_stochastic_manual(self, highs: np.array, lows: np.array, closes: np.array, k_period: int = 14) -> Dict:
        """Manual Stochastic calculation"""
        try:
            if len(closes) < k_period:
                return {'stoch_k': 50.0, 'stoch_d': 50.0}
                
            lowest_low = np.min(lows[-k_period:])
            highest_high = np.max(highs[-k_period:])
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                k_percent = 50.0
            else:
                k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Simple moving average for %D
            k_values = []
            for i in range(max(1, len(closes) - 3), len(closes)):
                if i >= k_period:
                    ll = np.min(lows[i-k_period:i])
                    hh = np.max(highs[i-k_period:i])
                    cc = closes[i-1]
                    if hh != ll:
                        k_val = ((cc - ll) / (hh - ll)) * 100
                        k_values.append(k_val)
            
            d_percent = np.mean(k_values) if k_values else k_percent
            
            return {
                'stoch_k': float(k_percent),
                'stoch_d': float(d_percent)
            }
        except:
            return {'stoch_k': 50.0, 'stoch_d': 50.0}

# Global metrics engine instance
metrics_engine = PathwayMetricsEngine()

def get_metrics_engine():
    """Get the global metrics engine instance"""
    return metrics_engine