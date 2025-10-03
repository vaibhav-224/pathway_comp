"""
Advanced AI Trading Signals
ML-based trading signal generation using technical indicators, sentiment analysis, and pattern recognition
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strength: str  # 'WEAK', 'MODERATE', 'STRONG'
    price_target: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str  # 'SHORT', 'MEDIUM', 'LONG'
    reasoning: List[str]
    technical_score: float
    sentiment_score: float
    ml_score: float
    timestamp: datetime

@dataclass
class MarketRegime:
    regime_type: str  # 'BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE'
    confidence: float
    volatility_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    trend_strength: float
    market_fear_greed: float  # 0-100 scale
    timestamp: datetime

class TechnicalAnalyzer:
    """Advanced technical analysis with pattern recognition"""
    
    def __init__(self):
        self.patterns = {
            'golden_cross': self._detect_golden_cross,
            'death_cross': self._detect_death_cross,
            'bullish_divergence': self._detect_bullish_divergence,
            'bearish_divergence': self._detect_bearish_divergence,
            'hammer': self._detect_hammer,
            'doji': self._detect_doji,
            'engulfing': self._detect_engulfing
        }
    
    def analyze_symbol(self, symbol: str, period: str = '3mo') -> Dict:
        """Perform comprehensive technical analysis"""
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if len(data) < 50:
                return {'error': 'Insufficient data'}
            
            # Calculate all technical indicators
            indicators = self._calculate_all_indicators(data)
            
            # Detect patterns
            patterns = self._detect_patterns(data)
            
            # Generate technical score
            tech_score = self._calculate_technical_score(indicators, patterns)
            
            return {
                'indicators': indicators,
                'patterns': patterns,
                'technical_score': tech_score,
                'support_resistance': self._find_support_resistance(data),
                'trend_analysis': self._analyze_trend(data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        try:
            # Trend indicators
            indicators['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50).iloc[-1]
            indicators['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26).iloc[-1]
            
            # Momentum indicators
            indicators['rsi'] = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
            indicators['stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close']).iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / data['Close'].iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume']).iloc[-1]
            indicators['mfi'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume']).iloc[-1]
            
            # Volatility
            indicators['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close']).iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _detect_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        patterns_found = {}
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                patterns_found[pattern_name] = pattern_func(data)
            except Exception as e:
                logger.warning(f"Error detecting {pattern_name}: {e}")
                patterns_found[pattern_name] = False
        
        return patterns_found
    
    def _detect_golden_cross(self, data: pd.DataFrame) -> bool:
        """Detect golden cross pattern (SMA50 crosses above SMA200)"""
        if len(data) < 200:
            return False
        
        sma50 = ta.trend.sma_indicator(data['Close'], window=50)
        sma200 = ta.trend.sma_indicator(data['Close'], window=200)
        
        # Check if 50 SMA recently crossed above 200 SMA
        recent_cross = (sma50.iloc[-1] > sma200.iloc[-1] and 
                       sma50.iloc[-5] <= sma200.iloc[-5])
        
        return recent_cross
    
    def _detect_death_cross(self, data: pd.DataFrame) -> bool:
        """Detect death cross pattern"""
        if len(data) < 200:
            return False
        
        sma50 = ta.trend.sma_indicator(data['Close'], window=50)
        sma200 = ta.trend.sma_indicator(data['Close'], window=200)
        
        recent_cross = (sma50.iloc[-1] < sma200.iloc[-1] and 
                       sma50.iloc[-5] >= sma200.iloc[-5])
        
        return recent_cross
    
    def _detect_bullish_divergence(self, data: pd.DataFrame) -> bool:
        """Detect bullish divergence between price and RSI"""
        if len(data) < 30:
            return False
        
        rsi = ta.momentum.rsi(data['Close'], window=14)
        
        # Simple divergence detection (price makes lower low, RSI makes higher low)
        price_recent = data['Close'].iloc[-10:]
        rsi_recent = rsi.iloc[-10:]
        
        price_min_idx = price_recent.idxmin()
        rsi_min_idx = rsi_recent.idxmin()
        
        # Check if price and RSI lows occurred at different times
        return price_min_idx != rsi_min_idx and rsi_recent.iloc[-1] > rsi_recent.min()
    
    def _detect_bearish_divergence(self, data: pd.DataFrame) -> bool:
        """Detect bearish divergence"""
        if len(data) < 30:
            return False
        
        rsi = ta.momentum.rsi(data['Close'], window=14)
        
        price_recent = data['Close'].iloc[-10:]
        rsi_recent = rsi.iloc[-10:]
        
        price_max_idx = price_recent.idxmax()
        rsi_max_idx = rsi_recent.idxmax()
        
        return price_max_idx != rsi_max_idx and rsi_recent.iloc[-1] < rsi_recent.max()
    
    def _detect_hammer(self, data: pd.DataFrame) -> bool:
        """Detect hammer candlestick pattern"""
        if len(data) < 2:
            return False
        
        recent = data.iloc[-1]
        body_size = abs(recent['Close'] - recent['Open'])
        lower_shadow = recent['Open'] - recent['Low'] if recent['Close'] > recent['Open'] else recent['Close'] - recent['Low']
        upper_shadow = recent['High'] - recent['Close'] if recent['Close'] > recent['Open'] else recent['High'] - recent['Open']
        
        # Hammer: small body, long lower shadow, little to no upper shadow
        return (lower_shadow > 2 * body_size and 
                upper_shadow < 0.5 * body_size and 
                body_size > 0)
    
    def _detect_doji(self, data: pd.DataFrame) -> bool:
        """Detect doji candlestick pattern"""
        if len(data) < 1:
            return False
        
        recent = data.iloc[-1]
        body_size = abs(recent['Close'] - recent['Open'])
        range_size = recent['High'] - recent['Low']
        
        # Doji: very small body relative to range
        return body_size < 0.1 * range_size if range_size > 0 else False
    
    def _detect_engulfing(self, data: pd.DataFrame) -> bool:
        """Detect engulfing pattern"""
        if len(data) < 2:
            return False
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        # Bullish engulfing
        prev_bearish = prev['Close'] < prev['Open']
        curr_bullish = curr['Close'] > curr['Open']
        engulfs = curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']
        
        return prev_bearish and curr_bullish and engulfs
    
    def _calculate_technical_score(self, indicators: Dict, patterns: Dict) -> float:
        """Calculate overall technical score (0-100)"""
        score = 50  # Neutral starting point
        
        try:
            # RSI scoring
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                score += 15  # Oversold = bullish
            elif rsi > 70:
                score -= 15  # Overbought = bearish
            
            # MACD scoring
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                score += 10
            else:
                score -= 10
            
            # Moving average scoring
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            if sma_20 > sma_50:
                score += 10
            else:
                score -= 10
            
            # Pattern scoring
            if patterns.get('golden_cross'):
                score += 20
            if patterns.get('death_cross'):
                score -= 20
            if patterns.get('bullish_divergence'):
                score += 15
            if patterns.get('bearish_divergence'):
                score -= 15
            if patterns.get('hammer'):
                score += 10
            if patterns.get('engulfing'):
                score += 10
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
        
        return max(0, min(100, score))
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        if len(data) < 20:
            return {'support': [], 'resistance': []}
        
        # Simple pivot point method
        highs = data['High'].rolling(window=5, center=True).max()
        lows = data['Low'].rolling(window=5, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(5, len(data) - 5):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        # Get most recent and significant levels
        current_price = data['Close'].iloc[-1]
        nearby_resistance = [r for r in resistance_levels if r > current_price][-3:]
        nearby_support = [s for s in support_levels if s < current_price][-3:]
        
        return {
            'support': sorted(nearby_support, reverse=True),
            'resistance': sorted(nearby_resistance)
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze overall trend"""
        if len(data) < 50:
            return {'trend': 'UNKNOWN', 'strength': 0}
        
        # Calculate trend using multiple timeframes
        sma_20 = ta.trend.sma_indicator(data['Close'], window=20)
        sma_50 = ta.trend.sma_indicator(data['Close'], window=50)
        
        current_price = data['Close'].iloc[-1]
        trend_score = 0
        
        # Price vs moving averages
        if current_price > sma_20.iloc[-1]:
            trend_score += 1
        if current_price > sma_50.iloc[-1]:
            trend_score += 1
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_score += 1
        
        # Recent price momentum
        price_change_20d = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
        if price_change_20d > 0.05:
            trend_score += 1
        elif price_change_20d < -0.05:
            trend_score -= 1
        
        # Determine trend
        if trend_score >= 3:
            trend = 'BULLISH'
        elif trend_score <= -1:
            trend = 'BEARISH'
        else:
            trend = 'SIDEWAYS'
        
        strength = abs(trend_score) / 4.0
        
        return {
            'trend': trend,
            'strength': strength,
            'score': trend_score
        }

class MLTradingModel:
    """Machine Learning model for trading signals"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = pd.DataFrame()
        
        # Price features
        features['returns_1d'] = data['Close'].pct_change()
        features['returns_5d'] = data['Close'].pct_change(5)
        features['returns_20d'] = data['Close'].pct_change(20)
        
        # Technical indicators
        features['rsi'] = ta.momentum.rsi(data['Close'])
        features['macd'] = ta.trend.MACD(data['Close']).macd()
        features['bb_position'] = (data['Close'] - ta.volatility.BollingerBands(data['Close']).bollinger_lband()) / (ta.volatility.BollingerBands(data['Close']).bollinger_hband() - ta.volatility.BollingerBands(data['Close']).bollinger_lband())
        
        # Volume features
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Volatility features
        features['volatility_20d'] = data['Close'].rolling(20).std()
        
        # Price position features
        features['price_vs_sma20'] = data['Close'] / ta.trend.sma_indicator(data['Close'], 20)
        features['price_vs_sma50'] = data['Close'] / ta.trend.sma_indicator(data['Close'], 50)
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_labels(self, data: pd.DataFrame, future_days: int = 5) -> pd.Series:
        """Create labels for training (future returns)"""
        future_returns = data['Close'].shift(-future_days) / data['Close'] - 1
        
        # Convert to categorical labels
        labels = pd.Series(index=data.index, dtype='int')
        labels[future_returns > 0.02] = 2  # Strong buy
        labels[(future_returns > 0.005) & (future_returns <= 0.02)] = 1  # Buy
        labels[(future_returns >= -0.005) & (future_returns <= 0.005)] = 0  # Hold
        labels[(future_returns >= -0.02) & (future_returns < -0.005)] = -1  # Sell
        labels[future_returns < -0.02] = -2  # Strong sell
        
        return labels
    
    def train_model(self, symbols: List[str], period: str = '2y'):
        """Train the ML model on historical data"""
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if len(data) < 100:
                    continue
                
                features = self.prepare_features(data)
                labels = self.create_labels(data)
                
                # Remove NaN values
                valid_idx = ~(features.isna().any(axis=1) | labels.isna())
                features_clean = features[valid_idx]
                labels_clean = labels[valid_idx]
                
                if len(features_clean) > 50:
                    all_features.append(features_clean)
                    all_labels.append(labels_clean)
                    
            except Exception as e:
                logger.warning(f"Error training on {symbol}: {e}")
        
        if not all_features:
            logger.error("No valid training data found")
            return False
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"✅ ML model trained on {len(X)} samples with {len(self.feature_names)} features")
        return True
    
    def predict_signal(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Predict trading signal for given data"""
        if not self.is_trained:
            return 0, 0.5
        
        try:
            features = self.prepare_features(data)
            
            if features.empty or len(features) < 50:
                return 0, 0.5
            
            # Use last row for prediction
            X = features.iloc[-1:][self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error predicting signal: {e}")
            return 0, 0.5

class AITradingSignalGenerator:
    """Main AI trading signal generator"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_model = MLTradingModel()
        self.sentiment_weights = {
            'news': 0.3,
            'social': 0.2,
            'technical': 0.4,
            'ml': 0.3
        }
    
    def initialize(self, symbols: List[str] = None):
        """Initialize the AI signal generator"""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY']
        
        logger.info("🤖 Training ML model for trading signals...")
        success = self.ml_model.train_model(symbols)
        
        if success:
            logger.info("✅ AI Trading Signal Generator initialized successfully")
        else:
            logger.warning("⚠️ ML model training failed, will use technical analysis only")
    
    def generate_signal(self, symbol: str, sentiment_data: Dict = None) -> TradingSignal:
        """Generate comprehensive trading signal"""
        try:
            # Get technical analysis
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol)
            
            if 'error' in technical_analysis:
                return self._create_neutral_signal(symbol, "Insufficient data for analysis")
            
            # Get ML prediction
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo')
            ml_prediction, ml_confidence = self.ml_model.predict_signal(data)
            
            # Combine scores
            technical_score = technical_analysis['technical_score']
            sentiment_score = self._calculate_sentiment_score(sentiment_data) if sentiment_data else 50
            ml_score = self._convert_ml_prediction_to_score(ml_prediction)
            
            # Weighted final score
            final_score = (
                technical_score * self.sentiment_weights['technical'] +
                sentiment_score * self.sentiment_weights['news'] +
                ml_score * self.sentiment_weights['ml']
            ) / (self.sentiment_weights['technical'] + self.sentiment_weights['news'] + self.sentiment_weights['ml'])
            
            # Generate signal
            signal_type, confidence, strength = self._score_to_signal(final_score, ml_confidence)
            
            # Calculate targets
            current_price = data['Close'].iloc[-1]
            atr = technical_analysis['indicators'].get('atr', current_price * 0.02)
            
            price_target = None
            stop_loss = None
            
            if signal_type == 'BUY':
                price_target = current_price * (1 + 0.05 + (confidence - 0.5) * 0.1)
                stop_loss = current_price * (1 - 0.03)
            elif signal_type == 'SELL':
                price_target = current_price * (1 - 0.05 - (confidence - 0.5) * 0.1)
                stop_loss = current_price * (1 + 0.03)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(technical_analysis, ml_prediction, sentiment_score)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                price_target=price_target,
                stop_loss=stop_loss,
                time_horizon=self._determine_time_horizon(technical_analysis),
                reasoning=reasoning,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                ml_score=ml_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._create_neutral_signal(symbol, f"Error: {str(e)}")
    
    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """Calculate sentiment score from news/social data"""
        if not sentiment_data:
            return 50.0
        
        # Simple sentiment scoring
        avg_sentiment = sentiment_data.get('average_sentiment', 0.5)
        return avg_sentiment * 100
    
    def _convert_ml_prediction_to_score(self, ml_prediction: int) -> float:
        """Convert ML prediction to 0-100 score"""
        # ML predictions: -2 (strong sell) to 2 (strong buy)
        return ((ml_prediction + 2) / 4) * 100
    
    def _score_to_signal(self, score: float, confidence: float) -> Tuple[str, float, str]:
        """Convert score to trading signal"""
        if score >= 70:
            signal_type = 'BUY'
            strength = 'STRONG' if score >= 80 else 'MODERATE'
        elif score <= 30:
            signal_type = 'SELL'
            strength = 'STRONG' if score <= 20 else 'MODERATE'
        else:
            signal_type = 'HOLD'
            strength = 'WEAK'
        
        # Adjust confidence based on how extreme the score is
        adjusted_confidence = confidence * (abs(score - 50) / 50)
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
        
        return signal_type, adjusted_confidence, strength
    
    def _determine_time_horizon(self, technical_analysis: Dict) -> str:
        """Determine appropriate time horizon for signal"""
        trend = technical_analysis.get('trend_analysis', {})
        trend_strength = trend.get('strength', 0)
        
        if trend_strength > 0.7:
            return 'LONG'
        elif trend_strength > 0.4:
            return 'MEDIUM'
        else:
            return 'SHORT'
    
    def _generate_reasoning(self, technical_analysis: Dict, ml_prediction: int, sentiment_score: float) -> List[str]:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        
        # Technical reasoning
        tech_score = technical_analysis['technical_score']
        if tech_score > 70:
            reasoning.append("Strong bullish technical indicators")
        elif tech_score < 30:
            reasoning.append("Strong bearish technical indicators")
        
        patterns = technical_analysis.get('patterns', {})
        if patterns.get('golden_cross'):
            reasoning.append("Golden cross pattern detected")
        if patterns.get('death_cross'):
            reasoning.append("Death cross pattern detected")
        if patterns.get('bullish_divergence'):
            reasoning.append("Bullish divergence in momentum indicators")
        if patterns.get('bearish_divergence'):
            reasoning.append("Bearish divergence in momentum indicators")
        
        # ML reasoning
        if ml_prediction >= 1:
            reasoning.append("ML model predicts positive price movement")
        elif ml_prediction <= -1:
            reasoning.append("ML model predicts negative price movement")
        
        # Sentiment reasoning
        if sentiment_score > 70:
            reasoning.append("Positive news sentiment")
        elif sentiment_score < 30:
            reasoning.append("Negative news sentiment")
        
        # Trend reasoning
        trend = technical_analysis.get('trend_analysis', {})
        if trend.get('trend') == 'BULLISH':
            reasoning.append("Strong upward trend confirmed")
        elif trend.get('trend') == 'BEARISH':
            reasoning.append("Strong downward trend confirmed")
        
        return reasoning if reasoning else ["Mixed signals, proceed with caution"]
    
    def _create_neutral_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a neutral/hold signal"""
        return TradingSignal(
            symbol=symbol,
            signal_type='HOLD',
            confidence=0.5,
            strength='WEAK',
            price_target=None,
            stop_loss=None,
            time_horizon='SHORT',
            reasoning=[reason],
            technical_score=50.0,
            sentiment_score=50.0,
            ml_score=50.0,
            timestamp=datetime.now()
        )

# Global instance
ai_signal_generator = AITradingSignalGenerator()

def initialize_ai_signals(symbols: List[str] = None):
    """Initialize the AI signal generator"""
    ai_signal_generator.initialize(symbols)
    return ai_signal_generator