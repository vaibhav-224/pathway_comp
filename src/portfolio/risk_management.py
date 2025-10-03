"""
Portfolio Risk Management
Advanced portfolio optimization, risk metrics, and rebalancing suggestions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    total_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    beta: float
    alpha: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    timestamp: datetime

@dataclass 
class RiskMetrics:
    symbol: str
    weight: float
    var_contribution: float
    beta: float
    correlation_to_portfolio: float
    concentration_risk: float
    liquidity_risk: str  # LOW, MEDIUM, HIGH
    sector_exposure: str
    market_cap: str  # LARGE, MID, SMALL
    
@dataclass
class RebalanceRecommendation:
    symbol: str
    current_weight: float
    target_weight: float
    action: str  # BUY, SELL, HOLD
    shares_to_trade: int
    dollar_amount: float
    reason: str
    priority: str  # HIGH, MEDIUM, LOW

class PortfolioOptimizer:
    """Modern Portfolio Theory based optimizer"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.lookback_days = 252  # 1 year of trading days
    
    def optimize_portfolio(self, symbols: List[str], method: str = 'max_sharpe', 
                          constraints: Dict = None) -> Dict:
        """Optimize portfolio allocation using various methods"""
        try:
            # Get historical data
            returns_data = self._get_returns_data(symbols)
            
            if returns_data.empty:
                return {'error': 'Unable to fetch historical data'}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Set up constraints
            n_assets = len(symbols)
            bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # No short selling
            constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            
            # Add custom constraints if provided
            if constraints:
                if 'max_weight' in constraints:
                    max_weight = constraints['max_weight']
                    for i in range(n_assets):
                        constraints_list.append({
                            'type': 'ineq', 
                            'fun': lambda x, i=i: max_weight - x[i]
                        })
                
                if 'min_weight' in constraints:
                    min_weight = constraints['min_weight']
                    for i in range(n_assets):
                        constraints_list.append({
                            'type': 'ineq', 
                            'fun': lambda x, i=i: x[i] - min_weight
                        })
            
            # Optimize based on method
            if method == 'max_sharpe':
                result = self._maximize_sharpe_ratio(expected_returns, cov_matrix, bounds, constraints_list)
            elif method == 'min_volatility':
                result = self._minimize_volatility(cov_matrix, bounds, constraints_list)
            elif method == 'max_return':
                result = self._maximize_return(expected_returns, bounds, constraints_list)
            elif method == 'equal_weight':
                result = {'x': np.array([1/n_assets] * n_assets), 'success': True}
            else:
                return {'error': f'Unknown optimization method: {method}'}
            
            if not result['success']:
                return {'error': 'Optimization failed to converge'}
            
            # Calculate portfolio metrics
            weights = result['x']
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Create allocation dictionary
            allocation = {}
            for i, symbol in enumerate(symbols):
                if weights[i] > 0.01:  # Only include weights > 1%
                    allocation[symbol] = {
                        'weight': float(weights[i]),
                        'expected_return': float(expected_returns.iloc[i]),
                        'volatility': float(np.sqrt(cov_matrix.iloc[i, i]))
                    }
            
            return {
                'allocation': allocation,
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'method': method,
                'total_weight': float(np.sum(weights))
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {'error': str(e)}
    
    def _get_returns_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get historical returns data for symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 50)
            
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            
            # Calculate daily returns
            returns = data.pct_change().dropna()
            
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()
    
    def _maximize_sharpe_ratio(self, expected_returns, cov_matrix, bounds, constraints):
        """Maximize Sharpe ratio"""
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        return minimize(objective, initial_guess, method='SLSQP', 
                       bounds=bounds, constraints=constraints)
    
    def _minimize_volatility(self, cov_matrix, bounds, constraints):
        """Minimize portfolio volatility"""
        n_assets = len(cov_matrix)
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        initial_guess = np.array([1/n_assets] * n_assets)
        
        return minimize(objective, initial_guess, method='SLSQP',
                       bounds=bounds, constraints=constraints)
    
    def _maximize_return(self, expected_returns, bounds, constraints):
        """Maximize expected return"""
        n_assets = len(expected_returns)
        
        def objective(weights):
            return -np.sum(weights * expected_returns)
        
        initial_guess = np.array([1/n_assets] * n_assets)
        
        return minimize(objective, initial_guess, method='SLSQP',
                       bounds=bounds, constraints=constraints)

class RiskAnalyzer:
    """Comprehensive risk analysis for portfolios"""
    
    def __init__(self):
        self.benchmark_symbol = 'SPY'  # S&P 500 as benchmark
    
    def calculate_portfolio_metrics(self, portfolio: Dict, period: str = '1y') -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            symbols = list(portfolio.keys())
            weights = [portfolio[symbol]['weight'] for symbol in symbols]
            
            # Get historical data
            returns_data = self._get_portfolio_returns(symbols, weights, period)
            benchmark_returns = self._get_benchmark_returns(period)
            
            if returns_data.empty or benchmark_returns.empty:
                return self._create_default_metrics()
            
            # Calculate metrics
            total_return = (1 + returns_data).prod() - 1
            annualized_return = (1 + returns_data.mean()) ** 252 - 1
            volatility = returns_data.std() * np.sqrt(252)
            
            # Risk metrics
            var_95 = returns_data.quantile(0.05)
            var_99 = returns_data.quantile(0.01)
            max_drawdown = self._calculate_max_drawdown(returns_data)
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility != 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns_data)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Alpha and Beta
            beta, alpha = self._calculate_alpha_beta(returns_data, benchmark_returns)
            
            # Information ratio
            active_returns = returns_data - benchmark_returns
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() != 0 else 0
            
            return PortfolioMetrics(
                total_value=sum(portfolio[s].get('value', 0) for s in symbols),
                total_return=float(total_return),
                annualized_return=float(annualized_return),
                volatility=float(volatility),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                var_95=float(var_95),
                var_99=float(var_99),
                beta=float(beta),
                alpha=float(alpha),
                information_ratio=float(information_ratio),
                calmar_ratio=float(calmar_ratio),
                sortino_ratio=float(sortino_ratio),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return self._create_default_metrics()
    
    def analyze_risk_contributions(self, portfolio: Dict) -> List[RiskMetrics]:
        """Analyze risk contribution of each holding"""
        risk_metrics = []
        
        try:
            symbols = list(portfolio.keys())
            weights = np.array([portfolio[symbol]['weight'] for symbol in symbols])
            
            # Get returns and covariance
            returns_data = self._get_returns_data(symbols)
            if returns_data.empty:
                return risk_metrics
            
            cov_matrix = returns_data.cov() * 252
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            for i, symbol in enumerate(symbols):
                try:
                    # Risk contribution
                    marginal_contrib = np.dot(cov_matrix, weights)[i]
                    risk_contrib = weights[i] * marginal_contrib / portfolio_var
                    
                    # Beta calculation
                    symbol_returns = returns_data[symbol]
                    market_returns = self._get_benchmark_returns('1y')
                    beta = self._calculate_symbol_beta(symbol_returns, market_returns)
                    
                    # Correlation to portfolio
                    portfolio_returns = (returns_data * weights).sum(axis=1)
                    correlation = symbol_returns.corr(portfolio_returns)
                    
                    # Get additional risk metrics
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    risk_metrics.append(RiskMetrics(
                        symbol=symbol,
                        weight=float(weights[i]),
                        var_contribution=float(risk_contrib),
                        beta=float(beta),
                        correlation_to_portfolio=float(correlation) if not pd.isna(correlation) else 0.0,
                        concentration_risk=self._assess_concentration_risk(weights[i]),
                        liquidity_risk=self._assess_liquidity_risk(info.get('averageVolume', 0)),
                        sector_exposure=info.get('sector', 'Unknown'),
                        market_cap=self._classify_market_cap(info.get('marketCap', 0))
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error analyzing risk for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error in risk contribution analysis: {e}")
        
        return risk_metrics
    
    def _get_portfolio_returns(self, symbols: List[str], weights: List[float], period: str) -> pd.Series:
        """Get portfolio returns based on weights"""
        returns_data = self._get_returns_data(symbols, period)
        if returns_data.empty:
            return pd.Series()
        
        portfolio_returns = (returns_data * weights).sum(axis=1)
        return portfolio_returns
    
    def _get_returns_data(self, symbols: List[str], period: str = '1y') -> pd.DataFrame:
        """Get returns data for symbols"""
        try:
            # Convert period to days
            days_map = {'1y': 365, '2y': 730, '3y': 1095, '6m': 180, '3m': 90}
            days = days_map.get(period, 365)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)
            
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            
            returns = data.pct_change().dropna()
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()
    
    def _get_benchmark_returns(self, period: str = '1y') -> pd.Series:
        """Get benchmark returns"""
        try:
            ticker = yf.Ticker(self.benchmark_symbol)
            
            days_map = {'1y': 365, '2y': 730, '3y': 1095, '6m': 180, '3m': 90}
            days = days_map.get(period, 365)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)
            
            data = ticker.history(start=start_date, end=end_date)['Close']
            returns = data.pct_change().dropna()
            
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching benchmark returns: {e}")
            return pd.Series()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - target_return / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        return (returns.mean() * 252 - target_return) / downside_deviation if downside_deviation != 0 else 0.0
    
    def _calculate_alpha_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark"""
        try:
            # Align series
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 30:
                return 0.0, 1.0
            
            portfolio_ret = aligned_data.iloc[:, 0]
            benchmark_ret = aligned_data.iloc[:, 1]
            
            # Calculate beta using linear regression
            beta, alpha_daily, _, _, _ = stats.linregress(benchmark_ret, portfolio_ret)
            alpha_annualized = alpha_daily * 252
            
            return beta, alpha_annualized
            
        except Exception as e:
            logger.error(f"Error calculating alpha/beta: {e}")
            return 0.0, 1.0
    
    def _calculate_symbol_beta(self, symbol_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta for individual symbol"""
        try:
            aligned_data = pd.concat([symbol_returns, market_returns], axis=1, join='inner').dropna()
            
            if len(aligned_data) < 30:
                return 1.0
            
            symbol_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            beta = symbol_ret.cov(market_ret) / market_ret.var()
            return float(beta) if not pd.isna(beta) else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating symbol beta: {e}")
            return 1.0
    
    def _assess_concentration_risk(self, weight: float) -> float:
        """Assess concentration risk based on position size"""
        if weight > 0.3:
            return 5.0  # High concentration risk
        elif weight > 0.15:
            return 3.0  # Medium concentration risk
        elif weight > 0.05:
            return 1.0  # Low concentration risk
        else:
            return 0.0  # Negligible concentration risk
    
    def _assess_liquidity_risk(self, avg_volume: int) -> str:
        """Assess liquidity risk based on trading volume"""
        if avg_volume > 1_000_000:
            return 'LOW'
        elif avg_volume > 100_000:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _classify_market_cap(self, market_cap: int) -> str:
        """Classify market cap size"""
        if market_cap > 10_000_000_000:  # > $10B
            return 'LARGE'
        elif market_cap > 2_000_000_000:  # > $2B
            return 'MID'
        else:
            return 'SMALL'
    
    def _create_default_metrics(self) -> PortfolioMetrics:
        """Create default metrics when calculation fails"""
        return PortfolioMetrics(
            total_value=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            var_99=0.0,
            beta=1.0,
            alpha=0.0,
            information_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            timestamp=datetime.now()
        )

class RebalancingEngine:
    """Portfolio rebalancing recommendations"""
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
        self.transaction_cost = 0.001  # 0.1% transaction cost
    
    def generate_rebalancing_recommendations(self, current_portfolio: Dict, 
                                           target_allocation: Dict,
                                           total_value: float) -> List[RebalanceRecommendation]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        try:
            # Get current prices
            all_symbols = set(list(current_portfolio.keys()) + list(target_allocation.keys()))
            current_prices = self._get_current_prices(list(all_symbols))
            
            for symbol in all_symbols:
                current_weight = current_portfolio.get(symbol, {}).get('weight', 0.0)
                target_weight = target_allocation.get(symbol, {}).get('weight', 0.0)
                
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) < 0.01:  # Less than 1% difference
                    continue
                
                current_price = current_prices.get(symbol, 0)
                if current_price == 0:
                    continue
                
                # Calculate dollar amounts
                target_dollar_value = target_weight * total_value
                current_dollar_value = current_weight * total_value
                dollar_difference = target_dollar_value - current_dollar_value
                
                # Calculate shares to trade
                shares_to_trade = int(dollar_difference / current_price)
                
                # Determine action
                if weight_diff > 0.01:
                    action = 'BUY'
                    priority = self._determine_priority(weight_diff, 'BUY')
                elif weight_diff < -0.01:
                    action = 'SELL'
                    priority = self._determine_priority(abs(weight_diff), 'SELL')
                else:
                    action = 'HOLD'
                    priority = 'LOW'
                
                # Generate reason
                reason = self._generate_rebalance_reason(current_weight, target_weight, action)
                
                recommendations.append(RebalanceRecommendation(
                    symbol=symbol,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    action=action,
                    shares_to_trade=abs(shares_to_trade),
                    dollar_amount=abs(dollar_difference),
                    reason=reason,
                    priority=priority
                ))
            
            # Sort by priority and dollar amount
            priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            recommendations.sort(key=lambda x: (priority_order.get(x.priority, 1), x.dollar_amount), reverse=True)
            
        except Exception as e:
            logger.error(f"Error generating rebalancing recommendations: {e}")
        
        return recommendations
    
    def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        prices = {}
        
        try:
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        prices[symbol] = float(hist['Close'].iloc[-1])
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {e}")
                    prices[symbol] = 0.0
        
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
        
        return prices
    
    def _determine_priority(self, weight_diff: float, action: str) -> str:
        """Determine rebalancing priority"""
        if weight_diff > 0.1:  # > 10% difference
            return 'HIGH'
        elif weight_diff > 0.05:  # > 5% difference
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_rebalance_reason(self, current_weight: float, target_weight: float, action: str) -> str:
        """Generate human-readable reason for rebalancing"""
        weight_diff = abs(target_weight - current_weight)
        
        if action == 'BUY':
            return f"Increase allocation from {current_weight:.1%} to {target_weight:.1%} (underweight by {weight_diff:.1%})"
        elif action == 'SELL':
            return f"Reduce allocation from {current_weight:.1%} to {target_weight:.1%} (overweight by {weight_diff:.1%})"
        else:
            return f"Maintain current allocation of {current_weight:.1%}"

# Global instances
portfolio_optimizer = PortfolioOptimizer()
risk_analyzer = RiskAnalyzer()
rebalancing_engine = RebalancingEngine()

def initialize_risk_management():
    """Initialize risk management components"""
    logger.info("✅ Portfolio Risk Management initialized")
    return portfolio_optimizer, risk_analyzer, rebalancing_engine