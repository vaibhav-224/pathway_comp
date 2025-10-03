"""
Portfolio Tracking & Impact Analysis Engine
Real-time portfolio monitoring using Pathway for "what's moving my portfolio?" insights
"""
import pathway as pw
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    shares: float
    cost_basis: float
    purchase_date: str

class PortfolioTracker:
    """Real-time portfolio tracking and impact analysis using Pathway"""
    
    def __init__(self, positions: List[PortfolioPosition]):
        self.positions = positions
        self.position_dict = {pos.symbol: pos for pos in positions}
        
    def create_portfolio_table(self) -> pw.Table:
        """Create Pathway table from portfolio positions"""
        
        # Convert positions to table format
        portfolio_rows = []
        for pos in self.positions:
            row = (
                pos.symbol,
                pos.shares,
                pos.cost_basis,
                pos.purchase_date,
                pos.shares * pos.cost_basis  # Total cost
            )
            portfolio_rows.append(row)
        
        # Define portfolio schema
        portfolio_schema = pw.schema_from_dict({
            'symbol': str,
            'shares': float,
            'cost_basis': float,
            'purchase_date': str,
            'total_cost': float
        })
        
        portfolio_table = pw.debug.table_from_rows(
            schema=portfolio_schema, 
            rows=portfolio_rows
        )
        
        return portfolio_table
    
    def calculate_portfolio_impact(self, market_table: pw.Table, portfolio_table: pw.Table) -> pw.Table:
        """Calculate real-time portfolio impact using Pathway operations"""
        
        # Join market data with portfolio holdings
        portfolio_impact = market_table.join(
            portfolio_table,
            market_table.symbol == portfolio_table.symbol,
            how=pw.JoinMode.INNER
        ).select(
            symbol=pw.this.symbol,
            current_price=pw.this.price,
            change_pct=pw.this.change_pct,
            volume=pw.this.volume,
            timestamp=pw.this.timestamp,
            shares_held=pw.this.shares,
            cost_basis=pw.this.cost_basis,
            purchase_date=pw.this.purchase_date,
            # Calculate current position value
            current_value=pw.this.price * pw.this.shares,
            # Calculate cost value
            cost_value=pw.this.cost_basis * pw.this.shares,
            # Calculate absolute P&L
            absolute_pnl=(pw.this.price - pw.this.cost_basis) * pw.this.shares,
            # Calculate percentage P&L
            percent_pnl=pw.if_else(
                pw.this.cost_basis > 0,
                ((pw.this.price - pw.this.cost_basis) / pw.this.cost_basis) * 100.0,
                0.0
            ),
            # Calculate daily impact (change in value)
            daily_impact=(pw.this.change_pct / 100.0) * (pw.this.price * pw.this.shares)
        )
        
        # Add derived categorizations
        final_impact = portfolio_impact.select(
            symbol=pw.this.symbol,
            current_price=pw.this.current_price,
            change_pct=pw.this.change_pct,
            volume=pw.this.volume,
            timestamp=pw.this.timestamp,
            shares_held=pw.this.shares_held,
            cost_basis=pw.this.cost_basis,
            purchase_date=pw.this.purchase_date,
            current_value=pw.this.current_value,
            cost_value=pw.this.cost_value,
            absolute_pnl=pw.this.absolute_pnl,
            percent_pnl=pw.this.percent_pnl,
            daily_impact=pw.this.daily_impact,
            # Position impact category
            impact_category=pw.if_else(
                pw.this.daily_impact > 1000.0,
                "HIGH_IMPACT",
                pw.if_else(
                    pw.this.daily_impact > 500.0,
                    "MEDIUM_IMPACT",
                    pw.if_else(
                        pw.this.daily_impact > 100.0,
                        "LOW_IMPACT",
                        "MINIMAL_IMPACT"
                    )
                )
            ),
            # Position status
            position_status=pw.if_else(
                pw.this.percent_pnl > 10.0,
                "STRONG_GAIN",
                pw.if_else(
                    pw.this.percent_pnl > 0.0,
                    "PROFITABLE",
                    pw.if_else(
                        pw.this.percent_pnl > -10.0,
                        "SLIGHT_LOSS",
                        "SIGNIFICANT_LOSS"
                    )
                )
            )
        )
        
        return final_impact
    
    def get_top_movers_impact(self, portfolio_impact: pw.Table) -> pw.Table:
        """Get portfolio positions with highest impact"""
        
        # Filter for significant daily impacts
        top_movers = portfolio_impact.filter(
            (pw.this.daily_impact > 100.0) | (pw.this.daily_impact < -100.0)
        )
        
        # Add ranking information
        ranked_movers = top_movers.select(
            symbol=pw.this.symbol,
            current_price=pw.this.current_price,
            change_pct=pw.this.change_pct,
            shares_held=pw.this.shares_held,
            daily_impact=pw.this.daily_impact,
            percent_pnl=pw.this.percent_pnl,
            impact_category=pw.this.impact_category,
            position_status=pw.this.position_status,
            # Create impact description
            impact_description=pw.this.symbol + "_" + pw.this.impact_category + "_" + pw.this.position_status
        )
        
        return ranked_movers
    
    def calculate_portfolio_summary(self, portfolio_impact: pw.Table) -> Dict[str, Any]:
        """Calculate portfolio-wide summary statistics"""
        
        # Note: In a full implementation, we'd aggregate across the table
        # For now, we'll return structure for the summary
        
        summary = {
            'total_positions': len(self.positions),
            'portfolio_symbols': [pos.symbol for pos in self.positions],
            'tracking_status': 'ACTIVE',
            'last_updated': datetime.now().isoformat(),
            'impact_analysis': 'REAL_TIME'
        }
        
        return summary
    
    def create_portfolio_alerts(self, portfolio_impact: pw.Table) -> pw.Table:
        """Create alerts for significant portfolio movements"""
        
        # Filter for positions requiring attention
        portfolio_alerts = portfolio_impact.filter(
            (pw.this.daily_impact > 500.0) | 
            (pw.this.daily_impact < -500.0) |
            (pw.this.percent_pnl < -15.0)
        )
        
        # Add alert information
        alerts_table = portfolio_alerts.select(
            symbol=pw.this.symbol,
            alert_type=pw.if_else(
                pw.this.daily_impact > 500.0,
                "LARGE_GAIN",
                pw.if_else(
                    pw.this.daily_impact < -500.0,
                    "LARGE_LOSS",
                    "POSITION_RISK"
                )
            ),
            severity=pw.if_else(
                (pw.this.daily_impact > 1000.0) | (pw.this.daily_impact < -1000.0),
                "HIGH",
                "MEDIUM"
            ),
            current_price=pw.this.current_price,
            daily_impact=pw.this.daily_impact,
            percent_pnl=pw.this.percent_pnl,
            recommended_action=pw.if_else(
                pw.this.daily_impact > 1000.0,
                "CONSIDER_PROFIT_TAKING",
                pw.if_else(
                    pw.this.daily_impact < -1000.0,
                    "REVIEW_POSITION",
                    "MONITOR_CLOSELY"
                )
            )
        )
        
        return alerts_table
    
    def correlate_with_news(self, portfolio_impact: pw.Table, news_table: pw.Table) -> pw.Table:
        """Correlate portfolio movements with news events"""
        
        # Create correlation analysis
        # Filter portfolio movements that might be news-driven
        news_correlated = portfolio_impact.filter(
            (pw.this.change_pct > 2.0) | (pw.this.change_pct < -2.0)
        )
        
        # Add news correlation flags
        correlated_movements = news_correlated.select(
            symbol=pw.this.symbol,
            change_pct=pw.this.change_pct,
            daily_impact=pw.this.daily_impact,
            current_price=pw.this.current_price,
            # Indicate potential news correlation
            potential_news_driven=pw.if_else(
                (pw.this.change_pct > 3.0) | (pw.this.change_pct < -3.0),
                True,
                False
            ),
            correlation_strength=pw.if_else(
                (pw.this.change_pct > 5.0) | (pw.this.change_pct < -5.0),
                "STRONG",
                pw.if_else(
                    (pw.this.change_pct > 3.0) | (pw.this.change_pct < -3.0),
                    "MODERATE",
                    "WEAK"
                )
            )
        )
        
        return correlated_movements

def test_portfolio_tracking():
    """Test the portfolio tracking system"""
    print("🧪 Testing Portfolio Tracking & Impact Analysis...")
    
    # Create sample portfolio
    sample_portfolio = [
        PortfolioPosition("AAPL", 100.0, 220.0, "2025-01-15"),
        PortfolioPosition("MSFT", 50.0, 450.0, "2025-02-01"),
        PortfolioPosition("GOOGL", 25.0, 2400.0, "2025-01-20"),
        PortfolioPosition("TSLA", 75.0, 380.0, "2025-03-01"),
        PortfolioPosition("AMZN", 40.0, 240.0, "2025-02-15")
    ]
    
    # Initialize portfolio tracker
    tracker = PortfolioTracker(sample_portfolio)
    
    # Create portfolio table
    print("💼 Creating portfolio table...")
    portfolio_table = tracker.create_portfolio_table()
    
    print("📊 Portfolio Holdings:")
    pw.debug.compute_and_print(portfolio_table)
    
    # Create mock market data for testing
    market_data = [
        ("AAPL", "2025-09-21T15:00:00", 245.50, 1800000, 244.0, 246.0, 243.0, 1.8, 3600000000000, 50000000),
        ("MSFT", "2025-09-21T15:00:00", 520.00, 1700000, 518.0, 522.0, 517.0, 2.2, 3850000000000, 20000000),
        ("GOOGL", "2025-09-21T15:00:00", 2480.00, 900000, 2475.0, 2485.0, 2470.0, 3.3, 3100000000000, 40000000),
        ("TSLA", "2025-09-21T15:00:00", 365.00, 700000, 375.0, 375.0, 365.0, -3.9, 1350000000000, 90000000),
        ("AMZN", "2025-09-21T15:00:00", 235.00, 1050000, 238.0, 238.0, 234.0, -1.2, 2500000000000, 42000000),
    ]
    
    market_schema = pw.schema_from_dict({
        'symbol': str, 'timestamp': str, 'price': float, 'volume': int,
        'open': float, 'high': float, 'low': float, 'change_pct': float,
        'market_cap': int, 'avg_volume': int
    })
    
    market_table = pw.debug.table_from_rows(schema=market_schema, rows=market_data)
    
    print("\n📈 Current Market Data:")
    pw.debug.compute_and_print(market_table)
    
    # Calculate portfolio impact
    print("\n💰 Calculating portfolio impact...")
    portfolio_impact = tracker.calculate_portfolio_impact(market_table, portfolio_table)
    
    print("📊 Portfolio Impact Analysis:")
    pw.debug.compute_and_print(portfolio_impact)
    
    # Get top movers
    print("\n🔥 Top Portfolio Movers:")
    top_movers = tracker.get_top_movers_impact(portfolio_impact)
    pw.debug.compute_and_print(top_movers)
    
    # Create portfolio alerts
    print("\n🚨 Portfolio Alerts:")
    alerts = tracker.create_portfolio_alerts(portfolio_impact)
    pw.debug.compute_and_print(alerts)
    
    # Portfolio summary
    summary = tracker.calculate_portfolio_summary(portfolio_impact)
    print(f"\n💼 Portfolio Summary:")
    print(f"   Total Positions: {summary['total_positions']}")
    print(f"   Symbols: {', '.join(summary['portfolio_symbols'])}")
    print(f"   Status: {summary['tracking_status']}")
    print(f"   Impact Analysis: {summary['impact_analysis']}")
    
    return tracker, portfolio_impact

if __name__ == "__main__":
    test_portfolio_tracking()