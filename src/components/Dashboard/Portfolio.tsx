import React, { useState } from 'react';
import './Portfolio.css';

interface PortfolioItem {
  symbol: string;
  name: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  totalValue: number;
  gainLoss: number;
  gainLossPercent: number;
}

const Portfolio: React.FC = () => {
  const [portfolioItems] = useState<PortfolioItem[]>([
    {
      symbol: 'RELIANCE.NS',
      name: 'Reliance Industries',
      shares: 100,
      avgPrice: 2450.50,
      currentPrice: 2520.75,
      totalValue: 252075,
      gainLoss: 7025,
      gainLossPercent: 2.87
    },
    {
      symbol: 'TCS.NS',
      name: 'Tata Consultancy Services',
      shares: 50,
      avgPrice: 3850.25,
      currentPrice: 3920.50,
      totalValue: 196025,
      gainLoss: 3512.5,
      gainLossPercent: 1.83
    },
    {
      symbol: 'INFY.NS',
      name: 'Infosys Limited',
      shares: 75,
      avgPrice: 1450.75,
      currentPrice: 1420.25,
      totalValue: 106518.75,
      gainLoss: -2287.5,
      gainLossPercent: -2.10
    },
    {
      symbol: 'HDFCBANK.NS',
      name: 'HDFC Bank',
      shares: 200,
      avgPrice: 1650.00,
      currentPrice: 1685.50,
      totalValue: 337100,
      gainLoss: 7100,
      gainLossPercent: 2.15
    }
  ]);

  const totalPortfolioValue = portfolioItems.reduce((sum, item) => sum + item.totalValue, 0);
  const totalGainLoss = portfolioItems.reduce((sum, item) => sum + item.gainLoss, 0);
  const totalGainLossPercent = (totalGainLoss / (totalPortfolioValue - totalGainLoss)) * 100;

  return (
    <div className="portfolio-container">
      <div className="portfolio-header">
        <h2>My Portfolio</h2>
        <div className="portfolio-summary">
          <div className="summary-item">
            <span className="summary-label">Total Value</span>
            <span className="summary-value">₹{totalPortfolioValue.toLocaleString()}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Total P&L</span>
            <span className={`summary-value ${totalGainLoss >= 0 ? 'positive' : 'negative'}`}>
              {totalGainLoss >= 0 ? '+' : ''}₹{totalGainLoss.toLocaleString()} ({totalGainLossPercent.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      <div className="portfolio-table">
        <div className="table-header">
          <div className="table-cell">Stock</div>
          <div className="table-cell">Shares</div>
          <div className="table-cell">Avg Price</div>
          <div className="table-cell">Current Price</div>
          <div className="table-cell">Total Value</div>
          <div className="table-cell">P&L</div>
        </div>

        {portfolioItems.map((item) => (
          <div key={item.symbol} className="table-row">
            <div className="table-cell stock-info">
              <div className="stock-symbol">{item.symbol}</div>
              <div className="stock-name">{item.name}</div>
            </div>
            <div className="table-cell">{item.shares}</div>
            <div className="table-cell">₹{item.avgPrice.toFixed(2)}</div>
            <div className="table-cell">₹{item.currentPrice.toFixed(2)}</div>
            <div className="table-cell">₹{item.totalValue.toLocaleString()}</div>
            <div className={`table-cell ${item.gainLoss >= 0 ? 'positive' : 'negative'}`}>
              {item.gainLoss >= 0 ? '+' : ''}₹{item.gainLoss.toLocaleString()}
              <div className="gain-loss-percent">({item.gainLossPercent.toFixed(2)}%)</div>
            </div>
          </div>
        ))}
      </div>

      <div className="portfolio-actions">
        <button className="action-button primary">Add Stock</button>
        <button className="action-button secondary">View Analytics</button>
        <button className="action-button secondary">Export Data</button>
      </div>
    </div>
  );
};

export default Portfolio;
