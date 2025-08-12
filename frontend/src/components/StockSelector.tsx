import React, { useState } from 'react';

interface StockSelectorProps {
  onSymbolChange: (symbol: string) => void;
  selectedSymbol: string;
}

const StockSelector: React.FC<StockSelectorProps> = ({ onSymbolChange, selectedSymbol }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  // Stock symbols that work with Alpha Vantage API
  const stockSymbols = [
    // Indian Stocks (NSE)
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'SUNPHARMA.NS',
    'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS',
    
    // US Stocks
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
    
    // Simple symbols for testing
    'RELIANCE', 'TCS', 'INFY', 'AAPL', 'GOOGL', 'MSFT'
  ];

  const filteredSymbols = stockSymbols.filter(symbol =>
    symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSymbolSelect = (symbol: string) => {
    onSymbolChange(symbol);
    setIsOpen(false);
    setSearchTerm('');
  };

  return (
    <div className="stock-selector">
      <label htmlFor="stock-symbol">
        Select Stock Symbol
      </label>
      <div style={{ position: 'relative' }}>
        <input
          type="text"
          id="stock-symbol"
          value={selectedSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          onFocus={() => setIsOpen(true)}
          placeholder="Enter stock symbol (e.g., RELIANCE.NS, AAPL)..."
        />
        
        {isOpen && (
          <div style={{ 
            position: 'absolute', 
            zIndex: 10, 
            width: '100%', 
            marginTop: '4px',
            backgroundColor: 'white',
            border: '1px solid #d1d5db',
            borderRadius: '0.375rem',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            maxHeight: '240px',
            overflow: 'auto'
          }}>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search symbols..."
              style={{ width: '100%', padding: '0.5rem 0.75rem', borderBottom: '1px solid #e5e7eb' }}
              autoFocus
            />
            <div style={{ maxHeight: '192px', overflow: 'auto' }}>
              {filteredSymbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => handleSymbolSelect(symbol)}
                  style={{ 
                    width: '100%', 
                    textAlign: 'left', 
                    padding: '0.5rem 0.75rem',
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <p style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.5rem' }}>
        💡 Tip: Use .NS suffix for Indian stocks (NSE), no suffix for US stocks
      </p>
    </div>
  );
};

export default StockSelector;
