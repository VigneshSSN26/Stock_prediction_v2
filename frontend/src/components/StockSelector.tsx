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
      <h3>📊 Stock Selection</h3>
      <div className="form-group">
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
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '16px',
              boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)',
              maxHeight: '240px',
              overflow: 'auto'
            }}>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="🔍 Search symbols..."
                style={{ 
                  width: '100%', 
                  padding: '0.875rem 1rem', 
                  borderBottom: '1px solid rgba(226, 232, 240, 0.5)',
                  border: 'none',
                  background: 'transparent',
                  fontSize: '1rem'
                }}
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
                      padding: '0.75rem 1rem',
                      border: 'none',
                      background: 'transparent',
                      cursor: 'pointer',
                      fontSize: '1rem',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'rgba(78, 205, 196, 0.1)';
                      e.currentTarget.style.color = '#2d3748';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = '#4a5568';
                    }}
                  >
                    📈 {symbol}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div style={{ 
        background: 'rgba(78, 205, 196, 0.1)', 
        padding: '1rem', 
        borderRadius: '12px',
        border: '1px solid rgba(78, 205, 196, 0.2)',
        marginTop: '1rem'
      }}>
        <p style={{ 
          fontSize: '0.875rem', 
          color: '#4a5568', 
          margin: 0,
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          💡 <strong>Tip:</strong> Use .NS suffix for Indian stocks (NSE), no suffix for US stocks
        </p>
      </div>
      
      <div style={{ 
        marginTop: '1rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '0.5rem'
      }}>
        <button
          onClick={() => onSymbolChange('RELIANCE.NS')}
          style={{
            padding: '0.5rem',
            fontSize: '0.75rem',
            background: selectedSymbol === 'RELIANCE.NS' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'rgba(255, 255, 255, 0.8)',
            color: selectedSymbol === 'RELIANCE.NS' ? 'white' : '#4a5568',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          RELIANCE.NS
        </button>
        <button
          onClick={() => onSymbolChange('AAPL')}
          style={{
            padding: '0.5rem',
            fontSize: '0.75rem',
            background: selectedSymbol === 'AAPL' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'rgba(255, 255, 255, 0.8)',
            color: selectedSymbol === 'AAPL' ? 'white' : '#4a5568',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          AAPL
        </button>
        <button
          onClick={() => onSymbolChange('GOOGL')}
          style={{
            padding: '0.5rem',
            fontSize: '0.75rem',
            background: selectedSymbol === 'GOOGL' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'rgba(255, 255, 255, 0.8)',
            color: selectedSymbol === 'GOOGL' ? 'white' : '#4a5568',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          GOOGL
        </button>
        <button
          onClick={() => onSymbolChange('MSFT')}
          style={{
            padding: '0.5rem',
            fontSize: '0.75rem',
            background: selectedSymbol === 'MSFT' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 'rgba(255, 255, 255, 0.8)',
            color: selectedSymbol === 'MSFT' ? 'white' : '#4a5568',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          MSFT
        </button>
      </div>
    </div>
  );
};

export default StockSelector;
