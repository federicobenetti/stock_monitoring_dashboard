# Stock Performance Dashboard

A comprehensive technical analysis dashboard for monitoring individual stock performance, built with Streamlit and powered by real-time market data from Yahoo Finance.

**🔗 Live Application**: [https://stockmonitoring.streamlit.app/](https://stockmonitoring.streamlit.app/)

## Overview

This interactive dashboard provides institutional-grade technical analysis and performance metrics for stocks, enabling traders and investors to make informed decisions through visual charting and quantitative indicators.

## Key Features

### 📊 Interactive Price Charts
- **Candlestick visualization** with OHLCV data
- **Moving averages** (customizable periods: 20, 50, 100, 200-day)
- **Bollinger Bands** with adjustable parameters
- **MACD indicator** with histogram visualization

### 📈 Technical Oscillators
- **RSI (Relative Strength Index)** with overbought/oversold levels
- **Stochastic Oscillator** (%K and %D)
- **CCI (Commodity Channel Index)**

### 💡 Performance Metrics
- **Returns analysis**: 1W, 1M, 3M, 6M, YTD
- **Trend indicators**: MA distances, breadth, and slope
- **Momentum signals**: MACD, RSI, Stochastic states
- **Volatility metrics**: Annualized volatility (30D/90D), volume percentiles
- **Risk measures**: Current drawdown, max drawdown (90D)
- **Liquidity**: Average daily volume, volume z-scores

### 🎯 Trading Signals
- MACD bullish/bearish crosses
- RSI oversold/overbought conditions
- Bollinger Band breakouts
- Signal change tracking

### 🎨 Visual Status Indicators
Color-coded emoji system for quick assessment:
- 🟢 Good/Bullish
- 🟡 Warning/Neutral
- 🔴 Bad/Bearish
- ⚪ Neutral/Informational

## How to Use

1. **Select a ticker** from the preset list or enter a custom symbol
2. **Adjust the date range** to focus on your period of interest
3. **Customize parameters** (moving averages, Bollinger Bands) via the sidebar
4. **Analyze the charts** and metrics to assess the stock's technical position
5. **Review the guide tab** for detailed indicator interpretations

## Technical Stack

- **Frontend**: Streamlit
- **Data**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly
- **Analysis**: Pandas, NumPy
- **Caching**: 1-hour TTL for optimal performance

## Preset Tickers

Includes major stocks and ETFs:
- Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Financials: JPM, V
- Indices: SPY, QQQ, DIA
- And more...

## Configuration

The dashboard supports customizable parameters through `config/params.yaml`:
- Moving average periods
- Bollinger Band period and standard deviation multiplier
- Default settings with reset functionality

## Data Refresh

Market data is cached for 1 hour to balance freshness with API rate limits. The dashboard automatically fetches 2 years of historical data for comprehensive analysis.

## Use Cases

- **Day traders**: Monitor intraday momentum and oscillator signals
- **Swing traders**: Identify trend changes and support/resistance levels
- **Portfolio managers**: Track performance metrics and risk indicators
- **Technical analysts**: Conduct multi-timeframe analysis with customizable parameters

## Contributing

This dashboard is designed for educational and informational purposes. Market data is provided by Yahoo Finance and should be verified before making trading decisions.

---

**⚠️ Disclaimer**: This tool is for informational purposes only and does not constitute financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.
