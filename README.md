# 📊 Portfolio Optimization & Forecasting System

An advanced Python-based portfolio optimization and forecasting system that combines financial engineering, machine learning, and data science techniques to provide interactive portfolio analysis and optimization capabilities.

## 🌟 Key Features

- **Multi-frequency Data Acquisition**: Support for daily, weekly, monthly, and yearly stock data retrieval
- **Comprehensive Financial Metrics**: Calculation of key indicators including annualized return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, VaR, and CVaR
- **Portfolio Optimization**: Mean-variance optimization with multiple risk measures (MV, CVaR, MAD, etc.)
- **Price Forecasting**: Future stock price prediction using Prophet time series models
- **Interactive Web Interface**: Streamlit-based web application for user-friendly interaction
- **Three-Portfolio Comparison**: Historical-based, forecast-based, and actual realized portfolio optimization comparison
- **Efficient Frontier Visualization**: Graphical representation of optimal risk-return portfolios

## 🏗️ System Architecture

```
portfolio-optimization/
├── app.py                 # Streamlit web application
├── optimizer.py           # Portfolio optimization core logic
├── financial_indicator.py # Financial metrics calculation
├── get_data.py           # Data acquisition from Yahoo Finance
├── requirements.txt      # Project dependencies
└── data/                 # Data storage directory
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Installation

1. **Clone or download the project files**
```bash
git clone <https://github.com/Cloris-la/portfolio-forecasting.git>
cd portfolio-optimization
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install additional dependencies if needed**
```bash
pip install yfinance riskfolio-lib plotly streamlit prophet scipy
```

## 🚀 Quick Start

1. **Run the Streamlit application**
```bash
streamlit run src/app.py
```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - Configure your analysis parameters in the sidebar
   - Click "Run Analysis" to start the optimization process

3. **Explore different tabs**
   - **Data Overview**: View historical price data and summary statistics
   - **Financial Metrics**: Analyze individual asset performance metrics
   - **Portfolio Optimization**: View optimal portfolio weights and expected performance
   - **Performance Comparison**: Compare historical, forecast, and actual portfolio performance
   - **Efficient Frontier**: Visualize the optimal risk-return tradeoff

## 🛠️ Usage Examples

### Basic Portfolio Optimization
```python
from optimizer import Portfolio_Optimizer
import pandas as pd

# Load your price data
prices = pd.read_csv('data/daily_stock_prices.csv', index_col=0, parse_dates=True)

# Initialize optimizer
optimizer = Portfolio_Optimizer(prices)

# Optimize for maximum Sharpe ratio
result = optimizer.optimize_portfolio(objective='Sharpe', risk_measure='MV')

print("Optimal weights:", result['weights'])
print("Expected return:", result['return'])
print("Volatility:", result['volatility'])
```

### Financial Metrics Analysis
```python
from financial_indicator import FinancialMetrics

metrics_calc = FinancialMetrics(frequency='daily')
metrics = metrics_calc.calculate_all_metrics(prices['AAPL'], 'AAPL')
print(metrics)
```

## 🔧 Configuration Options

### Data Parameters
- **Stocks**: Select from 10+ popular stocks (AAPL, MSFT, GOOGL, NVDA, TSLA, NFLX, JPM, AMZN, META, DIS.)
- **Date Range**: Customizable start and end dates
- **Frequency**: Daily, weekly, or monthly data

### Optimization Settings
- **Risk Measures**: MV (Mean-Variance), CVaR, MAD
- **Objectives**: Sharpe ratio maximization, risk minimization, return maximization
- **Forecast Period**: 10-90 days prediction horizon

## 📊 Output Features

1. **Optimal Portfolio Weights**: Percentage allocation for each asset
2. **Performance Metrics**: Expected return, volatility, Sharpe ratio, Sortino ratio
3. **Visualizations**: 
   - Interactive price charts
   - Risk-return scatter plots
   - Portfolio composition pie charts
   - Efficient frontier graphs
4. **Comparison Analysis**: Side-by-side comparison of different optimization approaches

## 🎯 Use Cases

- **Individual Investors**: Personal portfolio optimization and risk management
- **Financial Analysts**: Rapid prototyping of investment strategies
- **Educational Purposes**: Learning modern portfolio theory and optimization techniques
- **Research**: Experimental analysis of different risk measures and optimization objectives

## ⚙️ Technical Details

### Core Technologies
- **Data Acquisition**: yfinance for Yahoo Finance data
- **Optimization Engine**: Riskfolio-Lib for portfolio optimization
- **Forecasting**: Facebook Prophet for time series prediction
- **Web Interface**: Streamlit for interactive visualization
- **Data Processing**: Pandas, NumPy for financial calculations
- **Visualization**: Plotly for interactive charts

### Mathematical Foundations
- Modern Portfolio Theory (Markowitz)
- Mean-Variance Optimization
- Conditional Value at Risk (CVaR)
- Sharpe and Sortino Ratios
- Time Series Forecasting

## 🐛 Troubleshooting

### Common Issues

1. **Optimization fails**
   - Ensure you have sufficient historical data (>30 days)
   - Try different risk measures or objectives
   - Check for missing data or NaN values

2. **Prophet installation issues**
   - On Windows, you may need to install Microsoft C++ Build Tools
   - Use conda instead of pip: `conda install -c conda-forge prophet`

3. **Data fetching errors**
   - Check internet connection
   - Verify stock symbols are valid

### Performance Tips
- Use weekly data for faster analysis with large date ranges
- Limit the number of stocks for quicker optimization
- Consider using simpler risk measures (MV) for faster computation

## 🤝 Contributing

We welcome contributions to improve this project! Areas for contribution include:
- Additional risk measures and optimization objectives
- Enhanced forecasting models
- Improved visualization capabilities
- Performance optimizations
- Documentation improvements

Please follow standard GitHub pull request procedures for contributions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

Planned features for future releases:
- [ ] Additional forecasting models (LSTM, ARIMA)
- [ ] Black-Litterman model integration
- [ ] Transaction cost consideration
- [ ] Tax optimization features
- [ ] Real-time data streaming
- [ ] Advanced risk analytics
- [ ] Portfolio backtesting capabilities
- [ ] API integration with brokerage accounts

---

**Disclaimer**: This tool is for educational and research purposes only. It does not provide financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance is not indicative of future results.