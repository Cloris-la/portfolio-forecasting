# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules - Fixed imports
from optimizer import Portfolio_Optimizer, PortfolioComparison, ProphetForecaster
from financial_indicator import FinancialMetrics

# Setting page config
st.set_page_config(
    page_title="Portfolio Optimization System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed caching function
@st.cache_data(ttl=3600)
def fetch_data(symbols, start_date, end_date, frequency):
    """
    Get stock data with proper caching
    """
    interval_map = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo'}
    
    try:
        # Direct yfinance call instead of importing function incorrectly
        data = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval=interval_map[frequency],
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        
        if len(symbols) == 1:
            close_prices = data['Close'].to_frame()
            close_prices.columns = symbols
        else:
            close_prices = pd.DataFrame()
            for symbol in symbols:
                if symbol in data.columns.levels[0] if hasattr(data.columns, 'levels') else symbol in data:
                    if hasattr(data.columns, 'levels'):
                        close_prices[symbol] = data[symbol]['Close']
                    else:
                        close_prices[symbol] = data['Close'][symbol] if symbol in data['Close'].columns else data['Close']
        
        close_prices.dropna(inplace=True)
        return close_prices
        
    except Exception as e:
        st.error(f"Failed fetching data: {str(e)}")
        return pd.DataFrame()

# Main application function
def main():
    st.title("📊 Advanced Portfolio Optimization & Forecasting System")
    st.markdown("""
    This application demonstrates advanced portfolio optimization using:
    - **Historical optimization**: Based on past data
    - **Forecast optimization**: Using Prophet for price prediction
    - **Actual optimization**: The ideal portfolio (with perfect foresight)
    - **Multiple risk measures**: MV, CVaR, MAD, EVaR, CDaR
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Stock selection
        symbols = st.multiselect(
            "Select Stocks",
            options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'META', 'JPM', 'DIS'],
            default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            max_selections=10
        )

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*3),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )

        # Data frequency
        frequency = st.selectbox(
            "Data Frequency",
            options=['Daily', 'Weekly', 'Monthly'],
            index=0
        )

        # Risk measure
        risk_measure = st.selectbox(
            "Risk Measure",
            options=['MV', 'CVaR', 'MAD'],
            index=0,
            help="MV: Mean-Variance, CVaR: Conditional Value at Risk, MAD: Mean Absolute Deviation"
        )

        # Forecast days
        forecast_days = st.slider(
            "Forecast Days",
            min_value=10,
            max_value=90,
            value=30
        )

        # Optimization objective
        objective = st.selectbox(
            "Optimization Objective",
            options=['Sharpe', 'MinRisk', 'MaxRet', 'Utility'],
            index=0
        )
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Main content area
    if run_analysis and symbols:
        with st.spinner("Fetching data..."):
            # Get data
            prices = fetch_data(symbols, start_date, end_date, frequency)
            
            if prices.empty:
                st.error("Unable to fetch data. Please check your inputs.")
                return
            
            st.success(f"✅ Loaded {len(prices)} data points for {len(symbols)} stocks")

        # Create tabs
        tabs = st.tabs([
            "📈 Data Overview", 
            "📊 Financial Metrics", 
            "⚖️ Portfolio Optimization", 
            "🔮 Performance Comparison", 
            "📉 Efficient Frontier"
        ])
        
        with tabs[0]:
            st.header("Historical Price Data")
            
            # Price chart
            fig = go.Figure()
            for col in prices.columns:
                fig.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices[col],
                    mode='lines',
                    name=col,
                    hovertemplate='%{fullData.name}<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Stock Prices Over Time",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary_stats = prices.describe()
            st.dataframe(summary_stats.style.format("${:.2f}"), use_container_width=True)
        
        with tabs[1]:
            st.header("Financial Metrics Analysis")
            
            # Calculate metrics for each asset
            metrics_calc = FinancialMetrics(frequency=frequency.lower())
            
            metrics_data = []
            for symbol in prices.columns:
                metrics = metrics_calc.calculate_all_metrics(prices[symbol], symbol)
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Format for display
            display_df = metrics_df.copy()
            for col in ['Annual Return', 'Annual Volatility', 'Max Drawdown', 'VaR(95%)', 'CVaR(95%)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            
            for col in ['Sharpe Ratio', 'Sortino Ratio', 'Skewness', 'Kurtosis']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization of key metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk-Return scatter
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics_df['Annual Volatility'],
                    y=metrics_df['Annual Return'],
                    mode='markers+text',
                    text=metrics_df['Symbol'],
                    textposition='top center',
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    title="Risk-Return Profile",
                    xaxis_title="Annual Volatility",
                    yaxis_title="Annual Return",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sharpe Ratio comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics_df['Symbol'],
                    y=metrics_df['Sharpe Ratio'],
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Sharpe Ratio Comparison",
                    xaxis_title="Asset",
                    yaxis_title="Sharpe Ratio",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.header("Portfolio Optimization Results")
            
            try:
                # Run optimization
                optimizer = Portfolio_Optimizer(prices)
                result = optimizer.optimize_portfolio(objective, risk_measure)
                
                if result and 'weights' in result:
                    # Display weights
                    st.subheader("Optimal Portfolio Weights")
                    
                    weights_df = pd.DataFrame({
                        'Asset': result['weights'].index,
                        'Weight': result['weights'].values.flatten()
                    }).set_index('Asset')
                    
                    # Filter significant weights
                    significant_weights = weights_df[weights_df['Weight'] > 0.001]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(
                            significant_weights.style.format({'Weight': '{:.2%}'}),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=significant_weights.index,
                            values=significant_weights['Weight'],
                            hole=0.3
                        )])
                        fig.update_layout(
                            title="Portfolio Composition",
                            height=300,
                            margin=dict(t=50, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("Expected Portfolio Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Expected Return", f"{result.get('return', 0):.2%}")
                    with col2:
                        st.metric("Volatility", f"{result.get('volatility', 0):.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{result.get('sharpe', 0):.2f}")
                    with col4:
                        st.metric("Sortino Ratio", f"{result.get('sortino', 0):.2f}")
                else:
                    st.error("Optimization failed. Please try different parameters.")
                    
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
        
        with tabs[3]:
            st.header("Historical vs Forecast vs Actual Comparison")
            
            try:
                # Run comparison
                comparison = PortfolioComparison(prices, forecast_days)
                results = comparison.run_comparison(risk_measure)
                
                if results:
                    # Weights comparison
                    st.subheader("Portfolio Weights Comparison")
                    
                    weights_comparison = {}
                    for name, res in results.items():
                        if res and 'weights' in res:
                            weights_comparison[name] = res['weights']
                    
                    if weights_comparison:
                        weights_df = pd.DataFrame(weights_comparison)
                        st.dataframe(
                            weights_df.style.format("{:.2%}"),
                            use_container_width=True
                        )
                    
                    # Performance comparison
                    st.subheader("Performance Metrics Comparison")
                    
                    performance_data = []
                    for name, res in results.items():
                        if res:
                            row = {
                                'Portfolio': name,
                                'Expected Return': f"{res.get('return', 0):.2%}",
                                'Expected Volatility': f"{res.get('volatility', 0):.2%}",
                                'Expected Sharpe': f"{res.get('sharpe', 0):.2f}"
                            }
                            
                            if 'test_return' in res:
                                row.update({
                                    'Actual Return': f"{res['test_return']:.2%}",
                                    'Actual Volatility': f"{res['test_volatility']:.2%}",
                                    'Actual Sharpe': f"{res['test_sharpe']:.2f}"
                                })
                            
                            performance_data.append(row)
                    
                    if performance_data:
                        perf_df = pd.DataFrame(performance_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # Visualization
                        fig = go.Figure()
                        
                        for name in results.keys():
                            res = results[name]
                            if res:
                                values = [
                                    res.get('test_return', res.get('return', 0)),
                                    res.get('test_volatility', res.get('volatility', 0)),
                                    res.get('test_sharpe', res.get('sharpe', 0))
                                ]
                                
                                fig.add_trace(go.Bar(
                                    name=name,
                                    x=['Return', 'Volatility', 'Sharpe Ratio'],
                                    y=values
                                ))
                        
                        fig.update_layout(
                            title="Portfolio Performance Comparison",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Key insights
                    st.info("""
                    **Key Insights:**
                    - **Historical**: Portfolio optimized using only past data
                    - **Forecast**: Portfolio optimized using Prophet predictions
                    - **Actual**: The theoretically optimal portfolio (if we knew the future)
                    
                    The gap between Forecast and Actual shows prediction accuracy impact.
                    """)
                    
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
        
        with tabs[4]:
            st.header("Efficient Frontier Analysis")
            
            try:
                optimizer = Portfolio_Optimizer(prices)
                frontier = optimizer.get_efficient_frontier(risk_measure, n_points=50)
                
                if frontier is not None:
                    # Calculate frontier points
                    returns = []
                    volatilities = []
                    
                    for weights in frontier.T:
                        portfolio_return = (optimizer.returns @ weights).mean() * 252
                        portfolio_vol = np.sqrt(weights.T @ (optimizer.returns.cov() * 252) @ weights)
                        returns.append(portfolio_return)
                        volatilities.append(portfolio_vol)
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Efficient frontier
                    fig.add_trace(go.Scatter(
                        x=volatilities,
                        y=returns,
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Individual assets
                    for symbol in prices.columns:
                        stock_return = optimizer.returns[symbol].mean() * 252
                        stock_vol = optimizer.returns[symbol].std() * np.sqrt(252)
                        
                        fig.add_trace(go.Scatter(
                            x=[stock_vol],
                            y=[stock_return],
                            mode='markers+text',
                            name=symbol,
                            text=[symbol],
                            textposition='top center',
                            marker=dict(size=10)
                        ))
                    
                    fig.update_layout(
                        title=f"Efficient Frontier ({risk_measure})",
                        xaxis_title="Volatility (Risk)",
                        yaxis_title="Expected Return",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    The Efficient Frontier represents all optimal portfolios that offer:
                    - Maximum return for a given level of risk
                    - Minimum risk for a given level of return
                    
                    Points below the frontier are suboptimal.
                    """)
                    
            except Exception as e:
                st.error(f"Error calculating efficient frontier: {str(e)}")
    
    elif run_analysis and not symbols:
        st.error("Please select at least one stock for analysis.")
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to begin.")
        
        with st.expander("📖 How to Use This App"):
            st.markdown("""
            1. **Select Stocks**: Choose up to 10 stocks to analyze
            2. **Set Date Range**: Define your analysis period
            3. **Choose Frequency**: Daily, Weekly, or Monthly data
            4. **Select Risk Measure**: Different ways to measure portfolio risk
            5. **Set Forecast Period**: How many days to predict ahead
            6. **Choose Objective**: What to optimize for
            7. **Run Analysis**: Click to start the optimization
            
            The app will show you:
            - Historical data analysis
            - Individual asset metrics
            - Optimal portfolio weights
            - Comparison of different optimization approaches
            - The efficient frontier visualization
            """)

if __name__ == "__main__":
    main()