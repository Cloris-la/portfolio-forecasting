import riskfolio as rp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple
import logging

# log setting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Portfolio_Optimizer:
    '''
    Portfolio optimization using Riskfolio-Lib and analyzing with financial indicators
    '''
    def __init__(self, prices_df: pd.DataFrame, risk_free_rate: float = 0.02):
        self.prices_df = prices_df
        self.returns = self.prices_df.pct_change().dropna()
        self.risk_free_rate = risk_free_rate

        # make sure returns is DataFrame format
        if isinstance(self.returns, pd.Series):
            self.returns = self.returns.to_frame()
            
        self.port = rp.Portfolio(returns=self.returns)

        # Set portfolio parameters - using more compatible parameters
        try:
            self.port.assets_stats(method_mu='hist', method_cov='hist')
        except Exception as e:
            logger.warning(f"Error setting assets stats: {e}")
            # Fallback method
            try:
                self.port.assets_stats(method_mu='mean', method_cov='std')
            except Exception as e2:
                logger.error(f"Failed to set assets stats with fallback: {e2}")

    def optimize_portfolio(self, objective: str = 'Sharpe', risk_measure: str = 'MV') -> Optional[dict]:
        '''
        Optimize the portfolio based on the given objective and risk measure
        
        Parameters:
            objective (str): The optimization objective, e.g., 'Sharpe', 'MinRisk', 'MaxRet'
            risk_measure (str): The risk measure to use, e.g., 'MV' for Mean-Variance, 'CVaR' for Conditional Value at Risk
        
        Returns:
            dict: Dictionary containing weights and performance metrics
        '''
        try:
            # Check if there is enough data
            if len(self.returns) < 2:
                logger.error("Not enough data for optimization")
                return None

            # Ensure all columns are numeric
            for col in self.returns.columns:
                if not pd.api.types.is_numeric_dtype(self.returns[col]):
                    self.returns[col] = pd.to_numeric(self.returns[col], errors='coerce')

            # Remove any rows with NaN or Inf
            self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(self.returns) < 2:
                logger.error("Not enough valid data after cleaning")
                return None

            # Reinitialize portfolio object
            self.port = rp.Portfolio(returns=self.returns)
            self.port.assets_stats(method_mu='hist', method_cov='hist')

            # Map risk measures to Riskfolio-Lib format
            rm_map = {
                'MV': 'MV',
                'CVaR': 'CVaR',
                'MAD': 'MAD',
                'MSV': 'MSV',
                'WR': 'WR'
            }
            
            if risk_measure not in rm_map:
                logger.warning(f'Unsupported risk measure: {risk_measure}, using MV instead')
                risk_measure = 'MV'

            # Optimize based on objective
            weights = None
            if objective == 'Sharpe':
                weights = self.port.optimization(
                    model='Classic',
                    rm=rm_map[risk_measure],
                    obj='Sharpe',
                    rf=self.risk_free_rate,
                    hist=True
                )
            elif objective == 'MinRisk':
                weights = self.port.optimization(
                    model='Classic',
                    rm=rm_map[risk_measure],
                    obj='MinRisk',
                    hist=True
                )
            elif objective == 'MaxRet':
                weights = self.port.optimization(
                    model='Classic',
                    rm=rm_map[risk_measure],
                    obj='MaxRet',
                    hist=True
                )
            elif objective == 'Utility':
                weights = self.port.optimization(
                    model='Classic',
                    rm=rm_map[risk_measure],
                    obj='Utility',
                    rf=self.risk_free_rate,
                    hist=True
                )
            else:
                logger.error(f"Unknown objective: {objective}")
                return None

            # Check if weights are valid
            if weights is None or weights.empty:
                logger.error("Optimization returned no weights")
                return None

            # Ensure weights is DataFrame format
            if isinstance(weights, pd.Series):
                weights = weights.to_frame()

            # Calculate portfolio metrics
            try:
                # Ensure weights and returns have aligned indices
                common_cols = self.returns.columns.intersection(weights.index)
                if len(common_cols) == 0:
                    logger.error("No common columns between returns and weights")
                    return None
                
                aligned_returns = self.returns[common_cols]
                aligned_weights = weights.loc[common_cols]
                
                portfolio_returns = aligned_returns @ aligned_weights
                portfolio_return = portfolio_returns.mean().iloc[0] * 252
                portfolio_vol = np.sqrt(aligned_weights.T @ (aligned_returns.cov() * 252) @ aligned_weights).iloc[0, 0]
                
                portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0

                # Calculate Sortino ratio
                downside_returns = portfolio_returns[portfolio_returns < 0]
                if len(downside_returns) > 0:
                    downside_vol = downside_returns.std().iloc[0] * np.sqrt(252)
                    portfolio_sortino = (portfolio_return - self.risk_free_rate) / downside_vol if downside_vol != 0 else 0
                else:
                    portfolio_sortino = float('inf')  # No downside returns
                
                return {
                    "weights": aligned_weights,
                    "return": portfolio_return,
                    "volatility": portfolio_vol,
                    "sharpe": portfolio_sharpe,
                    "sortino": portfolio_sortino
                }
                
            except Exception as e:
                logger.error(f"Error calculating portfolio metrics: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error during portfolio optimization: {str(e)}")
            return None
    
    def get_efficient_frontier(self, risk_measure: str = 'MV', n_points: int = 50) -> Optional[pd.DataFrame]:
        '''
        Get efficient frontier
        '''
        try:
            # Check if there is enough data
            if len(self.returns) < 2:
                logger.error("Not enough data for efficient frontier")
                return None

            # Map risk measures
            rm_map = {
                'MV': 'MV',
                'CVaR': 'CVaR',
                'MAD': 'MAD',
                'EVaR': 'EVaR',
                'CDaR': 'CDaR'
            }
            
            if risk_measure not in rm_map:
                logger.warning(f'Unsupported risk measure: {risk_measure}, using MV instead')
                risk_measure = 'MV'

            # Calculate efficient frontier
            frontier = self.port.efficient_frontier(
                model='Classic',
                rm=rm_map[risk_measure],
                points=n_points,
                rf=self.risk_free_rate
            )
            
            if frontier is None:
                logger.error("Efficient frontier calculation returned None")
                return None

            # Ensure frontier is DataFrame format
            if not isinstance(frontier, pd.DataFrame):
                logger.error(f"Efficient frontier is not a DataFrame: {type(frontier)}")
                return None
                
            return frontier
            
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {str(e)}")
            return None


# ==================== Prophet Forecasting ====================
class ProphetForecaster:
    """
    Using Prophet to predict prices
    """
    
    def forecast_prices(self, historical_prices: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
        """
        Using Prophet to predict prices
        
        Args:
            historical_prices: Historical price data
            forecast_days: Number of days to forecast
            
        Returns:
            pd.DataFrame: Forecasted prices
        """
        try:
            # Check if Prophet is installed
            try:
                from prophet import Prophet
            except ImportError:
                logger.warning("Prophet not installed, using simple forecast method")
                return self._simple_forecast(historical_prices, forecast_days)
            
            logger.info(f"Forecasting prices for {forecast_days} days...")
            forecasted = pd.DataFrame()
            
            for col in historical_prices.columns:
                try:
                    prices = historical_prices[col].dropna()
                    
                    if len(prices) < 30:
                        logger.warning(f"{col} has insufficient data for prediction")
                        continue

                    # Prepare data for Prophet
                    df = prices.reset_index()
                    df.columns = ['ds', 'y']

                    # Create and fit model
                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        seasonality_mode='multiplicative',
                        interval_width=0.95
                    )

                    # Suppress Prophet output
                    import logging as prophet_logging
                    prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)
                    
                    model.fit(df)

                    # Create future dataframe
                    future = model.make_future_dataframe(periods=forecast_days)

                    # Make predictions
                    forecast = model.predict(future)

                    # Extract forecasted values
                    forecast_values = forecast['yhat'].iloc[-forecast_days:].values
                    forecast_dates = forecast['ds'].iloc[-forecast_days:].values

                    # Add to results DataFrame
                    forecast_series = pd.Series(
                        forecast_values,
                        index=pd.to_datetime(forecast_dates),
                        name=col
                    )
                    
                    if forecasted.empty:
                        forecasted = pd.DataFrame(index=forecast_series.index)
                    
                    forecasted[col] = forecast_series
                    
                except Exception as e:
                    logger.warning(f"Failed to forecast {col}: {str(e)}")
                    continue
            
            if forecasted.empty:
                logger.warning("Prophet forecasting failed, using simple method")
                return self._simple_forecast(historical_prices, forecast_days)
            
            logger.info("Price forecasting completed")
            return forecasted
            
        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            return self._simple_forecast(historical_prices, forecast_days)
    
    def _simple_forecast(self, historical_prices: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
        """
        Simple forecasting method as fallback
        """
        logger.info("Using simple moving average forecast")

        # Create future dates
        last_date = historical_prices.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecasted = pd.DataFrame(index=future_dates)
        
        for col in historical_prices.columns:
            prices = historical_prices[col].dropna()

            # Use the last 30 days to calculate trend
            recent_days = min(30, len(prices))
            recent_prices = prices.iloc[-recent_days:]

            # Simple linear trend
            daily_return = recent_prices.pct_change().mean()
            last_price = prices.iloc[-1]

            # Generate forecasts with some randomness
            forecast = []
            for i in range(forecast_days):
                # Add small random variations
                noise = np.random.normal(0, prices.pct_change().std() * 0.5)
                next_price = last_price * (1 + daily_return + noise)
                forecast.append(next_price)
                last_price = next_price
            
            forecasted[col] = forecast
        
        return forecasted


# ==================== Portfolio Comparison ====================
class PortfolioComparison:
    """
    Portfolio performance comparison analysis
    """
    
    def __init__(self, prices_df: pd.DataFrame, forecast_days: int = 30):
        self.prices_df = prices_df
        self.forecast_days = forecast_days

        # Data split: 80% training, 20% testing
        self.split_idx = int(len(prices_df) * 0.8)
        self.train_prices = prices_df.iloc[:self.split_idx]
        self.test_prices = prices_df.iloc[self.split_idx:self.split_idx + forecast_days]

        # Ensure sufficient testing data
        if len(self.test_prices) < forecast_days:
            self.test_prices = prices_df.iloc[-forecast_days:] if len(prices_df) > forecast_days else prices_df.iloc[-len(prices_df)//5:]
    
    def run_comparison(self, risk_measure: str = 'MV') -> Dict:
        """
        Compare three different portfolios
        """
        results = {}
        
        try:
            # 1. Optimize based on historical data
            logger.info("Optimizing based on historical data...")
            hist_optimizer = Portfolio_Optimizer(self.train_prices)
            hist_result = hist_optimizer.optimize_portfolio('Sharpe', risk_measure)
            if hist_result:
                results['Historical'] = hist_result

            # 2. Optimize based on forecast data
            logger.info("Optimizing based on forecast data...")
            forecaster = ProphetForecaster()
            forecasted_prices = forecaster.forecast_prices(self.train_prices, self.forecast_days)
            
            if not forecasted_prices.empty:
                combined_prices = pd.concat([self.train_prices, forecasted_prices])
                forecast_optimizer = Portfolio_Optimizer(combined_prices)
                forecast_result = forecast_optimizer.optimize_portfolio('Sharpe', risk_measure)
                if forecast_result:
                    results['Forecast'] = forecast_result

            # 3. Optimize based on actual future data (ideal case)
            logger.info("Optimizing based on actual future data...")
            if len(self.test_prices) > 0:
                actual_combined = pd.concat([self.train_prices, self.test_prices])
                actual_optimizer = Portfolio_Optimizer(actual_combined)
                actual_result = actual_optimizer.optimize_portfolio('Sharpe', risk_measure)
                if actual_result:
                    results['Actual'] = actual_result

            # Calculate actual test period performance
            if len(self.test_prices) > 0 and results:
                test_returns = self.test_prices.pct_change().dropna()
                
                for name, result in results.items():
                    if result and 'weights' in result:
                        try:
                            # Align weights and test returns columns
                            common_cols = test_returns.columns.intersection(result['weights'].index)
                            if len(common_cols) == 0:
                                logger.warning(f"No common columns between test returns and {name} weights")
                                continue
                                
                            aligned_weights = result['weights'].loc[common_cols]
                            aligned_returns = test_returns[common_cols]
                            
                            portfolio_returns = aligned_returns @ aligned_weights
                            # Check if portfolio_returns is Series or DataFrame
                            if hasattr(portfolio_returns, 'values'):
                                portfolio_returns = portfolio_returns.values.flatten()
                            else:
                                portfolio_returns = portfolio_returns.to_numpy().flatten()

                            # Calculate metrics
                            actual_return = np.mean(portfolio_returns) * 252
                            actual_vol = np.std(portfolio_returns) * np.sqrt(252)
                            actual_sharpe = (actual_return - 0.02) / actual_vol if actual_vol != 0 else 0

                            # Calculate Sortino ratio
                            downside_returns = portfolio_returns[portfolio_returns < 0]
                            if len(downside_returns) > 0:
                                downside_vol = np.std(downside_returns) * np.sqrt(252)
                                actual_sortino = (actual_return - 0.02) / downside_vol if downside_vol != 0 else 0
                            else:
                                actual_sortino = float('inf')
                            
                            result['test_return'] = actual_return
                            result['test_volatility'] = actual_vol
                            result['test_sharpe'] = actual_sharpe
                            result['test_sortino'] = actual_sortino
                            
                        except Exception as e:
                            logger.warning(f"Failed to calculate test performance for {name}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in portfolio comparison: {str(e)}")
        
        return results