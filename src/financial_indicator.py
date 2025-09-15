import pandas as pd
import numpy as np
from typing import Dict,Union,Optional
from scipy import stats
from datetime import datetime

class FinancialMetrics:
    '''
    Financial indicator metrics (support )
    '''
    def __init__(self,risk_free_rate:float = 0.02, frequency:str = 'daily'):
        '''
        Initialize financial metrics calculator

        Args:
            risk_free_rate (float): The risk-free rate used in calculations. Default is 0.02 (2%).
            frequency (str): The frequency of the data ('daily', 'weekly', 'monthly', 'yearly'). Default is 'daily'.
        '''
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

        # set diffrent periods for different frequency 设置不同频率的年华因子
        self.annual_factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'yearly': 1
        }

        if frequency not in self.annual_factors:
            raise ValueError("Frequency must be one of 'daily', 'weekly', 'monthly', or 'yearly'.")
        
    def calculate_returns(self,price:pd.Series) ->pd.Series:
        '''
        Calculate the returns of a price series

        Args:
            price (pd.Series): Series of prices

        Returns:
            pd.Series: Series of returns
        '''
        returns = price.pct_change().dropna()
        return returns
    
    def get_annual_factor(self) -> int:
        '''
        Get the annualization factor based on the frequency

        Returns:
            int: Annualization factor
        '''
        return self.annual_factors[self.frequency]
    
    def annual_return(self, prices: pd.Series) -> float:
        '''
        Calculate the annualized return 年化收益率

        Args:
            prices (pd.Series): Series of prices

        Returns:
            float: Annualized return
        '''
        if len(prices) < 2:
            return 0.0
        
        # calculate total return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        # choose calculation method based on frequency
        if self.frequency == 'yearly':
            return total_return
        else:
            # calculate using calendar days
            days = (prices.index[-1] - prices.index[0]).days
            if days > 0:
                annual_ret = (1 + total_return) ** (365.25 / days) - 1
                return annual_ret
            else:
                return 0.0
            
    def annual_volatility(self,prices:pd.Series) -> float:
        '''
        Calculate the annualized volatility 年化波动率

        Args:
            returns (pd.Series): Series of returns

        Returns:
            float: Annualized volatility
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 2:
            return 0.0
        
        # using yearly factor to annualize volatility
        period_vol = returns.std()
        annual_vol = period_vol * np.sqrt(self.get_annual_factor())

        return annual_vol
    
    def sharpe_ratio(self,prices:pd.Series) -> float:
        '''
        Calculate the Sharpe Ratio 夏普比率
        Args:
            prices (pd.Series): Series of prices
            
        Returns:
            float: Sharpe Ratio
        '''
        ann_return = self.annual_return(prices)
        ann_vol = self.annual_volatility(prices)

        if ann_vol == 0:
            return 0.0
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        return sharpe
    
    def sortino_ratio(self,prices:pd.Series) -> float:
        '''
        Calculate the Sortino Ratio 索提诺比率

        Args:
            prices (pd.Series): Series of prices

        Returns:
            float: Sortino Ratio
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 2:
            return 0.0
        
        ann_return = self.annual_return(prices)

        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf # infinite downside risk
        
        # annualize downside deviation
        downside_vol = downside_returns.std() * np.sqrt(self.get_annual_factor())

        if downside_vol == 0:
            return 0.0

        sortino = (ann_return - self.risk_free_rate) / downside_vol
        return sortino
    
    def max_drawdown(self,prices:pd.Series) -> float:
        '''
        Calculate the Maximum Drawdown 最大回撤

        Args:
            prices (pd.Series): Series of prices

        Returns:
            float: Maximum Drawdown
        '''
        if len(prices) < 2:
            return 0.0
        
        peak = prices.expanding(min_periods=1).max()
        drawdowns = (prices - peak) / peak
        max_dd = drawdowns.min()

        return max_dd
    
    def skewness(self,prices:pd.Series) -> float:
        '''
        Calculate the Skewness 偏度

        Args:
            prices (pd.Series): Series of prices

        Returns:
            float: Skewness
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 3:
            return 0.0
        
        skewness = stats.skew(returns,nan_policy='omit')
        return skewness
    
    def kurtosis(self,prices:pd.Series) -> float:
        '''
        Calculate the Kurtosis 峰度

        Args:
            prices (pd.Series): Series of prices

        Returns:
            float: Kurtosis
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 4:
            return 0.0
        
        kurt = stats.kurtosis(returns,fisher=True,nan_policy='omit')
        return kurt
    
    def value_at_risk(self,prices:pd.Series, confidence_level:float = 0.95) -> float:
        '''
        Calculate the Value at Risk (VaR) 风险价值

        Args:
            prices (pd.Series): Series of prices
            confidence_level (float): Confidence level for VaR calculation. Default is 0.95.

        Returns:
            float: Value at Risk
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 2:
            return 0.0
        
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        
        var = np.percentile(returns.dropna(), (1 - confidence_level) * 100)
        return var
    
    def conditional_value_at_risk(self,prices:pd.Series, confidence_level:float = 0.95) -> float:
        '''
        Calculate the Conditional Value at Risk (CVaR) 条件风险价值

        Args:
            prices (pd.Series): Series of prices
            confidence_level (float): Confidence level for CVaR calculation. Default is 0.95.

        Returns:
            float: Conditional Value at Risk
        '''
        returns = self.calculate_returns(prices)
        if len(returns) < 2:
            return 0.0
        
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        
        var = self.value_at_risk(prices, confidence_level)
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        cvar = tail_returns.mean()
        return cvar
    
    # =================== Multi-asset metrics: Covariance Matrix & Correlation Matrix===================

    def covariance_matrix(self,prices_df:pd.DataFrame) ->pd.DataFrame:
        '''
        Calculate the covariance matrix of returns 收益协方差矩阵：两只资产的收益一起上升或一起下降的程度

        Args:
            prices_df(pd.DataFrame): DataFrame of prices, columns are different assets
        
        Returns:
            pd.DataFrame: Covariance matrix of returns
        '''
        # calculate returns for each asset
        returns_df = prices_df.pct_change().dropna()

        if len(returns_df) < 2:
            return pd.DataFrame()
        
        #calculate covariance matrix and annualize it 年化协方差矩阵
        cov_matrix = returns_df.cov() * self.get_annual_factor()
        return cov_matrix
    
    def correlation_matrix(self,prices_df:pd.DataFrame) ->pd.DataFrame:
        '''
        Calculate the correlation matrix of returns 收益率相关系数矩阵

        Args:
            prices_df (pd.DataFrame): DataFrame of prices, each column is a different asset

        Returns:
            pd.DataFrame: Correlation matrix of returns
        '''
        # Calculate returns for all assets
        returns_df = prices_df.pct_change().dropna()
        
        if len(returns_df) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix 相关系数矩阵：不年化
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
# =========================== Summary Metrics ================================

    def calculate_all_metrics(self,prices:pd.Series,symbol:str = None) -> Dict[str, Union[str, float]]:
        '''
        Calculate all financial metrics and return as a dictionary

        Args:
            prices (pd.Series): Series of prices
            symbol (str, optional): Symbol or identifier for the asset. Default is None.

        Returns:
            Dict[str, Union[str, float]]: Dictionary of all financial metrics
        '''
        metrics = {
            'Symbol': symbol if symbol else 'N/A',
            'Annual Return': self.annual_return(prices),
            'Annual Volatility': self.annual_volatility(prices),
            'Sharpe Ratio': self.sharpe_ratio(prices),
            'Sortino Ratio': self.sortino_ratio(prices),
            'Max Drawdown': self.max_drawdown(prices),
            'Skewness': self.skewness(prices),
            'Kurtosis': self.kurtosis(prices),
            'VaR(95%)': self.value_at_risk(prices),
            'CVaR(95%)': self.conditional_value_at_risk(prices)
        }

        return metrics
    

def analyze_portfolio_metrics(data_path:str,frequency:str = 'daily',symbols:list = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
        
    Analyze portfolio metrics from a CSV file containing price data

    Args:
        data_path (str): Path to the CSV file containing price data
        frequency (str): Frequency of the data ('daily', 'weekly', 'monthly', 'yearly'). Default is 'daily'.
        symbols (list, optional): List of symbols to analyze. If None, analyze all columns. Default is None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - DataFrame containing financial metrics for each symbol
            - Covariance matrix of returns
            - Correlation matrix of returns
    '''
        # load data
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f'Data loaded successfully from {data_path},shape: {data.shape}')
    except Exception as e:
        print(f'Error loading data: {e}')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # Using all columns if symbols not provided
    if symbols is None:
        symbols = data.columns.tolist()

    # Filter data to only include specified symbols
    data = data[symbols].dropna()

    # create FinancialMetrics instance
    calculator = FinancialMetrics(risk_free_rate=0.02, frequency=frequency)

    # save all results
    results = {}

    # calculate metrics for each symbol
    for symbol in symbols:
        if symbol in data.columns:
            print(f"calculating metrics for {symbol}...")
            prices = data[symbol]

            if len(prices) > 1:
                metrics = calculator.calculate_all_metrics(prices, symbol)
                results[symbol] = metrics
            else:
                print(f"Not enough data to calculate metrics for {symbol}.")
        else:
            print(f"Symbol {symbol} not found in data columns.")

    # convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # format float columns to 4 decimal places
    float_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[float_columns] = results_df[float_columns].round(4)

    # calculate covariance and correlation matrices
    print("Calculating covariance and correlation matrices...")
    cov_matrix = calculator.covariance_matrix(data)
    corr_matrix = calculator.correlation_matrix(data)

    return results_df, cov_matrix, corr_matrix


if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'DIS', 'JPM', 'META']

    # analyze diffrent frequency data
    frequencies = ['daily', 'weekly', 'monthly']
    
    for freq in frequencies:
        print(f"\n{'='*50}")
        print(f"Analyzing {freq} data...")
        print(f"{'='*50}")

        file_path = f'data/{freq}_stock_prices.csv'

        try:
            metrics_df, cov_matrix, corr_matrix = analyze_portfolio_metrics(file_path, freq, symbols)

            # Display metrics
            print(f"\nFinancial Metrics for {freq} data:")
            print(metrics_df)

            # Display covariance matrix
            print(f"\nCovariance Matrix for {freq} data:")
            print(cov_matrix.round(6))
            
            # Display correlation matrix
            print(f"\nCorrelation Matrix for {freq} data:")
            print(corr_matrix.round(4))

            # saving results
            metrics_df.to_csv(f'data/{freq}_financial_metrics.csv')
            cov_matrix.to_csv(f'data/{freq}_covariance_matrix.csv')
            corr_matrix.to_csv(f'data/{freq}_correlation_matrix.csv')

            print(f"\nResults saved to:")
            print(f"\nResults already saved in'data/{freq}_financial_metrics.csv'")
            print(f"- data/{freq}_covariance_matrix.csv")
            print(f"- data/{freq}_correlation_matrix.csv")

        except Exception as e:
            print(f'Failed{e} when analyzing {freq}data')

    
    # analyze yearly data (if we have)
    try:
        print(f"\n{'='*50}")
        print("Analyzing yearly data...")
        print(f"{'='*50}")
        
        yearly_metrics, yearly_cov, yearly_corr = analyze_portfolio_metrics('data/yearly_stock_prices.csv', 'yearly', symbols)
        
        # Display metrics
        print(f"\nFinancial Metrics for yearly data:")
        print(yearly_metrics)
        
        # Display covariance matrix
        print(f"\nCovariance Matrix for yearly data:")
        print(yearly_cov.round(6))
        
        # Display correlation matrix
        print(f"\nCorrelation Matrix for yearly data:")
        print(yearly_corr.round(4))

        yearly_metrics.to_csv('data/yearly_financial_metrics.csv')
        yearly_cov.to_csv('data/yearly_covariance_matrix.csv')
        yearly_corr.to_csv('data/yearly_correlation_matrix.csv')
        
        print(f"\nResults saved to:")
        print(f"- data/yearly_financial_metrics.csv")
        print(f"- data/yearly_covariance_matrix.csv")
        print(f"- data/yearly_correlation_matrix.csv")
        
    except FileNotFoundError:
        print(f'\nYearly data file not found, skipping yearly data analysis.')
    except Exception as e:
        print(f'Failed {e} when analyzing yearly data.')