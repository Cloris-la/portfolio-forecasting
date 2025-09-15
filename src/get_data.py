import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import os

def fetch_stock_data(symbols,start_date,end_date,interval='1d'):
    '''
    Fetching stock data from Yahoo Finance

    Args:
        symbols (list): Stock symbol list, eg:['AAPL', 'MSFT']
        start_date (str): Start date, format 'YYYY-MM-DD'
        end_date (str): End date, format 'YYYY-MM-DD'
        interval (str): Data interval, optional '1d'(daily), '1wk'(weekly), '1mo'(monthly)

    Returns:
        DataFrame: DataFrame containing all stock adjusted close prices
    '''
    try:
        # download data
        data = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=True
        )

        if len(symbols) == 1:
            close_prices = data['Close'].to_frame()
            close_prices.columns = [symbols[0]]
        else:
            close_prices = pd.DataFrame()
            for symbol in symbols:
                if symbol in data:
                    close_prices[symbol] = data[symbol]['Close']

        close_prices.dropna(inplace=True)
        print(f"Fetched {len(close_prices)} row data")
        return close_prices

    except Exception as e:
        print(f"Failed fetching data:{str(e)}")

# create data folder
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"Created directory: {data_folder}")
else:
    print(f"Directory already exists: {data_folder}")

# setting parameters
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA','NFLX','NVDA','DIS','JPM','META']
start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# get data
print('Fetching daily data...')
daily_data = fetch_stock_data(symbols,start_date,end_date,'1d')

print('\n Fetching weekly data...')
weekly_data = fetch_stock_data(symbols,start_date,end_date,'1wk')

print('\n Fetching monthly data...')
monthly_data = fetch_stock_data(symbols,start_date,end_date,'1mo')

# yearly data is setted as last month adjusted close prices 按年聚合（取每年最后一个月的收盘价）
if monthly_data is not None:
    yearly_data = monthly_data.resample('Y').last()

# save data
if daily_data is not None:
    daily_path = os.path.join(data_folder, 'daily_stock_prices.csv')
    daily_data.to_csv(daily_path)
    print(f"Daily data saved to {daily_path}")

if weekly_data is not None:
    weekly_path = os.path.join(data_folder, 'weekly_stock_prices.csv')
    weekly_data.to_csv(weekly_path)
    print(f"Weekly data saved to {weekly_path}")

if monthly_data is not None:
    monthly_path = os.path.join(data_folder, 'monthly_stock_prices.csv')
    monthly_data.to_csv(monthly_path)
    print(f"Monthly data saved to {monthly_path}")

if yearly_data is not None:
    yearly_path = os.path.join(data_folder, 'yearly_stock_prices.csv')
    yearly_data.to_csv(yearly_path)
    print(f"Yearly data saved to {yearly_path}")

# simple view of data
if daily_data is not None:
    print("\nFirst 5 rows of daily data:")
    print(daily_data.head())