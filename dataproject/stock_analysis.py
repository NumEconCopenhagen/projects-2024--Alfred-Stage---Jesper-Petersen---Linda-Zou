import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def analyze_stock_return(stock_ticker, start_date):
    stock_data = yf.download(stock_ticker, start=start_date)

    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change() * 100

    down_days = stock_data[stock_data['Daily Return'] <= -5]

    next_day_returns = stock_data['Daily Return'].shift(-1)

    aligned_next_day_returns = next_day_returns.loc[down_days.index]

    aligned_next_day_returns.dropna(inplace=True)

    likelihood_increase = (aligned_next_day_returns > 0).mean()

    return aligned_next_day_returns, likelihood_increase

def plot_return_series(returns_series):
    plt.scatter(returns_series.index, returns_series, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Next Day Returns After a 5% Drop")
    plt.xlabel("Date")
    plt.ylabel("Next Day Return (%)")
    plt.grid(True)
    plt.show()
