import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def analyze_stock_return(stock_ticker, start_date):
    # Load historical data for the stock
    stock_data = yf.download(stock_ticker, start=start_date)

    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change() * 100

    # Filter the days where the stock fell by 5% or more
    down_days = stock_data[stock_data['Daily Return'] <= -5]

    # Shift the data to get the next day's returns
    next_day_returns = stock_data['Daily Return'].shift(-1)

    # Align the down days with their next day's returns
    aligned_next_day_returns = next_day_returns.loc[down_days.index]

    # Drop NaN values that may have been added by the shift at the end
    aligned_next_day_returns.dropna(inplace=True)

    # Calculate the likelihood of an increase after a -5% drop
    likelihood_increase = (aligned_next_day_returns > 0).mean()

    # Return the aligned next day returns and the likelihood
    return aligned_next_day_returns, likelihood_increase

def plot_return_series(returns_series):
    # Plot the results
    plt.scatter(returns_series.index, returns_series, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Next Day Returns After a 5% Drop")
    plt.xlabel("Date")
    plt.ylabel("Next Day Return (%)")
    plt.grid(True)
    plt.show()
