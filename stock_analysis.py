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


def kelly_fraction(win_prob, win_loss_ratio):
    if win_loss_ratio > 0:  # To avoid division by zero
        return max((win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio, 0)
    else:
        return 0

def calculate_win_loss_ratios(stock_data):
    wins = stock_data[stock_data['Next Day Return'] > 0]
    losses = stock_data[stock_data['Next Day Return'] <= 0]
    win_prob = len(wins) / len(stock_data) if len(stock_data) > 0 else 0
    average_win = wins['Next Day Return'].mean() if len(wins) > 0 else 0
    average_loss = losses['Next Day Return'].mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else 0
    return win_prob, win_loss_ratio

# Backtesting function using the Kelly Criterion
def backtest_kelly_strategy(stock_data, initial_capital, max_risk, risk_per_trade, kelly_fractional):
    portfolio_values = [initial_capital]
    capital = initial_capital

    win_prob, win_loss_ratio = calculate_win_loss_ratios(stock_data[stock_data['Trading Signal']])

    for index, row in stock_data.iterrows():
        if row['Trading Signal']:
            kelly_f = kelly_fraction(win_prob, win_loss_ratio) * kelly_fractional
            # Trade size:
            trade_size = min(kelly_f * capital, risk_per_trade * capital)
            trade_size = min(trade_size, max_risk * initial_capital - capital)
            
            # Simulating
            trade_outcome = trade_size * row['Next Day Return']
            capital += trade_outcome
            
            # Risk constraint
            capital = max(capital, initial_capital * (1 - max_risk))
            
        portfolio_values.append(capital)

    return portfolio_values
