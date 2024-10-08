from yahoo_fin.stock_info import get_data
import yfinance as yf
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import scipy.stats
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", message="Passing literal html to 'read_html' is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance.utils", message="The 'unit' keyword in TimedeltaIndex construction is deprecated")

class MeanReversionAlgo:
    def __init__(self, index='NASDAQ', start_date="01/01/2020", end_date="22/03/2024"):
        """
        Initializes the MeanReversionAlgo class with the specified parameters.
        
        :param index: Stock market index to analyze (default is 'NASDAQ').
        :param start_date: Start date for fetching historical data (default is '01/01/2020').
        :param end_date: End date for fetching historical data (default is '22/03/2024').
        """
        self.index = index
        self.start_date = start_date
        self.end_date = end_date
        self.historical_data = {}
        self.failed_tickers = []
        self.merged_df = None  

    def fetch_tickers(self):
        """
        Fetches a list of tickers based on the index specified during initialization.
        
        :return: List of stock tickers.
        """
        if self.index.upper() == 'NASDAQ':
            ticker_list = si.tickers_nasdaq()
        elif self.index.upper() == 'DOW':
            ticker_list = si.tickers_dow()
        elif self.index.upper() == 'NIFTY50':
            ticker_list = si.tickers_nifty50()
        else:
            print("Index not supported.")
            return []
        return [ticker for ticker in ticker_list if not ticker.endswith('W')]

    def fetch_data_parallel(self, ticker_list):
        """
        Fetches historical data for the given list of tickers in parallel.
        
        :param ticker_list: List of stock tickers.
        """
        def fetch_data(ticker):
            try:
                data = get_data(ticker, start_date=self.start_date, end_date=self.end_date, interval="1d")
                return ticker, data, None
            except Exception as e:
                return ticker, None, e

        tickers_total = len(ticker_list)
        tickers_done = 0

        with ThreadPoolExecutor(max_workers=7) as executor:
            future_to_ticker = {executor.submit(fetch_data, ticker): ticker for ticker in ticker_list}

            for future in as_completed(future_to_ticker):
                tickers_done += 1
                progress_percentage = (tickers_done / tickers_total) * 100
                print(f"\rFetching data progress: {progress_percentage:.2f}%", end="")

                ticker, data, error = future.result()
                if error:
                    self.failed_tickers.append(ticker)
                else:
                    self.historical_data[ticker] = data

    def calculate_pct_change(self):
        """
        Calculates the daily percentage change for each ticker's historical data.
        """
        for ticker, data in self.historical_data.items():
            self.historical_data[ticker]['Pct_Change'] = data['close'].pct_change(fill_method=None) * 100

    def find_winners_and_losers(self):
        """
        Finds the daily winners and losers and their next day's performance.
        """
        all_dates = sorted(self.historical_data[list(self.historical_data.keys())[0]].index.unique())
        daily_analysis = []

        for i in range(len(all_dates) - 2):
            current_date = all_dates[i]
            next_date = all_dates[i + 1]
            day_after_next = all_dates[i + 2]

            pct_changes = {}
            for company, df in self.historical_data.items():
                if current_date in df.index and next_date in df.index:
                    df_filtered = df.loc[[current_date, next_date], 'close']
                    pct_change = df_filtered.pct_change(fill_method=None).iloc[-1] * 100
                    pct_changes[company] = pct_change

            if not pct_changes:
                continue  

            biggest_loser_company, biggest_loss_pct = min(pct_changes.items(), key=lambda x: x[1])
            biggest_winner_company, biggest_win_pct = max(pct_changes.items(), key=lambda x: x[1])

            next_day_pct_change_loser = None
            next_day_pct_change_winner = None
            if day_after_next in self.historical_data[biggest_loser_company].index:
                next_day_pct_change_loser = self.historical_data[biggest_loser_company].loc[day_after_next, 'Pct_Change']
            if day_after_next in self.historical_data[biggest_winner_company].index:
                next_day_pct_change_winner = self.historical_data[biggest_winner_company].loc[day_after_next, 'Pct_Change']

            daily_analysis.append({
                'Date': current_date,
                'Biggest Loser': biggest_loser_company,
                'Loss %': biggest_loss_pct,
                'Next Day Change % (Loser)': next_day_pct_change_loser,
                'Biggest Winner': biggest_winner_company,
                'Win %': biggest_win_pct,
                'Next Day Change % (Winner)': next_day_pct_change_winner
            })

        self.analysis_df = pd.DataFrame(daily_analysis)
        self._calculate_index_performance(all_dates)

    def _calculate_index_performance(self, all_dates):
        """
        Calculates the overall performance of the index for each day.
        
        :param all_dates: List of all unique dates in the historical data.
        """
        index_performance = {}
        for date in all_dates[:-1]:
            daily_pct_changes = [df.loc[date, 'Pct_Change'] for company, df in self.historical_data.items() if date in df.index]
            if daily_pct_changes:
                index_performance[date] = np.mean(daily_pct_changes)
        
        index_performance_df = pd.DataFrame(list(index_performance.items()), columns=['Date', 'Index Performance'])
        self.merged_df = pd.merge(self.analysis_df, index_performance_df, on='Date')

    def calculate_correlations(self):
        """
        Calculates correlations and conditional correlations for the analysis results.
        """
        if not hasattr(self, 'merged_df'):
            print("Error: Analysis data not found.")
            return

        merged_df = self.merged_df

        def calculate_and_print_correlation(df, x_col, y_col, description):
            clean_df = df.dropna(subset=[x_col, y_col])
            clean_df = clean_df[np.isfinite(clean_df[x_col]) & np.isfinite(clean_df[y_col])]

            if len(clean_df) < 2 or clean_df[x_col].nunique() < 2 or clean_df[y_col].nunique() < 2:
                print(f"Insufficient data for {description} correlation calculation.")
                return

            correlation, _ = scipy.stats.pearsonr(clean_df[x_col], clean_df[y_col])
            n = len(clean_df)
            t_stat = correlation * np.sqrt((n-2) / (1-correlation**2))
            print(f"{description}: Correlation: {correlation:.2f}, T-statistic: {t_stat:.2f}, Observations: {n}")

        calculate_and_print_correlation(merged_df, 'Next Day Change % (Loser)', 'Loss %', 'General')
        positive_index_df = merged_df[merged_df['Index Performance'] > 0]
        negative_index_df = merged_df[merged_df['Index Performance'] < 0]

        calculate_and_print_correlation(positive_index_df, 'Next Day Change % (Loser)', 'Loss %', 'Positive Index Performance')
        calculate_and_print_correlation(negative_index_df, 'Next Day Change % (Loser)', 'Loss %', 'Negative Index Performance')

    def _print_conditional_correlations(self, df, condition):
        """
        Helper function to print conditional correlations.
        
        :param df: DataFrame containing the data.
        :param condition: Condition description for the correlation.
        """
        if len(df) >= 2 and df['Next Day Change % (Loser)'].nunique() > 1 and df['Loss %'].nunique() > 1:
            correlation, _ = scipy.stats.pearsonr(df['Next Day Change % (Loser)'], df['Loss %'])
            n = len(df)
            t_stat = correlation * np.sqrt((n-2) / (1-correlation**2))
            print(f"{condition}: Correlation: {correlation:.2f}, T-statistic: {t_stat:.2f}, Observations: {n}")
        else:
            print(f"Insufficient or constant data for {condition}. Correlation calculation not possible.")

    def _print_threshold_correlations(self, df):
        """
        Helper function for printing correlations based on loss thresholds.
        
        :param df: DataFrame containing the data.
        """
        thresholds = [(-7, None), (-7, -3), (-3, 0)]
        for lower, upper in thresholds:
            filtered_df = df[df['Loss %'] < upper] if upper else df[df['Loss %'] < lower]
            if lower is not None:
                filtered_df = filtered_df[filtered_df['Loss %'] >= lower]

            if len(filtered_df) >= 2 and filtered_df['Next Day Change % (Loser)'].nunique() > 1 and filtered_df['Loss %'].nunique() > 1:
                correlation, _ = scipy.stats.pearsonr(filtered_df['Next Day Change % (Loser)'], filtered_df['Loss %'])
                n = len(filtered_df)
                t_stat = correlation * np.sqrt((n-2) / (1-correlation**2))
                print(f"Correlation (Loss {lower}% to {upper}%): {correlation:.2f}, T-statistic: {t_stat:.2f}, Observations: {n}")
            else:
                print(f"Insufficient or constant data for loss threshold {lower}% to {upper}%. Correlation calculation not possible.")

    def plot_results(self, index_condition=None, category_condition=None):
        """
        Plots the results of the analysis.
        
        :param index_condition: Condition for the index performance ('up' or 'down').
        :param category_condition: Dictionary specifying the category condition and thresholds.
        """
        if not hasattr(self, 'merged_df') or self.merged_df.empty:
            print("Error: Analysis data not found or empty.")
            return

        df = self.merged_df
        x_label = 'Win %'
        y_label = 'Next Day Change % (Winner)'
        title = 'Next-Day Performance of Daily Biggest Winners'

        if index_condition == 'up':
            df = df[df['Index Performance'] > 0]
            title += ' on Positive Index Days'
        elif index_condition == 'down':
            df = df[df['Index Performance'] < 0]
            title += ' on Negative Index Days'

        if category_condition:
            category_type = category_condition['type']
            threshold = category_condition['thresholds'][0]

            if category_type == 'winner':
                df = df[df['Win %'] > threshold]
            elif category_type == 'loser':
                x_label = 'Loss %'
                y_label = 'Next Day Change % (Loser)'
                df = df[df['Loss %'] < -threshold]
                title = 'Next-Day Performance of Daily Biggest Losers'

        if df.empty:
            print("No data meets the specified conditions.")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_label], df[y_label], alpha=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(0, color='grey', linestyle='--')
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executes the mean reversion analysis pipeline.
        """
        tickers = self.fetch_tickers()
        if not tickers:
            print("No tickers fetched. Exiting...")
            return

        print(f"Fetching data for {len(tickers)} tickers in the {self.index} index... If Nasdaq, please be patient. Make some coffee?")
        self.fetch_data_parallel(tickers)

        if self.failed_tickers:
            print(f"Failed to fetch data for {len(self.failed_tickers)} tickers.")
        else:
            print("Successfully fetched data for all tickers.")

        print("Calculating percentage changes for all tickers...")
        self.calculate_pct_change()

        self.find_winners_and_losers()
        print("Calculating correlations...")
        self.calculate_correlations()
        print("Plotting results...")
        self.plot_results()
        print("Analysis complete.")

    def calculate_segmented_correlation(self, index_condition=None, category_condition=None):
        """
        Calculates segmented correlation based on the given conditions.
        
        :param index_condition: Condition for the index performance ('up' or 'down').
        :param category_condition: Dictionary specifying the category condition and thresholds.
        """
        if not hasattr(self, 'merged_df') or self.merged_df.empty:
            print("Error: Analysis data not found or empty.")
            return

        df = self.merged_df.copy()

        if index_condition == 'up':
            df = df[df['Index Performance'] > 0]
        elif index_condition == 'down':
            df = df[df['Index Performance'] < 0]

        column_type = 'Win %' if category_condition and category_condition['type'] == 'winner' else 'Loss %'
        y_col = 'Next Day Change % (Winner)' if category_condition and category_condition['type'] == 'winner' else 'Next Day Change % (Loser)'

        if category_condition:
            lower_threshold, upper_threshold = category_condition.get('thresholds', (None, None))
            if lower_threshold is not None:
                df = df[df[column_type] >= lower_threshold]
            if upper_threshold is not None:
                df = df[df[column_type] <= upper_threshold]

        df = df.dropna(subset=[column_type, y_col])
        df = df[np.isfinite(df[column_type]) & np.isfinite(df[y_col])]

        if len(df) >= 2 and df[column_type].nunique() > 1:
            correlation, p_value = scipy.stats.pearsonr(df[column_type], df[y_col])
            print(f"Correlation: {correlation:.3f}, P-value: {p_value:.3g}, Sample size: {len(df)}")
        else:
            print("Insufficient or non-varied data for correlation calculation.")

    def analyze_stock_return(self, stock_ticker, start_date, verbose=True, plot=True):
        """
        Analyzes the stock return after a significant drop.
        
        :param stock_ticker: Stock ticker symbol.
        :param start_date: Start date for fetching historical data.
        :param verbose: If True, prints the likelihood of an increase after a drop.
        :param plot: If True, plots the next day returns after a significant drop.
        :return: Series of next day returns after a significant drop and the likelihood of an increase.
        """
        stock_data = yf.download(stock_ticker, start=start_date)
        self.historical_data[stock_ticker] = stock_data

        stock_data['Daily Return'] = stock_data['Adj Close'].pct_change() * 100
        down_days = stock_data[stock_data['Daily Return'] <= -5]
        next_day_returns = stock_data['Daily Return'].shift(-1)
        aligned_next_day_returns = next_day_returns.loc[down_days.index]
        aligned_next_day_returns.dropna(inplace=True)
        likelihood_increase = (aligned_next_day_returns > 0).mean()
        stock_data['Trading Signal'] = stock_data['Daily Return'] <= -5
        stock_data['Next Day Return'] = stock_data['Daily Return'].shift(-1)

        if verbose:
            print(f"The likelihood of an increase after a -5% drop for {stock_ticker} is: {likelihood_increase:.2%}")

        if plot:
            self.plot_return_series(aligned_next_day_returns)

        return aligned_next_day_returns, likelihood_increase

    def plot_return_series(self, returns_series):
        """
        Plots the series of returns.
        
        :param returns_series: Series of returns to plot.
        """
        plt.scatter(returns_series.index, returns_series, color='blue')
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Next Day Returns After a 5% Drop")
        plt.xlabel("Date")
        plt.ylabel("Next Day Return (%)")
        plt.grid(True)
        plt.show()

    def kelly_fraction(self, win_prob, win_loss_ratio):
        """
        Calculates the Kelly fraction.
        
        :param win_prob: Probability of a win.
        :param win_loss_ratio: Ratio of win to loss.
        :return: Kelly fraction.
        """
        if win_loss_ratio > 0:
            return max((win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio, 0)
        else:
            return 0

    def calculate_win_loss_ratios(self, stock_data):
        """
        Calculates win probability and win/loss ratio.
        
        :param stock_data: DataFrame containing stock data.
        :return: Tuple of win probability and win/loss ratio.
        """
        wins = stock_data[stock_data['Next Day Return'] > 0]
        losses = stock_data[stock_data['Next Day Return'] <= 0]
        win_prob = len(wins) / len(stock_data) if len(stock_data) > 0 else 0
        average_win = wins['Next Day Return'].mean() if len(wins) > 0 else 0
        average_loss = losses['Next Day Return'].mean() if len(losses) > 0 else 0
        win_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else 0

        return win_prob, win_loss_ratio

    def backtest_kelly_strategy(self, initial_capital, max_risk, risk_per_trade, kelly_fractional):
        """
        Backtests the Kelly strategy.
        
        :param initial_capital: Initial capital for the strategy.
        :param max_risk: Maximum risk per trade as a fraction of capital.
        :param risk_per_trade: Risk per trade as a fraction of capital.
        :param kelly_fractional: Fraction of Kelly criterion to use.
        :return: List of portfolio values over time.
        """
        if not self.historical_data:
            print("No historical data available for backtesting.")
            return []

        capital = initial_capital
        portfolio_values = [capital]

        for ticker, data in self.historical_data.items():
            if 'Trading Signal' not in data.columns or 'Next Day Return' not in data.columns:
                continue

            win_prob, win_loss_ratio = self.calculate_win_loss_ratios(data[data['Trading Signal']])
            
            for index, row in data.iterrows():
                if row['Trading Signal']:
                    kelly_f = self.kelly_fraction(win_prob, win_loss_ratio) * kelly_fractional
                    max_trade_risk = capital * max_risk
                    trade_risk = min(kelly_f * capital, risk_per_trade * capital)
                    trade_size = min(trade_risk, max_trade_risk)
                    trade_outcome = trade_size * row['Next Day Return']
                    capital += trade_outcome
                    portfolio_values.append(capital)

        return portfolio_values
    
    def plot_portfolio_performance(self, portfolio_values, initial_capital):
        """
        Plots the portfolio performance over time.
        
        :param portfolio_values: List of portfolio values over time.
        :param initial_capital: Initial capital for reference.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values, label='Portfolio Value')
        plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
        plt.title("Portfolio Performance Using Kelly Criterion")
        plt.xlabel("Number of Trades")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()
