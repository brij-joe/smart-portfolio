import os
import logging
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from pypfopt import EfficientFrontier, objective_functions
from pypfopt.expected_returns import mean_historical_return, capm_return
from pypfopt.risk_models import CovarianceShrinkage, exp_cov
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from config.model_config import ModelConfig
from config.stock_tickers import yf_tickers
from data.data_provider import DataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================
# Plotting Utilities
# ==========================
def plot_barh(series, xlabel, title):
    plt.figure(figsize=(10, 6))
    series.sort_values().plot(kind="barh")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==========================
# Portfolio Utilities
# ==========================
class PortfolioOptimizer:

    def __init__(self, df_prices: pd.DataFrame, risk_free_rate: float):
        self.df = df_prices
        self.rfr = risk_free_rate

    def compute_annual_metrics(self):
        returns = self.df.pct_change().dropna()

        annual_return = returns.mean() * 252
        annual_std = returns.std() * (252 ** 0.5)

        return annual_return, annual_std

    def optimize(self):
        logger.info("Running portfolio optimization...")

        df = self.df.copy()

        mu = mean_historical_return(df)
        S = CovarianceShrinkage(df).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.add_constraint(lambda w: w <= 0.2)
        weights = ef.max_sharpe(risk_free_rate=self.rfr)
        cleaned_weights = ef.clean_weights()

        logger.info(f"Optimized Weights: {cleaned_weights}")

        perf = ef.portfolio_performance(verbose=True, risk_free_rate=self.rfr)

        return weights, cleaned_weights, perf

    def allocate(self, weights, total_value):
        latest_prices = get_latest_prices(self.df).dropna()

        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_value)
        allocation, leftover = da.greedy_portfolio()

        logger.info(f"Allocation: {allocation}")
        logger.info(f"Funds remaining: {leftover}")

        return allocation, leftover

    def capm_analysis(self, market_col: str):
        logger.info("Running CAPM analysis...")

        mu = capm_return(
            self.df,
            market_prices=pd.DataFrame(self.df[market_col]),
            risk_free_rate=self.rfr
        )

        plot_barh(mu, "Return", "CAPM Expected Returns")

    def plot_covariance(self):
        logger.info("Plotting covariance matrix...")
        exp_cov(self.df, plot_correlation=True)


# ==========================
# Main Pipeline
# ==========================
def main():
    load_dotenv()

    start_date = "2022-04-01"
    end_date = (pd.to_datetime("today") - timedelta(days=1)).date().isoformat()
    rfr = 0.071
    portfolio_value = 500000

    cfg = ModelConfig(
        stock_name= "STOCK_PORTFOLIO",
        index_name = ["^NSEI"],
        model_name= None,
        train_start_date=start_date,
        train_end_date=end_date,
        timesteps=30,
        num_features=1,
        pct_train=0.7,
        root_path=os.environ.get("ROOT_PATH")
    )

    tickers = list(yf_tickers.keys())
    ticker_names = list(yf_tickers.values())
    logger.info("Fetching portfolio data...")
    df = DataProvider.get_close_price_data(tickers, ticker_names, cfg.train_start_date, cfg.train_end_date, cfg.cache_data, cfg.stock_data_path)
    df.columns = ticker_names

    # ==========================
    # Portfolio Optimization
    # ==========================
    optimizer = PortfolioOptimizer(df, rfr)

    annual_return, annual_std = optimizer.compute_annual_metrics()

    plot_barh(annual_return, "Return", "Annual Returns")
    plot_barh(annual_std, "Risk", "Annual Volatility")

    weights, cleaned_weights, _ = optimizer.optimize()

    optimizer.allocate(weights, portfolio_value)

    optimizer.capm_analysis(market_col=ticker_names[0])  # e.g., Nifty

    optimizer.plot_covariance()


if __name__ == "__main__":
    main()