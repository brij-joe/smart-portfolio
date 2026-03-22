import logging

import pandas as pd
from pypfopt import CovarianceShrinkage, DiscreteAllocation, EfficientFrontier, get_latest_prices, objective_functions
from pypfopt.expected_returns import capm_return, mean_historical_return
from pypfopt.risk_models import exp_cov

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Portfolio Utilities
# ==========================
class StockPortfolioOptimizer:

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

        return mu

    def plot_covariance(self):
        logger.info("Plotting covariance matrix...")

        exp_cov(self.df, plot_correlation=True)


