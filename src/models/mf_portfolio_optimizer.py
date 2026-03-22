# ==========================
# Portfolio Optimizer
# ==========================
import logging
from typing import Dict

import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage, DiscreteAllocation, EfficientFrontier, get_latest_prices, objective_functions
from pypfopt.expected_returns import capm_return, mean_historical_return
from pypfopt.risk_models import exp_cov

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MFPortfolioOptimizer:

    def __init__(self, df_prices: pd.DataFrame, risk_free_rate: float):
        self.df = df_prices
        self.rfr = risk_free_rate

    def compute_log_returns(self):
        returns = np.log(self.df.pct_change() + 1).dropna()

        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)

        logger.info(f"Annual Returns:\n{annual_return}")
        logger.info(f"Annual Risk:\n{annual_std}")

        return annual_return, annual_std

    def optimize_with_constraints(
        self,
        sector_mapper: Dict[str, str],
        sector_lower: Dict[str, float],
        sector_upper: Dict[str, float],
    ):
        logger.info("Optimizing portfolio with sector constraints...")

        mu = mean_historical_return(self.df)
        S = CovarianceShrinkage(self.df).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)

        if sector_mapper:
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

        weights = ef.max_sharpe(risk_free_rate=self.rfr)
        cleaned_weights = ef.clean_weights()

        logger.info(f"Optimized Weights:\n{cleaned_weights}")

        ef.portfolio_performance(verbose=True, risk_free_rate=self.rfr)

        return weights, cleaned_weights

    def allocate(self, weights, total_value):
        latest_prices = get_latest_prices(self.df).dropna()

        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_value)
        allocation, leftover = da.greedy_portfolio()

        logger.info(f"Allocation:\n{allocation}")
        logger.info(f"Remaining cash: {leftover}")

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

