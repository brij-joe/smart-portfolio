import logging
import os
from datetime import timedelta
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pypfopt import EfficientFrontier, objective_functions
from pypfopt.expected_returns import mean_historical_return, capm_return
from pypfopt.risk_models import CovarianceShrinkage, exp_cov
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from config.model_config import ModelConfig
from data.data_provider import DataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================
# Plot Utilities
# ==========================
def plot_barh(series, title, xlabel):
    plt.figure(figsize=(10, 6))
    series.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


# ==========================
# Portfolio Optimizer
# ==========================
class PortfolioOptimizer:

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

        plot_barh(mu, "CAPM Expected Returns", "Return")

    def plot_covariance(self):
        logger.info("Plotting covariance matrix...")
        exp_cov(self.df, plot_correlation=True)


# ==========================
# Main
# ==========================
def main():
    load_dotenv()

    tickers = [
        '0P0000PTGR.BO', '0P00005VCH.BO', '0P00005WL6.BO', '0P0000KPO1.BO',
        '0P0000GB48.BO', '0P00005WLX.BO', '0P00005X22.BO', '0P00005UP8.BO',
        '0P00005VC9.BO', '0P00009JAQ.BO', '0P00005WNE.BO', '^BSESN'
    ]

    names = [
        'Nippon_Small_Cap_G', 'Quant_Mid_Cap_G', 'UTI_Nifty50_Index_G',
        'ICICI_Pru_Corp_Bond_G', 'ICICI_Pru_Bluechip_Growth',
        'HDFC_Focused30_Growth', 'ABSL_Dynamic_Bond_Fund',
        'Nippon_Mid_Cap_G', 'Quant_Small_Cap_G',
        'Nippon_Large_Cap_G', 'HDFC_Top100_G', 'BSE_Sensex'
    ]

    start_date = "2022-04-01"
    end_date = (pd.to_datetime("today") - timedelta(days=1)).date().isoformat()
    rfr = 0.071
    portfolio_value = 40_000_000

    cfg = ModelConfig(
        stock_name = "STOCK_PORTFOLIO",
        index_name = ["^NSEI"],
        model_name = None,
        train_start_date = start_date,
        train_end_date = end_date,
        timesteps = 30,
        num_features = 1,
        pct_train = 0.7,
        root_path = os.environ.get("ROOT_PATH")
    )

    # ==========================
    # Load Data
    # ==========================
    df = DataProvider.fetch_mf_nav(tickers, names, cfg.train_start_date, cfg.train_end_date, cfg.cache_data, cfg.stock_data_path)

    # Remove benchmark
    df_assets = df.drop("BSE_Sensex", axis=1)

    # ==========================
    # Sector Constraints
    # ==========================
    sector_mapper = {
        'Nippon_Small_Cap_G': 'small_cap',
        'Quant_Mid_Cap_G': 'mid_cap',
        'UTI_Nifty50_Index_G': 'large_cap',
        'ICICI_Pru_Corp_Bond_G': 'debt',
        'ICICI_Pru_Bluechip_Growth': 'large_cap',
        'HDFC_Focused30_Growth': 'large_cap',
        'ABSL_Dynamic_Bond_Fund': 'debt',
        'Nippon_Mid_Cap_G': 'mid_cap',
        'Quant_Small_Cap_G': 'small_cap',
        'Nippon_Large_Cap_G': 'large_cap',
        'HDFC_Top100_G': 'large_cap',
    }

    sector_lower = {}  # optional
    sector_upper = {
        "large_cap": 0.2,
        "mid_cap": 0.3,
        "small_cap": 0.4,
        "debt": 0.1,
    }

    # ==========================
    # Optimize Portfolio
    # ==========================
    optimizer = PortfolioOptimizer(df, rfr)

    annual_return, annual_std = optimizer.compute_log_returns()

    plot_barh(annual_return, "Annual Returns", "Return")
    plot_barh(annual_std, "Annual Volatility", "Risk")

    weights, cleaned_weights = optimizer.optimize_with_constraints(
        sector_mapper, sector_lower, sector_upper
    )

    optimizer.allocate(weights, portfolio_value)

    # ==========================
    # Advanced Analytics
    # ==========================
    optimizer.capm_analysis("BSE_Sensex")
    optimizer.plot_covariance()


if __name__ == "__main__":
    main()