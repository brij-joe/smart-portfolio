import logging
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from config.model_config import ModelConfig
from data.data_provider import DataProvider
from models.stock_portfolio_optimizer import StockPortfolioOptimizer

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


# ==========================
# Plotting Utilities
# ==========================
def plot_barh(series, xlabel, title):
    plt.figure(figsize = (10, 6))
    series.sort_values().plot(kind = "barh")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==========================
# Main Pipeline
# ==========================
def main():
    load_dotenv()

    start_date = "2022-04-01"
    end_date = (pd.to_datetime("today") - timedelta(days = 1)).date().isoformat()
    end_date = '2025-12-31'
    rfr = 0.071
    portfolio_value = 500_000

    cfg = ModelConfig(
        stock_name = "STOCK_PORTFOLIO",
        index_name = ["^NSEI"],
        model_name = None,
        train_start_date = start_date,
        train_end_date = end_date,
        timesteps = 30,
        num_features = 1,
        pct_train = 0.7,
        root_path = os.environ.get("ROOT_PATH"),
    )

    tickers = [
        'HIKAL.NS', 'BALRAMCHIN.NS', 'ADANIPORTS.NS', 'ADANIGREEN.NS', 'RELIANCE.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'HINDUNILVR.NS', 'SBIN.NS', 'INFY', 'TCS.NS', 'TMCV.NS', 'TMPV.NS', 'IRFC.NS', 'WEBELSOLAR.NS', 'EASEMYTRIP.NS',
        'ASIANPAINT.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'ICICIBANK.NS', 'TITAN.NS', 'DIVISLAB.NS', 'BERGEPAINT.NS',
        'IDFCFIRSTB.NS', 'HDFCBANK.NS', 'ITC.NS', '^NSEI'
    ]
    print(tickers)

    ticker_names = [
        'Hikal Ltd', 'Balrampur Chini Mills Ltd', 'Adani Ports & Special Economic Zone Ltd', 'Adani Green Energy Ltd',
        'Reliance Industries Ltd', 'Kotak Mahindra Bank Ltd', 'Axis Bank Ltd', 'Hindustan Unilever Ltd',
        'State Bank of India', 'Infosys Ltd', 'Tata Consultancy Services Ltd', 'Tata Motors Commercial Vehicles',
        'Tata Motors Passenger Vehicles', 'Indian Railway Finance Corporation Ltd', 'Websol Energy System Ltd',
        'Easy Trip Planners Ltd', 'Asian Paints Ltd', 'Bajaj Finance Ltd', 'Bajaj Finserv Ltd', 'ICICI Bank Ltd',
        'Titan Company Ltd', "Divi's Laboratories Ltd", 'Berger Paints India Ltd', 'IDFC First Bank Ltd',
        'HDFC Bank Ltd', 'ITC Ltd', 'Nifty 50'
    ]
    print(ticker_names)

    logger.info("Fetching portfolio data...")
    df = DataProvider.get_close_price_data(
        tickers, ticker_names,
        cfg.train_start_date,
        cfg.train_end_date,
        cfg.cache_data,
        cfg.stock_data_path,
    )
    df.columns = ticker_names

    # ==========================
    # Portfolio Optimization
    # ==========================
    optimizer = StockPortfolioOptimizer(df, rfr)

    annual_return, annual_std = optimizer.compute_annual_metrics()

    plot_barh(annual_return, "Return", "Annual Returns")
    plot_barh(annual_std, "Risk", "Annual Volatility")

    weights, cleaned_weights, _ = optimizer.optimize()

    optimizer.allocate(weights, portfolio_value)

    mu = optimizer.capm_analysis(market_col = ticker_names[0])  # e.g., Nifty
    plot_barh(mu, "Return", "CAPM Expected Returns")

    optimizer.plot_covariance()


if __name__ == "__main__":
    main()