import logging
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from config.model_config import ModelConfig
from data.data_provider import DataProvider
from models.mf_portfolio_optimizer import MFPortfolioOptimizer

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
        index_name = ["^BSESN"],
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
        "large_cap": 0.4,
        "mid_cap": 0.3,
        "small_cap": 0.2,
        "debt": 0.1,
    }

    # ==========================
    # Optimize Portfolio
    # ==========================
    optimizer = MFPortfolioOptimizer(df, rfr)

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
    mu = optimizer.capm_analysis("BSE_Sensex")
    plot_barh(mu, "CAPM Expected Returns", "Return")

    optimizer.plot_covariance()


if __name__ == "__main__":
    main()