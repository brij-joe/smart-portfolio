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
        'ICICIBANK.BO',
        'IDFCFIRSTB.BO',
        'JIOFIN.BO',
        'PPLPHARMA.BO',
        'POLYCAB.BO',
        'SAGCEM.BO',
        'SAILIFE.BO',
        'TMCV.BO',
        'TATASTEEL.BO',
        '0P0000PTGR.BO',
        '0P00005VCH.BO',
        '0P00005WLX.BO',
        '0P0000GB48.BO',
        '0P000093TC.BO',
        '0P00005VC5.BO'
        '^BSESN'
    ]

    names = [
        'ICICI Bank',
        'IDFC First Bank',
        'Jio Finance',
        'Piramal Pharma',
        'Poly Cab',
        'Sagar Cement',
        'Sai Life',
        'Tata Motors Commercial Vehicle',
        'Tata Steel',
        'Nippon_Small_Cap_G',
        'Quant_Mid_Cap_G',
        'HDFC Focused Fund',
        'ICICI Pru Large Cap'
        'Invesco Mid Cap',
        'BSE_Sensex'
    ]

    start_date = "2022-04-01"
    end_date = (pd.to_datetime("today") - timedelta(days=1)).date().isoformat()
    rfr = 0.071
    portfolio_value = 40_000_000

    cfg = ModelConfig(
        stock_name = "STOCK_PORTFOLIO",
        index_name = ["^BSESN"],
        model_name = None,
        start = start_date,
        end = end_date,
        timesteps = 30,
        num_features = 1,
        pct_train = 0.7,
        storage_root_path = os.environ.get("ROOT_PATH")
    )

    # ==========================
    # Load Data
    # ==========================
    df = DataProvider.fetch_mf_nav(tickers, names, cfg.start, cfg.end, cfg.cache_data, cfg.stock_data_path)

    # Remove benchmark
    df_assets = df.drop("BSE_Sensex", axis=1)

    # ==========================
    # Sector Constraints
    # ==========================
    sector_mapper = {
        'ICICI Bank': 'large_cap',
        'IDFC First Bank': 'mid_cap',
        'Jio Finance': 'large_cap',
        'Piramal Pharma': 'small_cap',
        'Poly Cab': 'large_cap',
        'Sagar Cement': 'small_cap',
        'Sai Life': 'small_cap',
        'TMCV': 'large_cap',
        'Tata Steel': 'large_cap',
        'Nippon Small Cap': 'small_cap',
        'Quant Mid Cap': 'mid_cap',
        'HDFC Focused Fund': 'large_cap',
        'ICICI Pru Large Cap': 'large_cap',
        'Invesco Mid Cap': 'mid_cap',
        'BSE_Sensex': 'Index'
    }

    sector_lower = {}  # optional
    sector_upper = {
        "large_cap": 0.4,
        "mid_cap": 0.3,
        "small_cap": 0.3,
        "debt": 0,
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