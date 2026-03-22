import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:

    @staticmethod
    def _load_or_download(symbols, start, end, path: Path, cached: bool):
        if cached and path.exists():
            logger.info(f"Loading cached data: {path}")
            return pd.read_pickle(path)

        logger.info(f"Downloading data: {symbols}")
        df = yf.download(symbols, start=start, end=end)
        df.to_pickle(path)
        return df

    @staticmethod
    def get_stock_data(stock_name, start, end, cached, path):
        return DataProvider._load_or_download(stock_name, start, end, path, cached)

    @staticmethod
    def get_index_data(indices, start, end, cached, path):
        return DataProvider._load_or_download(indices, start, end, path, cached)