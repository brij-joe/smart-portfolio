import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:

    @staticmethod
    def get_stock_data(stock_name, start, end, cached, path):
        if cached and os.path.exists(path):
            logger.info("Loading cached stock data")
            return pd.read_pickle(path)

        logger.info("Downloading stock data")
        df = yf.download(stock_name, start=start, end=end)
        df.to_pickle(path)
        return df

    @staticmethod
    def get_index_data(tickers, start, end, cached, path):
        if cached and os.path.exists(path):
            logger.info("Loading cached index data")
            return pd.read_pickle(path)

        logger.info("Downloading index data")
        df = yf.download(tickers, start=start, end=end)
        df.to_pickle(path)
        return df