import logging
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:

    @staticmethod
    def _load_or_download(symbols: str | List[str], start: str, end: str, file_path: Path, cached: bool) -> pd.DataFrame:
        if cached and file_path.exists():
            logger.info(f"Loading cached data: {file_path}")
            return pd.read_pickle(file_path)

        logger.info(f"Downloading data: {symbols}")
        df = yf.download(symbols, start=start, end=end)
        df.to_pickle(file_path)
        return df


    @staticmethod
    def get_stock_data(ticker: str, start: str, end: str, cached: bool, path:Path) -> pd.DataFrame:
        file_path = path / f"{ticker}_{start}_{end}.pkl"
        return DataProvider._load_or_download(ticker, start, end, file_path, cached)


    @staticmethod
    def get_index_data(indices: List[str], start:str, end: str, cached: bool, path: Path) -> pd.DataFrame:
        file_path = path / f"index_{len(indices)}_{start}_{end}.pkl"
        return DataProvider._load_or_download(indices, start, end, file_path, cached)


    @staticmethod
    def get_close_price_data(codes: List[str], names: List[str], start: str, end: str, cached: bool, path: Path) -> pd.DataFrame:
        df_list = []
        for ticker, ticker_name in zip(codes, names):
            file_path = path / f"{ticker}_{start}_{end}.pkl"
            df = DataProvider._load_or_download(ticker, start, end, file_path, cached)
            df.rename(columns = {ticker: ticker_name}, inplace = True)
            df.interpolate(inplace = True, method = "time")
            df.dropna(inplace = True, how = "any")
            df_list.append(df)

        # Concatenate them horizontally (column-wise)
        joined_df = pd.concat(df_list, axis = 1)
        return joined_df['Close']

    @staticmethod
    def fetch_mf_nav(codes: List[str], names: List[str], start: str, end: str, cached: bool, path: Path) -> pd.DataFrame:
        logger.info("Fetching mutual fund NAV data...")
        df_list = []
        for ticker, ticker_name in zip(codes, names):
            file_path = path / f"{ticker}_{start}_{end}.pkl"
            df = DataProvider._load_or_download(ticker, start, end, file_path, cached)
            df.rename(columns = {ticker: ticker_name}, inplace = True)
            df.interpolate(inplace = True, method = "time")
            df.dropna(inplace = True, how = "any")
            df_list.append(df)

        # Concatenate them horizontally (column-wise)
        joined_df = pd.concat(df_list, axis = 1)
        joined_df = joined_df.interpolate().dropna()
        return joined_df['Close']
