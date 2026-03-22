import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def merge(self, stock_df, index_df):
        df = pd.concat([stock_df['Close'], index_df['Close']], axis=1)
        df = df.dropna()
        return df

    def scale(self, train_df, test_df):
        self.scaler.fit(train_df)

        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)

        return train_scaled, test_scaled

    def create_sequences(self, data, timesteps):
        X, y = [], []

        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps:i])
            y.append(data[i, 0])  # target = stock price

        return np.array(X), np.array(y)

    def inverse_transform(self, values):
        dummy = np.zeros((len(values), self.scaler.n_features_in_))
        dummy[:, 0] = values
        return self.scaler.inverse_transform(dummy)[:, 0]

    @staticmethod
    def calc_returns(df_nav):
        # Compute the logarithmic returns from the prices
        df_nav.dropna(inplace = True)
        df_log_ret = np.log(df_nav.pct_change() + 1)
        df_log_ret.dropna(inplace = True)
        df_ret_mean = df_log_ret.mean(axis = 0)
        df_ret_var = df_log_ret.var(axis = 0)
        df_ret_std = df_log_ret.std(axis = 0)
        df_ret_cov = df_log_ret.cov()
        df_ret_annual = df_ret_mean * 252
        df_annual_std = df_ret_std * np.sqrt(252)
        print("===============\nAnnual Returns:\n===============\n", df_ret_annual)
        print("===============\nAnnual Std.Dev.:\n===============\n", df_annual_std)
        return df_ret_annual, df_annual_std
