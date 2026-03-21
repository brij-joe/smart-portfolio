import numpy as np
import pandas as pd



class DataProcessor:

    @staticmethod
    def merge_data(stock, index):
        df = pd.concat([stock['Close'], index['Close']], axis=1)
        df = df.where(lambda x: x > 0).dropna()
        df = df.interpolate(method="time")
        return df

    @staticmethod
    def split_data(df: pd.DataFrame, pct: float):
        n = int(len(df) * pct)
        return df.iloc[:n], df.iloc[n:]

    @staticmethod
    def normalize(train, test):
        min_val = train.min()
        max_val = train.max()

        train_norm = (train - min_val) / (max_val - min_val)
        test_norm = (test - min_val) / (max_val - min_val)

        return train_norm, test_norm, min_val, max_val

    @staticmethod
    def create_dataset(dataset, look_back, features):
        X, y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:i + look_back])
            y.append(dataset[i + look_back, -features])
        return np.array(X), np.array(y)