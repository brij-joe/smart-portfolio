import logging
import os

import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model, Sequential

from config.model_config import ModelConfig
from data.data_processor import DataProcessor
from data.data_provider import DataProvider
from training.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cached_model(model_path) -> Sequential:
    if os.path.exists(model_path):
        logger.info("Loading cached model")
        return load_model(model_path)
    else:
        raise ValueError(f"Model not found at {model_path}")



def plot_future(df, future_preds, last_date, future_days):
    future_dates = pd.date_range(start=last_date, periods=future_days + 1)[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df.iloc[:, 0], label="Historical")
    plt.plot(future_dates, future_preds, label="Forecast")
    plt.legend()
    plt.show()


def main():
    config = ModelConfig()
    model = load_cached_model(config.model_path)

    # Data loading
    stock = DataProvider.get_stock_data(
        config.stock_name,
        config.train_start_date,
        config.train_end_date,
        config.cache_data,
        config.stock_data_path
    )

    index = DataProvider.get_index_data(
        config.index_name,
        config.train_start_date,
        config.train_end_date,
        config.cache_data,
        config.index_data_path
    )

    # Processing
    df = DataProcessor.merge_data(stock, index)
    train_df, test_df = DataProcessor.split_data(df, config.pct_train)
    train, test, min_val, max_val = DataProcessor.normalize(train_df.values, test_df.values)
    trainX, trainY = DataProcessor.create_dataset(train, config.timesteps, config.num_feature)
    testX, testY = DataProcessor.create_dataset(test, config.timesteps, config.num_feature)

    print(trainX.shape, trainY.shape)

    # Take last available sequence from test set
    last_sequence = testX[-1]
    last_date = df.index[-1]
    future_days = 30
    trainer = ModelTrainer(model, config)
    future_preds = trainer.forecast_future(
        last_sequence,
        n_steps = future_days,
        scaler = None,
        min_val = min_val,
        max_val = max_val
    )

    print("Future Predictions:", future_preds)
    plot_future(df, future_preds, last_date, future_days)

if __name__ == "__main__":
    main()