import logging
import os

import pandas as pd
from dotenv import load_dotenv
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
    load_dotenv(verbose=True)
    cfg = ModelConfig(
        stock_name="HDFCBANK.NS",
        index_name =["^NSEI"],
        start = "2020-01-01",
        end = "2026-03-18",
        model_name="lstm_model",
        timesteps=30,
        num_features=2,
        pct_train=0.7,
        storage_root_path =os.environ.get("ROOT_PATH")
    )

    model = load_cached_model(cfg.model_path)

    # Data loading
    stock = DataProvider.get_stock_data(
        cfg.stock_name,
        cfg.start,
        cfg.end,
        cfg.cache_data,
        cfg.stock_data_path
    )

    index = DataProvider.get_index_data(
        cfg.index_name,
        cfg.start,
        cfg.end,
        cfg.cache_data,
        cfg.index_data_path
    )

    # Processing
    processor = DataProcessor()

    df = processor.merge(stock, index)
    split = int(len(df) * cfg.pct_train)
    train_df, test_df = df[:split], df[split:]

    train_scaled, test_scaled = processor.scale(train_df, test_df)
    trainX, trainY = processor.create_sequences(train_scaled, cfg.timesteps)
    testX, testY = processor.create_sequences(test_scaled, cfg.timesteps)
    print(trainX.shape, trainY.shape)

    # Take last available sequence from test set
    last_sequence = testX[-1]
    last_date = df.index[-1]
    future_days = 30
    trainer = ModelTrainer(model, cfg)

    future_preds = trainer.forecast_future(
        last_sequence,
        n_steps = future_days,
    )

    future_preds = processor.inverse_transform(future_preds)

    print("Future Predictions:", future_preds)
    plot_future(df, future_preds, last_date, future_days)

if __name__ == "__main__":
    main()