import os
import logging
from dotenv import load_dotenv

from config.model_config import ModelConfig
from data.data_provider import DataProvider
from data.data_processor import DataProcessor
from models.lstm_model import LSTMModel
from training.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)


def main():
    load_dotenv()

    cfg = ModelConfig(
        stock_name="HDFCBANK.NS",
        index_list=["^NSEI"],
        train_start_date="2020-01-01",
        train_end_date="2026-03-18",
        model_name="lstm_model",
        timesteps=30,
        num_features=2,
        pct_train=0.7,
        root_path=os.environ.get("ROOT_PATH")
    )

    # Fetch data
    stock = DataProvider.get_stock_data(
        cfg.stock_name, cfg.train_start_date, cfg.train_end_date,
        cfg.cache_data, cfg.stock_data_path
    )

    index = DataProvider.get_index_data(
        cfg.index_list, cfg.train_start_date, cfg.train_end_date,
        cfg.cache_data, cfg.index_data_path
    )

    processor = DataProcessor()

    df = processor.merge(stock, index)

    split = int(len(df) * cfg.pct_train)
    train_df, test_df = df[:split], df[split:]

    train_scaled, test_scaled = processor.scale(train_df, test_df)

    X_train, y_train = processor.create_sequences(train_scaled, cfg.timesteps)
    X_test, y_test = processor.create_sequences(test_scaled, cfg.timesteps)

    model_wrapper = LSTMModel(cfg)
    model = model_wrapper.load_or_build()

    trainer = ModelTrainer(model, cfg)
    # --- Train ---
    # history = trainer.train(X_train, y_train, X_test, y_test)
    history = trainer.train_if_needed(X_train, y_train, X_test, y_test)
    if history:
        model_wrapper.save()
        trainer.plot_loss(history)

    preds = trainer.predict(X_test)
    preds = processor.inverse_transform(preds.flatten())
    actual = processor.inverse_transform(y_test)

    trainer.plot_predictions(df.index[-len(actual):], actual, preds)


if __name__ == "__main__":
    main()