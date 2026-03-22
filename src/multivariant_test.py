import os
import logging
from dotenv import load_dotenv

from config.model_config import ModelConfig
from data.data_provider import DataProvider
from data.data_processor import DataProcessor
from models.cnn_lstm_model import CNNLSTMModel
from training.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    cfg = ModelConfig(
        stock_name="HDFCBANK.NS",
        index_list=['GC=F', 'CL=F', '^BSESN', '^GSPC', '^TNX', '^FTSE'],
        train_start_date="2020-01-01",
        train_end_date="2026-03-18",
        model_name = "cnn_lstm_model",
        timesteps=30,
        num_features=7,
        pct_train=0.7,
        root_path=os.environ.get("ROOT_PATH")
    )

    # --- Fetch Data ---
    stock = DataProvider.get_stock_data(
        cfg.stock_name,
        cfg.train_start_date,
        cfg.train_end_date,
        cfg.cache_data,
        cfg.stock_data_path
    )

    index = DataProvider.get_index_data(
        cfg.index_list,
        cfg.train_start_date,
        cfg.train_end_date,
        cfg.cache_data,
        cfg.index_data_path
    )

    # --- Process Data ---
    processor = DataProcessor()

    df = processor.merge(stock, index)
    print(f" df.shape: {df.shape}")
    split_idx = int(len(df) * cfg.pct_train)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    train_scaled, test_scaled = processor.scale(train_df, test_df)

    X_train, y_train = processor.create_sequences(train_scaled, cfg.timesteps)
    X_test, y_test = processor.create_sequences(test_scaled, cfg.timesteps)

    logger.info(f"Train shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test shape: {X_test.shape}, {y_test.shape}")

    # --- Model ---
    model_wrapper = CNNLSTMModel(cfg)
    model = model_wrapper.load_or_build()

    trainer = ModelTrainer(model, cfg)

    # --- Train ---
    # history = trainer.train(X_train, y_train, X_test, y_test)
    history = trainer.train_if_needed(X_train, y_train, X_test, y_test)
    if history:
        model_wrapper.save()
        trainer.plot_loss(history)


    # --- Predict ---
    predictions = trainer.predict(X_test).flatten()

    # --- Inverse Scaling ---
    predictions = processor.inverse_transform(predictions)
    actual = processor.inverse_transform(y_test)

    # --- Plot ---
    dates = df.index[-len(actual):]
    trainer.plot_predictions(dates, actual, predictions)


if __name__ == "__main__":
    main()