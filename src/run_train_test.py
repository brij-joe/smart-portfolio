import logging

from config.model_config import ModelConfig
from data.data_provider import DataProvider
from data.data_processor import DataProcessor
from training.model_trainer import ModelTrainer
from models.lstm_model import LSTMModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = ModelConfig()
    logger.info(config)

    # Data loading
    stock = DataProvider.get_stock_data(
        config.stock_name,
        config.train_start_date,
        config.train_end_date,
        config.cache_data,
        config.stock_data_path
    )

    index = DataProvider.get_index_data(
        config.index_list,
        config.train_start_date,
        config.train_end_date,
        config.cache_data,
        config.index_data_path
    )

    # Processing
    df = DataProcessor.merge_data(stock, index)

    train_df, test_df = DataProcessor.split_data(df, config.pct_train)

    train, test, min_val, max_val = DataProcessor.normalize(
        train_df.values, test_df.values
    )

    trainX, trainY = DataProcessor.create_dataset(
        train, config.look_back, config.features
    )

    testX, testY = DataProcessor.create_dataset(
        test, config.look_back, config.features
    )

    # Model
    model_wrapper = LSTMModel(config)
    model = model_wrapper.load_or_build()

    trainer = ModelTrainer(model, config)

    # Train
    trainer.train(trainX, trainY)
    model_wrapper.save()

    # Predict
    train_pred = trainer.predict(trainX)
    test_pred = trainer.predict(testX)

    # Inverse scaling
    test_pred = trainer.inverse_scale(test_pred, min_val, max_val)
    testY = trainer.inverse_scale(testY, min_val, max_val)

    # Plot
    dates = test_df[-len(testY):].index
    trainer.plot(dates, testY, test_pred, config.stock_name)


if __name__ == "__main__":
    main()