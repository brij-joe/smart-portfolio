import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def train(self, X_train, y_train, X_val, y_val):
        logger.info("Training started")

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            shuffle=False
        )

        return history

    def train_if_needed(self, X_train, y_train, X_val, y_val):
        model_path = Path(self.cfg.model_path)
        if model_path.exists() and self.cfg.cache_model:
            logger.info("Model already exists. Skipping training.")
            return None

        logger.info("Training model...")
        history = self.model.fit(
            X_train,
            y_train,
            validation_data = (X_val, y_val),
            epochs = self.cfg.epochs,
            batch_size = self.cfg.batch_size,
            shuffle = False
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def plot_loss(history):
        if not history:
            return

        plt.figure()
        plt.plot(history.history.get('loss', []), label="train")

        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label="val")

        plt.legend()
        plt.title("Loss")
        plt.show()

    @staticmethod
    def plot_predictions(dates, actual, predicted):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, label="Actual")
        plt.plot(dates, predicted, label="Predicted")
        plt.legend()
        plt.show()