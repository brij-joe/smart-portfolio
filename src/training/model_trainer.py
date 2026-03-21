import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, X, y):
        logger.info("Training started")

        self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=2,
            shuffle=False
        )

        logger.info("Training completed")

    def predict(self, X):
        return self.model.predict(X, batch_size=self.config.batch_size)

    @staticmethod
    def inverse_scale(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    def plot(self, dates, actual, predicted, stock_name):
        plt.figure(figsize=(10, 5))
        plt.plot(dates, actual, label="Actual")
        plt.plot(dates, predicted, label="Predicted")
        plt.title(f"{stock_name} Prediction")
        plt.legend()
        plt.show()

    def forecast_future(self, last_sequence, n_steps, scaler, min_val, max_val):
        """
        last_sequence: shape (look_back, features)
        n_steps: number of future days to predict
        """
        predictions = []
        current_input = last_sequence.copy()

        for _ in range(n_steps):
            # reshape for model
            x_input = current_input.reshape(1, current_input.shape[0], current_input.shape[1])

            # predict next value
            yhat = self.model.predict(x_input, verbose = 0)

            # store prediction
            predictions.append(yhat[0][0])

            # create next input row
            next_row = current_input[-1].copy()

            # update ONLY target column (stock price)
            TARGET_COL = 0
            next_row[TARGET_COL] = yhat[0][0]

            # shift window
            current_input = np.vstack((current_input[1:], next_row))

        # inverse scaling
        predictions = np.array(predictions)
        predictions = self.inverse_scale(predictions, min_val, max_val)
        return predictions