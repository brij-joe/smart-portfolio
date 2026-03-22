import logging
from pathlib import Path
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

logger = logging.getLogger(__name__)


class LSTMModel:

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def build(self):
        model = Sequential([
            LSTM(self.cfg.units, return_sequences=True,
                 input_shape=(self.cfg.timesteps, self.cfg.num_features)),
            LSTM(self.cfg.units),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.cfg.learning_rate),
            loss='mse'
        )

        logger.info("Model built")
        return model

    def load_or_build(self):
        path = Path(self.cfg.model_path)

        if path.exists() and self.cfg.cache_model:
            logger.info("Loading cached model")
            self.model = load_model(path)
        else:
            self.model = self.build()

        return self.model

    def save(self):
        if self.model:
            self.model.save(self.cfg.model_path)