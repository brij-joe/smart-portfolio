import logging
from pathlib import Path

from keras.models import Sequential, load_model
from keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam

from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class CNNLSTMModel:

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = None

    def build(self):
        model = Sequential()

        # --- CNN Feature Extraction ---
        model.add(
            Conv1D(
                filters=self.cfg.conv_filters,
                kernel_size=self.cfg.conv_kernel_size,
                activation=self.cfg.conv_activation,
                padding="same",
                input_shape=(self.cfg.timesteps, self.cfg.num_features),
            )
        )

        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(
            Conv1D(
                filters=self.cfg.conv_filters * 2,
                kernel_size=self.cfg.conv_kernel_size,
                activation=self.cfg.conv_activation,
                padding="same",
            )
        )

        model.add(MaxPooling1D(pool_size=2))

        # --- LSTM Layers ---
        model.add(LSTM(self.cfg.units, return_sequences=True))
        model.add(Dropout(self.cfg.dropout))

        model.add(LSTM(self.cfg.units))
        model.add(Dropout(self.cfg.dropout))

        # --- Output ---
        model.add(Dense(1))

        model.compile(
            optimizer=Adam(learning_rate=self.cfg.learning_rate),
            loss="mse"
        )

        logger.info("CNN-LSTM model built")
        self.model = model
        return model

    def load_or_build(self):
        path = Path(self.cfg.model_path)

        if path.exists() and self.cfg.cache_model:
            logger.info("Loading cached CNN-LSTM model")
            self.model = load_model(path)
        else:
            self.model = self.build()

        return self.model

    def save(self):
        if self.model:
            self.model.save(self.cfg.model_path)
            logger.info(f"Model saved at {self.cfg.model_path}")