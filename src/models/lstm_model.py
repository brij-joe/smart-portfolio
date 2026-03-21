import logging
import os
from keras.layers import Dense, LSTM
from keras.models import load_model, Sequential
from config.model_config import ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel:

    def __init__(self, config):
        self.config = config
        self.model = None

    def build(self):
        model = Sequential()
        model.add(LSTM(self.config.units, return_sequences = True, input_shape = (self.config.look_back, self.config.features)))
        model.add(LSTM(self.config.units))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        self.model = model
        logger.info("Model built successfully")
        return model

    def load_or_build(self):
        if os.path.exists(self.config.model_path) and self.config.cache_model:
            logger.info("Loading cached model")
            self.model = load_model(self.config.model_path)
        else:
            self.build()

        return self.model

    def save(self):
        if self.model:
            self.model.save(self.config.model_path)
