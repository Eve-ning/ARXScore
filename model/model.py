from dataclasses import dataclass
from typing import List

from matplotlib import pyplot as plt
from reamber.osu import OsuMap

from osrparse.mania import ManiaHitError

import numpy as np
from tensorflow import keras

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Bidirectional


@dataclass
class Model:
    regressor: keras.Model
    key: int

    @staticmethod
    def load_model(key, path="models/estimator"):
        regressor = keras.models.load_model(f"{path}{key}")
        return Model(regressor, key)

    def save_model(self, path="models/estimator"):
        self.regressor.save(f"{path}{self.key}")

    @staticmethod
    def create_model(key):
        regressor = Sequential()
        regressor.add(Bidirectional(LSTM(units=20,
                                         return_sequences=True,
                                         input_shape=(None, Model._no_features(key)))))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=key * 2))

        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return Model(regressor, key)

    @staticmethod
    def _no_features(key):
        return (key * 2) ** 2 * 2

    @property
    def no_features(self):
        return self._no_features(self.key)

    def train(self, data:List[np.ndarray]):
        if not isinstance(data, list):
            data = [data]
        ar = np.hstack(data)

        FEATURES = self.no_features

        self.regressor.fit(ar[..., :FEATURES],
                           ar[..., FEATURES:],
                           epochs=100, batch_size=16)

    def evaluate(self, data:np.ndarray):
        FEATURES = self.no_features
        return self.regressor.evaluate(data[..., :FEATURES],
                                       data[..., FEATURES:])

    def compare_and_plot(self, data:np.ndarray, map_path:str):
        FEATURES = self.no_features

        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.regressor.predict(data[..., :FEATURES])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(2,1,1)
        plt.plot(np.mean(pred.squeeze(), axis=1))

        # Density
        plt.subplot(2,1,2)
        plt.hist(a, bins=100)

    def predict_and_plot(self, data:np.ndarray, map_path:str):
        FEATURES = self.no_features

        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.regressor.predict(data[..., :FEATURES])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(3,1,1)
        plt.plot(np.mean(pred.squeeze(), axis=1))

        # Expected
        plt.subplot(3,1,2)
        plt.plot(np.mean(data.squeeze()[:,FEATURES:], axis=1))

        # Density
        plt.subplot(3,1,3)
        plt.hist(a, bins=100)

        plt.show()
