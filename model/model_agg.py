import os
from dataclasses import dataclass
from os import walk
from typing import List

from matplotlib import pyplot as plt
from reamber.osu import OsuMap
from tqdm import tqdm

from consts import CONSTS
from input import Input
from osrparse.mania import ManiaHitError

import numpy as np
from tensorflow import keras

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense, Bidirectional


@dataclass
class ModelAggregated:
    regressor: keras.Model
    key: int
    threshold: int = 200
    window:int = 1000
    epochs:int = 50
    batch_size:int = 16
    verbose: int = 1

    @staticmethod
    def load_model(key, path="models/estimatorAgg"):
        regressor = keras.models.load_model(f"{path}{key}")
        return ModelAggregated(regressor, key)

    def save_model(self, path="models/estimatorAgg"):
        self.regressor.save(f"{path}{self.key}")

    @staticmethod
    def create_model(key):
        regressor = Sequential()
        regressor.add(LSTM(units=200,
                           return_sequences=True,
                           input_shape=(None, ModelAggregated._no_features(key))))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=key * 3))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return ModelAggregated(regressor, key)

    @staticmethod
    def _no_features(key):
        return (key * 2) ** 2 * 2

    @property
    def no_features(self):
        return self._no_features(self.key)

    def train_from_rsc(self, retrain=False):
        _, dirs, filenames = next(walk(f'{CONSTS.RSC_PATH}'))
        trainables = []
        for map_dir in dirs:
            data_dir = f"{CONSTS.RSC_PATH}/{map_dir}/{CONSTS.DATA_AGG_NAME}"
            data_path = f"{data_dir}/{CONSTS.DATA_AGG_NAME}.npy"
            if not os.path.exists(data_path):
                # This means it's new data
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                print(f"Generating Data and Training on {map_dir}")
                data = Input(self.threshold, self.window,
                             aggregated_output=True).load_from(map_dir)
                np.save(data_path, data)
                print("Successfully Generated Data.")
                trainables.append(data)
            elif retrain:
                # This means it's processed data but we retrain
                data = np.load(data_path)
                print(f"Training on {map_dir}")
                trainables.append(data)
        if len(trainables) == 0:
            print("No new Data to train on.")
            return
        self.train(trainables)

    def train(self, data:List[np.ndarray]):
        if not isinstance(data, list):
            data = [data]

        FEATURES = self.no_features
        for ar in tqdm(data, desc="Training ... "):
            self.regressor.fit(ar[..., :FEATURES],
                               ar[..., FEATURES:],
                               epochs=self.epochs, batch_size=self.batch_size,
                               verbose=self.verbose)

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

    def predict(self, data:np.ndarray):
        FEATURES = self.no_features
        return self.regressor.predict(data[..., :FEATURES])

    def predict_agg_and_plot(self, data:np.ndarray, map_path:str):
        FEATURES = self.no_features

        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.regressor.predict(data[..., :FEATURES])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(3,1,1)
        plt.plot(np.median(pred.squeeze(), axis=1))

        # Expected
        plt.subplot(3,1,2)
        plt.plot(np.median(data.squeeze()[:,FEATURES:], axis=1))

        # Density
        plt.subplot(3,1,3)
        plt.hist(a, bins=100)

        plt.show()
