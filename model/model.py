import os
from collections import Callable

import matplotlib.ticker as plticker
from dataclasses import dataclass, field
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
class Model:
    regressor: keras.Model
    key: int
    threshold: int = CONSTS.THRESHOLD
    window:int = CONSTS.WINDOW
    epochs:int = 50
    batch_size:int = 16
    verbose: int = 1
    aggregated: bool = False
    aggregation_method: Callable = np.median

    @staticmethod
    def load_model(key: int, aggregated: bool = False, aggregation_method=CONSTS.AGG_METHOD):
        return Model(keras.models.load_model(Model.model_path(key, aggregated)), key,
                     aggregated=aggregated, aggregation_method=aggregation_method)

    def save_model(self):
        self.regressor.save(Model.model_path(self.key, self.aggregated))

    @staticmethod
    def create_model(key, aggregated: bool = False, aggregation_method=CONSTS.AGG_METHOD):
        """ Creates a model by its key and aggregation. """
        regressor = Sequential()
        regressor.add(LSTM(units=200,
                           return_sequences=True,
                           input_shape=(None, Model._features(key))))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=key * 3))
        regressor.add(Dropout(0.1))

        if aggregated:
            regressor.add(Dense(units=1))
        else:
            regressor.add(Dense(units=key * 2))

        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return Model(regressor, key, aggregated=aggregated, aggregation_method=aggregation_method)

    @staticmethod
    def _features(key):
        return (key * 2) ** 2 * 2

    @property
    def features(self):
        return self._features(self.key)

    def train_from_rsc(self, retrain=False, lim=None):
        _, dirs, filenames = next(walk(f'{CONSTS.RSC_PATH}'))
        trainables = []
        for map_dir in np.asarray(dirs)[np.random.choice(len(dirs), lim, replace=False)]:
            if self.aggregated:
                data_dir = f"{CONSTS.RSC_PATH}/{map_dir}/{CONSTS.DATA_NAME}{CONSTS.AGG}"
                data_path = f"{data_dir}/{CONSTS.DATA_NAME}{CONSTS.AGG}.npy"
            else:
                data_dir = f"{CONSTS.RSC_PATH}/{map_dir}/{CONSTS.DATA_NAME}"
                data_path = f"{data_dir}/{CONSTS.DATA_NAME}.npy"

            if not os.path.exists(data_path):
                # This means it's new data
                # There may be cases where the dir is generated but there's no data
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                print(f"Generating Data and Training on {map_dir}")
                data = Input(self.threshold, self.window,
                             aggregated=self.aggregated,
                             aggregation_method=self.aggregation_method).load_from(map_dir)
                np.save(data_path, data)
                trainables.append(data)
            elif retrain:
                # This means it's processed data but we retrain
                data = np.load(f"{data_path}")
                print(f"Training on {map_dir}")
                trainables.append(data)

        if len(trainables) == 0:
            print("No new Data to train on.")
            return

        self.train(trainables)

    def train(self, data:List[np.ndarray]):
        """ Trains the model using the data generated. """
        if not isinstance(data, list):
            data = [data]

        for ar in tqdm(data, desc="Training ... "):
            if self.aggregated:
                assert ar.shape[-1] == self.features + 1,\
                    f"Aggregated method is invalid. {self.aggregation_method}," \
                    f"{ar.shape[-1]} mismatched {self.features + 1}"

            self.regressor.fit(x=ar[..., :self.features],y=ar[..., self.features:],
                               epochs=self.epochs, batch_size=self.batch_size,
                               verbose=self.verbose,
                               metrics=[keras.metrics.Accuracy()])

    def evaluate(self, data:np.ndarray):
        return self.regressor.evaluate(data[..., :self.features],
                                       data[..., self.features:])

    def predict(self, data:np.ndarray):
        return self.regressor.predict(data[..., :self.features])

    def predict_agg_and_plot(self, data:np.ndarray, map_path:str):

        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.predict(data[..., :self.features])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(3,1,1)
        if self.aggregated:
            plt.plot(pred.squeeze())
        else:
            plt.plot(self.aggregation_method(pred.squeeze(), axis=1))

        # Expected
        plt.subplot(3,1,2)
        plt.plot(self.aggregation_method(data.squeeze()[:,self.features:], axis=1))

        # Density
        plt.subplot(3,1,3)
        plt.hist(a, bins=100)
        plt.show()

    def predict_and_plot(self, data:np.ndarray, map_path:str):
        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.predict(data[..., :self.features])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(3,1,1)
        plt.plot(pred.squeeze(), c='r',alpha=0.3)

        # Expected
        plt.subplot(3,1,2)
        plt.plot(data.squeeze()[:,self.features:], c='blue', alpha=0.3)

        # Density
        plt.subplot(3,1,3)
        plt.hist(a, bins=100)

        plt.show()

    @staticmethod
    def model_path(key, aggregated):
        return CONSTS.MODEL_PATH + f'/estimator{key}{CONSTS.AGG}' if aggregated else \
            CONSTS.MODEL_PATH + f'/estimator{key}'
