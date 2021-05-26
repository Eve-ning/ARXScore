import os
from collections import Callable

from dataclasses import dataclass, field
from os import walk
from typing import List

from matplotlib import pyplot as plt
from reamber.osu import OsuMap
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from consts import CONSTS
from osrparse.mania import ManiaHitError

import numpy as np


@dataclass
class XGBoostModel:
    regressor: XGBRegressor
    key: int
    regressor_mul: MultiOutputRegressor = field(init=False)

    def __post_init__(self):
        self.regressor_mul = MultiOutputRegressor(self.regressor)

    @property
    def input_size(self): return CONSTS.INPUT_SIZE(self.key)
    @property
    def output_size(self): return CONSTS.OUTPUT_SIZE(self.key)

    def train_model(self, data:List[np.ndarray]):
        """ Trains the model using the data generated. """
        if not isinstance(data, list):
            data = [data]
        regressor_mul = MultiOutputRegressor(self.regressor)
        for ar in tqdm(data, desc="Training ... "):
            regressor_mul.fit(
                X=ar[..., :self.input_size],
                y=ar[..., self.input_size:])

    def evaluate_model(self, data:List[np.ndarray], kfolds:int = 5):
        data = np.hstack(data)
        x, y = data[:, :self.input_size], data[:, self.input_size:]
        for train_ix, test_ix in KFold(n_splits=kfolds).split(data):
            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            self.regressor_mul.fit(x_train, y_train)
            y_pred = self.regressor_mul.predict(x_test)
            print("MSE: ", np.mean((y_pred - y_test) ** 2))

    def predict(self, data:np.ndarray):
        return self.regressor.predict(data[..., :self.input_size])

    def evaluate(self, data:np.ndarray):
        return np.mean((self.predict(data) - data[..., self.input_size]) ** 2)

    def predict_agg_and_plot(self, data:np.ndarray, map_path:str):

        m = ManiaHitError.parse_map(OsuMap.readFile(map_path))
        pred = self.predict(data[..., :self.input_size])

        # Flattened Map
        a = [i for k in m[:1] for j in k for i in j]

        # Predicted
        plt.subplot(3,1,1)
        plt.plot(np.mean(pred.squeeze(), axis=1))

        # Expected
        plt.subplot(3,1,2)
        plt.plot(np.mean(data.squeeze()[:,self.input_size:], axis=1))

        # Density
        plt.subplot(3,1,3)
        plt.hist(a, bins=300)
        plt.show()

    @staticmethod
    def model_path(key):
        return CONSTS.MODEL_PATH + f'/estimator{key}'
