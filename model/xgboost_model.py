import pandas as pd
from joblib import dump, load

from dataclasses import dataclass, field
from typing import List, Union

from matplotlib import pyplot as plt
from reamber.osu import OsuMap
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from consts import CONSTS
from osrparse.mania import ManiaHitError

import numpy as np


class XGBoostModel:
    regressor: MultiOutputRegressor
    key: int

    def __init__(self,
                 key:int,
                 regressor:Union[XGBRegressor, MultiOutputRegressor]):
        self.regressor = MultiOutputRegressor(regressor) if isinstance(regressor, XGBRegressor) else regressor
        self.key = key

    def __post_init__(self):
        if isinstance(self.regressor, XGBRegressor):
            # Cast Regressor as Multi if not already
            self.regressor = MultiOutputRegressor(self.regressor)
        else:
            self.regressor = self.regressor

    @property
    def input_size(self): return CONSTS.INPUT_SIZE(self.key)
    @property
    def output_size(self): return CONSTS.OUTPUT_SIZE(self.key)

    def train_model(self, data:List[np.ndarray]):
        """ Trains the model using the data generated. """
        if not isinstance(data, list):
            data = [data]
        for ar in tqdm(data, desc="Training ... "):
            self.regressor.fit(
                X=ar[..., :self.input_size],
                y=ar[..., self.input_size:])


    def evaluate_model(self, data:List[np.ndarray], kfolds:int = 5):
        data = np.vstack(data)
        x, y = self.input(data), self.output(data)
        for train_ix, test_ix in KFold(n_splits=kfolds).split(data):
            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            self.regressor.fit(x_train, y_train, verbose=True)
            y_pred = self.regressor.predict(x_test)
            print("MSE: ", np.mean((y_pred - y_test) ** 2))

    def predict(self, data:np.ndarray):
        return self.regressor.predict(self.input(data))

    def evaluate(self, data:np.ndarray):
        return np.mean((self.predict(data) - self.input(data)) ** 2)

    def input(self, data:np.ndarray):
        return data[:, :self.input_size]

    def output(self, data:np.ndarray):
        return data[:, self.input_size:]

    def save(self, ver="0"):
        dump(self.regressor, f'models/xgboost/xgboost{self.key}k_{ver}.joblib')

    @staticmethod
    def load(key, ver="0"):
        regressor = load(f'models/xgboost/xgboost{key}k_{ver}.joblib')
        return XGBoostModel(key, regressor)

    def density(self, map_path:str):
        er = ManiaHitError.parse_map(OsuMap.readFile(
             f"{CONSTS.RSC_PATH}/{self.key}/{map_path}/{map_path}.osu"))
        offsets = [*[i for j in er[0] for i in j], *[i for j in er[1] for i in j]]
        df = pd.DataFrame(np.ones_like(offsets), index=offsets)
        df.index = pd.to_datetime(df.index, unit='ms')
        return df.groupby(pd.Grouper(freq=f'{1000}ms')).sum().to_numpy().squeeze()

    def predict_and_plot(self, data:np.ndarray, map_path:str):
        ax = plt.subplot(3, 1, 1)
        plt.plot(self.predict(data), c='red', alpha=0.5)
        ax.set_ylabel("Predicted")
        ax = plt.subplot(3, 1, 2, sharey=ax, sharex=ax)
        plt.plot(self.output(data), c='blue', alpha=0.5)
        ax.set_ylabel("Actual")
        ax = plt.subplot(3, 1, 3, sharex=ax)
        plt.plot(self.density(map_path), c='black')
        ax.set_ylabel("Density")

    def predict_and_plot_agg(self, data:np.ndarray, map_path:str):
        ax = plt.subplot(3, 1, 1)
        plt.plot(np.mean(self.predict(data),axis=-1), c='red')
        ax.set_ylabel("Predicted")
        ax = plt.subplot(3, 1, 2, sharey=ax, sharex=ax)
        plt.plot(np.mean(self.output(data),axis=-1), c='blue')
        ax.set_ylabel("Actual")
        ax = plt.subplot(3, 1, 3, sharex=ax)
        plt.plot(self.density(map_path), c='black')
        ax.set_ylabel("Density")

