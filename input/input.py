from dataclasses import dataclass
from os import walk

import numpy as np
import pandas as pd
from reamber.osu import OsuMap
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from osrparse import Replay, parse_replay_file
from osrparse.mania import ManiaHitError


@dataclass
class Input:

    threshold: int = 200
    window_ms: float = 1000

    def load_from(self, map_str):
        _, _, filenames = next(walk(f"rsc/{map_str}/rep/"))
        reps = [parse_replay_file(f"rsc/{map_str}/rep/{f}") for f in filenames]
        map = OsuMap.readFile(f"rsc/{map_str}/{map_str}.osu")
        return self.create_data(reps, map)

    def create_data(self, reps, map):
        errors = ManiaHitError(reps, map).errors()

        map = [*[np.asarray(h) for h in errors.hit_map],
               *[np.asarray(h) for h in errors.rel_map]]

        df_map_features = self._minmax(
            self._map_features(map, self.window_ms, self.threshold)
        )
        if reps:
            df_rep_error = self._rep_error(errors, self.window_ms)
            df = pd.concat([df_map_features, df_rep_error], axis=1)
        else:
            df = df_map_features

        return df.to_numpy()[np.newaxis, ...]

    @staticmethod
    def _minmax(df):
        return ((df - df.min()) / (df.max() - df.min())).fillna(0)

    @staticmethod
    def _std(df):
        return ((df - df.mean()) / df.std()).fillna(0)

    @staticmethod
    def _rep_error(errors, window_ms):
        rep_errors = []
        for rep_hit_error, rep_rel_error in zip(errors.hit_errors, errors.rel_errors):
            rep_errors.append([*[e for k in rep_hit_error for e in k],
                               *[e for k in rep_rel_error for e in k]])
        ar_error = np.asarray(rep_errors, dtype=int)

        map_hit = [e for k in errors.hit_map for e in k]
        map_rel = [e for k in errors.rel_map for e in k]
        ar_map = np.asarray([*map_hit, *map_rel])
        df_error = pd.DataFrame(np.abs(ar_error.transpose()), index=ar_map)
        df_error = df_error.sort_index()
        df_error.index = pd.to_datetime(df_error.index, unit='ms')

        df_error = df_error.groupby(pd.Grouper(freq=f'{window_ms}ms')).sum()
        return Input._std(df_error)

    @staticmethod
    def _map_features(map, window_ms, threshold):
        CANDIDATES = len(map)

        diffs = []
        for i in tqdm(range(CANDIDATES), desc="Feature Calculation Progress"):
            b = np.zeros([CANDIDATES * 2 + 1, len(map[i])])
            b[0] = map[i]
            for j in range(CANDIDATES):
                a = np.abs(map[i] - map[j][..., None])
                a = np.where(a < threshold, 1 / (1 + a), 0)
                a_sum = np.sum(a, axis=0)
                a_count = np.count_nonzero(a, axis=0)
                b[j * 2 + 1] = a_sum
                b[j * 2 + 2] = a_count
            b = pd.DataFrame(b.transpose()).set_index(0)
            b.index = pd.to_datetime(b.index, unit='ms')
            b = b.groupby(pd.Grouper(freq=f'{window_ms}ms')).sum()
            diffs.append(b)
        df = pd.concat(diffs, axis=1)
        df = df.fillna(0)
        return df
