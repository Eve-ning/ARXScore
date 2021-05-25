from collections import Callable
from dataclasses import dataclass
from os import walk

import numpy as np
import pandas as pd
from _lzma import LZMAError
from reamber.osu import OsuMap
from tqdm import tqdm

from consts import CONSTS
from osrparse import parse_replay_file
from osrparse.mania import ManiaHitError


@dataclass
class Input:

    """ Inputs

    Threshold is the look around threshold, that means, how much ms to look around to find related patterns.

    Window is the rolling window aggregation. this defines how precise the rolling should be.

    Aggregation is called upon the errors received by load_from, this will give you different shapes depending on this.

    """
    threshold: int = CONSTS.THRESHOLD
    window_ms: float = CONSTS.WINDOW
    aggregated: bool = False
    aggregation_method: Callable = np.median

    def load_from(self, map_str):
        """ This will load a map and replays and create the features from it.

        Note that the features are affected by aggregated and aggregation_method.

        If aggregated is False, you will receive output of the individual key errors on hit and release,
        else you'd get an aggregated column of those.
        """
        map_path = f"{CONSTS.RSC_PATH}/{map_str}"
        _, _, filenames = next(walk(f"{map_path}/{CONSTS.REP_NAME}/"))
        reps = []
        for f in filenames:
            try:
                reps.append(parse_replay_file(f"{map_path}/{CONSTS.REP_NAME}/{f}"))
            except LZMAError:
                print(f"Detected bad Replay on {f}")
        map = OsuMap.readFile(f"{map_path}/{map_str}.osu")
        return self._create_data(reps, map)

    def _create_data(self, reps, map):
        errors = ManiaHitError(reps, map).errors()

        map = [*[np.asarray(h) for h in errors.hit_map],
               *[np.asarray(h) for h in errors.rel_map]]

        df_map_features = self._minmax(
            self._map_features(map, self.window_ms, self.threshold)
        )
        if reps:
            df_rep_error = self._rep_error(errors, self.window_ms)
            if self.aggregated:
                df_rep_error = pd.DataFrame(self.aggregation_method(df_rep_error,axis=1),
                                            index=df_rep_error.index)
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

        keys = len(errors.hit_map)
        error_k = []
        for k in range(keys):
            df = pd.DataFrame(np.asarray([e[k] for e in errors.hit_errors]).transpose(),
                         index=errors.hit_map[k])
            df.index = pd.to_datetime(df.index, unit='ms')
            df = df.groupby(pd.Grouper(freq=f'{window_ms}ms')).sum()
            error_k.append(df.mean(axis=1))

            df = pd.DataFrame(np.asarray([e[k] for e in errors.rel_errors]).transpose(),
                         index=errors.rel_map[k])
            df.index = pd.to_datetime(df.index, unit='ms')
            df = df.groupby(pd.Grouper(freq=f'{window_ms}ms')).sum()
            error_k.append(df.mean(axis=1))

        return Input._std(pd.concat(error_k,axis=1))

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
        return Input._std(df)
