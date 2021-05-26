from collections import Callable
from dataclasses import dataclass, field
from os import walk

import numpy as np
import pandas as pd
# This works anyways
# noinspection PyCompatibility
from _lzma import LZMAError
from reamber.osu import OsuMap
from scipy.stats import moment
from tqdm import tqdm

from consts import CONSTS
from osrparse import parse_replay_file
from osrparse.mania import ManiaHitError


@dataclass
class Preprocessing:

    """ Inputs

    Threshold is the look around threshold, that means, how much ms to look around to find related patterns.

    Window is the rolling window aggregation. this defines how precise the rolling should be.

    """
    keys: int
    neighbour_threshold: int = CONSTS.THRESHOLD
    smooth_size: int = CONSTS.SMOOTH_SIZE
    window_ms: float = CONSTS.WINDOW
    _INCLUDE_LN: bool = CONSTS.INCLUDE_LN_AS_START_COMBO

    def load_from(self, map_str):
        """ This will load a map and replays and create the features from it. """
        map_path = f"{CONSTS.RSC_PATH}/{self.keys}/{map_str}"
        _, _, filenames = next(walk(f"{map_path}/{CONSTS.REP_NAME}/"))
        reps = []
        for f in filenames:
            try:
                reps.append(parse_replay_file(f"{map_path}/{CONSTS.REP_NAME}/{f}"))
            except LZMAError:
                print(f"Detected bad Replay on {f}")

        map = OsuMap.readFile(f"{map_path}/{map_str}.osu")
        features = np.nan_to_num(self._make_features(reps, map))
        return self._standardize_by_chunks(features)

    def _make_features(self, reps, map):
        errors = ManiaHitError(reps, map).errors()

        map = [*[np.asarray(h) for h in errors.hit_map],
               *[np.asarray(h) for h in errors.rel_map]]

        df_map_features = self._map_features(map)

        if reps:
            df_rep_error = self._rep_error(errors)
            df = pd.concat([df_map_features, df_rep_error], axis=1)

        else:
            df = df_map_features

        return df.to_numpy()

    def _rep_error(self, errors):
        rep_errors = []
        for rep_hit_error, rep_rel_error in zip(errors.hit_errors, errors.rel_errors):
            rep_errors.append([*[e for k in rep_hit_error for e in k],
                               *[e for k in rep_rel_error for e in k]])

        error_k = []
        for k in range(self.keys):
            for er, map in zip((errors.hit_errors, errors.rel_errors),
                               (errors.hit_map, errors.rel_map)):
                # Prepare the DataFrame
                # Each iteration goes through a specific key. We have n replays.
                df = pd.DataFrame(np.asarray([e[k] for e in er]).transpose(), index=map[k])

                # Change Index to Date for Grouping
                # noinspection PyTypeChecker
                df.index = pd.to_datetime(df.index, unit='ms')
                # We sum the errors by a Window (ms) specified
                df = df.groupby(pd.Grouper(freq=f'{CONSTS.WINDOW}ms')).sum()

                if df.empty:
                    error_k.append(df.mean(axis=1))
                else:
                    # We smooth the absolute errors by Windows * Smooth (ms)
                    # Also, converge them as mean
                    # We could also include the stdev but idk if we need a confidence window.
                    ar = np.abs(df.to_numpy()).mean(axis=1)
                    # We append this as the error per key
                    error_k.append(pd.DataFrame(ar,index=df.index))

        return self._rearrange_errors(pd.concat(error_k,axis=1))

    def _map_features(self, map):
        # We strictly do not render the Release -> Any combination.
        CANDIDATES = self.keys * 2 if CONSTS.INCLUDE_LN_AS_START_COMBO else self.keys
        diffs = []
        for from_i in tqdm(range(CANDIDATES), desc="Feature Calculation Progress"):
            # We have the first column as the offset
            b = np.zeros([CANDIDATES * CONSTS.MOMENTS + 1, len(map[from_i])])
            b[0] = map[from_i]
            for to_i in range(CANDIDATES):
                a = np.abs(map[from_i] - map[to_i][..., None])
                a = np.where(a < CONSTS.THRESHOLD, 1 / (1 + a / 300), np.nan)
                # b[to_i * CONSTS.MOMENTS + 1] = np.count_nonzero(~np.isnan(a),axis=0)
                b[to_i * CONSTS.MOMENTS + 1] = np.nansum(a, axis=0)
                b[to_i * CONSTS.MOMENTS + 2] = moment(a, 2, nan_policy='omit', axis=0)
                b[to_i * CONSTS.MOMENTS + 3] = moment(a, 3, nan_policy='omit', axis=0)
                b[to_i * CONSTS.MOMENTS + 4] = moment(a, 4, nan_policy='omit', axis=0)
            b = pd.DataFrame(b.transpose()).set_index(0)
            b.index = pd.to_datetime(b.index, unit='ms')

            # If key = 4 from = 0
            # 0: Count of 0 -> 0 | 4: Count of 0 -> 1 ...
            # 1: M1 of    0 -> 0 | 5: M1 of    0 -> 1
            # 2: M2 of    0 -> 0 | 6: M2 of    0 -> 1
            # 3: M3 of    0 -> 0 | 7: M3 of    0 -> 1

            # Thus if Key = 4 and No LN, we expect 16 features per key
            b = b.groupby(pd.Grouper(freq=f'{CONSTS.WINDOW}ms')).sum()
            diffs.append(b)
        df = pd.concat(diffs, axis=1).fillna(0)
        return self._rearrange_moments(df)

    def _rearrange_moments(self, df):
        # This complex function swaps the columns so that the moments are all together
        return df.iloc[:, (np.r_[[np.arange(i, self._input_size, CONSTS.MOMENTS)
                  for i in range(CONSTS.MOMENTS)]].flatten())]

    def _rearrange_errors(self, df):
        # This complex function swaps the columns so that the hit and rel are together
        return df.iloc[:,(np.r_[[np.arange(i, self._output_size, 2) for i in range(2)]]).flatten()]

    def _standardize_by_chunks(self, ar):
        # The fences define where the chunks should slice
        fences = [*np.arange(0, self._input_size, self.keys ** 2),
                  *(np.arange(0, self._output_size, self.keys) + self._input_size),
                  ar.shape[-1]]
        for a, b in zip(fences[:-1], fences[1:]):
            chunk = ar[...,a:b]
            min, max = chunk.min(), chunk.max()
            if max - min == 0:
                ar[..., a:b] = (chunk - min)
            else:
                ar[..., a:b] = (chunk - min) / (max - min)

        return ar

    @property
    def _input_size(self):
        return self.keys ** 2 * CONSTS.MOMENTS

    @property
    def _output_size(self):
        return self.keys * 2

