from dataclasses import dataclass
from os import walk
from typing import List, Tuple

import numpy as np
import pandas as pd
# This works anyways
# noinspection PyCompatibility
from _lzma import LZMAError
from reamber.osu import OsuMap
from scipy.stats import moment
from sklearn.preprocessing import robust_scale
from tqdm import tqdm

from consts import CONSTS
from osrparse import parse_replay_file, Replay
from osrparse.mania import ManiaHitError
from osrparse.mania.hit_error import ManiaHitErrorEvents


@dataclass
class Preprocessing:

    """ Inputs

    Threshold is the look around threshold, that means, how much ms to look around to find related patterns.

    Window is the rolling window aggregation. this defines how precise the rolling should be.

    """
    keys: int
    scale: bool = True
    neighbour_threshold: int = CONSTS.NEIGHBOUR_THRESHOLD
    smooth_size: int = CONSTS.SMOOTH_SIZE
    window_ms: float = CONSTS.WINDOW
    _INCLUDE_LN: bool = CONSTS.INCLUDE_LN_AS_START_COMBO

    def features_from_rsc(self, map_str: str) -> np.ndarray:
        """ This will load a map and replays and create the features from it. """
        map_path = f"{CONSTS.RSC_PATH}/{self.keys}/{map_str}"
        rep_paths = next(walk(f"{map_path}/{CONSTS.REP_NAME}/"))[-1]
        return self.features_from_path(
            map_path=f"{map_path}/{map_str}.osu",
            rep_paths=[f"{map_path}/{CONSTS.REP_NAME}/{f}" for f in rep_paths])

    def features_from_path(self, map_path: str, rep_paths: List[str] = []):
        reps = []
        map = OsuMap.readFile(map_path)
        for f in rep_paths:
            try:
                reps.append(parse_replay_file(f))
            except LZMAError:
                print(f"Detected bad Replay on {f}")

        return self.features(map=map, reps=reps)

    def features(self, map: OsuMap, reps: List[Replay]) -> np.ndarray:
        """ Main function to handle the features

        If loading from rsc pool, use features_from_rsc
        If loading from path, use features_from_path
        """
        errors = ManiaHitError(reps, map).errors()
        df = self._map_features(errors)
        if reps: df = pd.concat([df, self._rep_error(errors)], axis=1) if reps else df
        return np.nan_to_num(df.to_numpy())

    def _rep_error(self, errors: ManiaHitErrorEvents) -> pd.DataFrame:
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
                df = df.abs().groupby(pd.Grouper(freq=f'{CONSTS.WINDOW}ms')).sum()

                if df.empty:
                    error_k.append(df.median(axis=1))
                else:
                    # We smooth the absolute errors by Windows * Smooth (ms)
                    # Also, converge them as median
                    # We could also include the stdev but idk if we need a confidence window.
                    ar = np.median(np.abs(df.to_numpy()), axis=1)
                    # We append this as the error per key
                    error_k.append(pd.DataFrame(ar,index=df.index))

        return self._rearr_output(pd.concat(error_k, axis=1).fillna(0))

    def _map_features(self, errors: ManiaHitErrorEvents) -> pd.DataFrame:
        map = [*[np.asarray(h) for h in errors.hit_map],
               *[np.asarray(h) for h in errors.rel_map]]
        
        INPUTS = self.keys * 2
        features = []
        col_names = []
        for from_i in tqdm(range(INPUTS), desc="Feature Calculation Progress"):
            features_k = np.zeros([INPUTS * CONSTS.FEATURE_PER_COMBO, len(map[from_i])])
            for to_i in range(INPUTS):
                diff_k = np.abs(map[from_i] - map[to_i][..., None])
                diff_k = np.where(diff_k < CONSTS.NEIGHBOUR_THRESHOLD,
                                  1 / (1 + diff_k / CONSTS.DIFF_CORRECTION_FACTOR), np.nan)

                if from_i == to_i:
                    # Do not include itself as a match.
                    np.fill_diagonal(diff_k, np.nan)

                all_nan_slices = np.all(np.isnan(diff_k),axis=0)
                diff_k[:, all_nan_slices] = 0
                # NEAREST ONLY
                try:
                    features_k[to_i * CONSTS.FEATURE_PER_COMBO + 0] = np.nanmax(diff_k, axis=0)
                except ValueError:
                    features_k[to_i * CONSTS.FEATURE_PER_COMBO + 0] = 0
                col_names.append(f"{from_i}_{to_i}max")
                # SUM
                features_k[to_i * CONSTS.FEATURE_PER_COMBO + 1] = np.nansum(diff_k, axis=0)
                col_names.append(f"{from_i}_{to_i}sum")
                # MEAN
                features_k[to_i * CONSTS.FEATURE_PER_COMBO + 2] = moment(diff_k, 2, nan_policy='omit', axis=0)
                col_names.append(f"{from_i}_{to_i}mean")
                # VAR
                features_k[to_i * CONSTS.FEATURE_PER_COMBO + 3] = moment(diff_k, 3, nan_policy='omit', axis=0)
                col_names.append(f"{from_i}_{to_i}var")
                # SKEW
                features_k[to_i * CONSTS.FEATURE_PER_COMBO + 4] = moment(diff_k, 4, nan_policy='omit', axis=0)
                col_names.append(f"{from_i}_{to_i}skew")
            features_k = pd.DataFrame(features_k.transpose(), index=map[from_i])
            # noinspection PyTypeChecker
            features_k.index = pd.to_datetime(features_k.index, unit='ms')

            features_k = features_k.groupby(pd.Grouper(freq=f'{CONSTS.WINDOW}ms')).sum()
            features.append(features_k)
        df = pd.concat(features, axis=1).fillna(0)
        df.columns = col_names
        return self._rearr_input(df)

    @staticmethod
    def feature_names(keys: int) -> List[str]:
        OUTPUTS = keys * 2
        col_names = []
        for from_i in range(OUTPUTS):
            for to_i in range(OUTPUTS):
                col_names.append(f"{from_i}_{to_i}max")
                col_names.append(f"{from_i}_{to_i}sum")
                col_names.append(f"{from_i}_{to_i}mean")
                col_names.append(f"{from_i}_{to_i}var")
                col_names.append(f"{from_i}_{to_i}skew")
        return col_names

    def _rearr_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # This complex function swaps the columns so that the aggs are all together
        return df.iloc[:, (np.r_[[np.arange(i, self._input_size * 4, CONSTS.FEATURE_PER_COMBO)
                                  for i in range(CONSTS.FEATURE_PER_COMBO)]].flatten())]

    def _rearr_output(self, df: pd.DataFrame) -> pd.DataFrame:
        # This complex function swaps the columns so that the hit and rel are together
        return df.iloc[:,(np.r_[[np.arange(i, self._output_size, 2) for i in range(2)]]).flatten()]

    def _scale_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale: return df
        fences = np.arange(0, self._input_size * 4 + self.keys ** 2, self.keys ** 2)
        for a, b in zip(fences[:-1], fences[1:]):
            df.iloc[:, a:b] = self._scale_chunk(df.iloc[:, a:b],
                                                method_=CONSTS.CHUNK_SCALING_METHOD[0])
        return df

    def _scale_output(self, df:pd.DataFrame) -> pd.DataFrame:
        if not self.scale: return df
        fences = np.arange(0, self._output_size + self.keys, self.keys)

        for a, b in zip(fences[:-1], fences[1:]):
            df.iloc[:, a:b] = self._scale_chunk(df.iloc[:, a:b],
                                                method_=CONSTS.CHUNK_SCALING_METHOD[1])
        return df

    @staticmethod
    def _scale_chunk(chunk: pd.DataFrame, method_: Tuple):
        if method_ == 'minmax':
            # noinspection PyShadowingBuiltins
            min, max = np.min(chunk.to_numpy()), np.max(chunk.to_numpy())
            if max - min == 0: return (chunk - min)
            else:              return (chunk - min) / (max - min)
        elif method_ == 'std':
            mean, var = np.mean(chunk.to_numpy()), np.var(chunk.to_numpy())
            if var == 0: return (chunk - mean)
            else:        return (chunk - mean) / var
        elif method_ == 'robust':
            return robust_scale(chunk.flatten(), unit_variance=True).reshape(chunk.shape)

    @property
    def _input_size(self) -> int:
        return self.keys ** 2 * CONSTS.FEATURE_PER_COMBO

    @property
    def _output_size(self) -> int:
        return self.keys * 2
