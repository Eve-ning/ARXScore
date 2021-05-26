import os

from dataclasses import dataclass
from os import walk

from consts import CONSTS
from input import Preprocessing

import numpy as np


@dataclass
class Resource:
    threshold: int = CONSTS.THRESHOLD
    window:int = CONSTS.WINDOW
    overwrite_data: bool = False

    def features(self, keys:int, get_new_only=False, lim=None, ver="0") -> list:
        _, dirs, filenames = next(walk(f'{CONSTS.RSC_PATH}/{keys}'))
        trainables = []
        if lim: dirs = np.asarray(dirs)[np.random.choice(len(dirs), lim, replace=False)]
        for map_dir in dirs:
            data_dir = f"{CONSTS.RSC_DATA_PATH}/{keys}/" \
                       f"version{ver}_{self.threshold}_{self.window}/" \
                       f"{map_dir}"
            data_path = f"{data_dir}/{CONSTS.DATA_NAME}.npy"

            if not os.path.exists(data_path) or self.overwrite_data:
                # This means it's new data
                # There may be cases where the dir is generated but there's no data
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                print(f"Generating Data on {map_dir}")
                data = Preprocessing(keys, self.threshold, self.window).load_from(map_dir)
                np.save(data_path, data)
                trainables.append(data)
            elif not get_new_only:
                # This means we get the old maps
                data = np.load(f"{data_path}")
                print(f"Getting {map_dir}")
                trainables.append(data)

        if len(trainables) == 0:
            print("No new Data to get.")

        return trainables
