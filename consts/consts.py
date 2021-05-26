import numpy as np


class CONSTS:
    RSC_PATH = "rsc"
    RSC_DATA_PATH = "rsc_data"
    RSC_UNSORTED_PATH = "rsc_unsorted"
    DATA_NAME = "data"
    REP_NAME = "rep"
    MODEL_PATH = "models"
    MOMENTS = 4

    THRESHOLD = 2000
    EPOCHS = 50

    BATCH_SIZE = 48
    WINDOW = 1000

    # This is used to smooth the replay data so that it's not too jagged.
    # This uses a Gaussian Kernel.
    # Size is the shape, STD is the standard deviation.
    SMOOTH_SIZE = 3
    SMOOTH_STD = 1

    # This means to include LN as starting combinations for pattern processing
    INCLUDE_LN_AS_START_COMBO = False

    @staticmethod
    def INPUT_SIZE(key):
        return key ** 2 * CONSTS.MOMENTS
    @staticmethod
    def OUTPUT_SIZE(key):
        return key * 2