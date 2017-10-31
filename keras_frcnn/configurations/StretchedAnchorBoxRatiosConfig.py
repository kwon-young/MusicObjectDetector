import math
from keras import backend as K

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class StretchedAnchorBoxRatiosConfig(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[16, 24, 32, 64, 128],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(3), 3 / math.sqrt(3)],
                                            [3 / math.sqrt(3), 1 / math.sqrt(3)]],
                         resize_smallest_side_of_image_to=350)

    def name(self) -> str:
        return "stretched_anchor_box_ratios"


if __name__ == "__main__":
    configuration = StretchedAnchorBoxRatiosConfig()
    print(configuration.summary())
