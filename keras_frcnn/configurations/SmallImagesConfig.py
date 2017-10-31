import math
from keras import backend as K

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class SmallImagesConfig(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[8, 16, 24, 32, 64],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(2), 2 / math.sqrt(2)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=200)

    def name(self) -> str:
        return "small_images"


if __name__ == "__main__":
    configuration = SmallImagesConfig()
    print(configuration.summary())
