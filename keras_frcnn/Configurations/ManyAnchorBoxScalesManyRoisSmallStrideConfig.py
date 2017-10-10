import math
from keras import backend as K

from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class ManyAnchorBoxScalesManyRoisSmallStrideConfig(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(network='resnet50',
                         anchor_box_scales=[16, 24, 32, 64, 128, 256],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(2), 2 / math.sqrt(2)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=350, number_of_ROIs_at_once=64, rpn_stride=4)

    def name(self) -> str:
        return "many_anchor_box_scales_many_rois_small_stride"

if __name__ == "__main__":
    configuration = ManyAnchorBoxScalesManyRoisSmallStrideConfig()
    print(configuration.summary())
