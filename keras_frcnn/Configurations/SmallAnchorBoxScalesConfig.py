from keras import backend as K

from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class SmallAnchorBoxScalesConfig(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(network='resnet50',
                         anchor_box_scales=[16, 24, 32, 64], anchor_box_ratios=[[1, 1], [1, 2], [2, 1]],
                         resize_smallest_side_of_image_to=350)

    def name(self) -> str:
        return "small_anchor_box_scales"

if __name__ == "__main__":
    configuration = SmallAnchorBoxScalesConfig()
    print(configuration.summary())
