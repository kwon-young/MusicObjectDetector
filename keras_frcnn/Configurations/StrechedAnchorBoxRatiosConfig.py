from keras import backend as K

from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class StrechedAnchorBoxRatiosConfig(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(network='resnet50',
                         anchor_box_scales=[16, 24, 32, 64, 128], anchor_box_ratios=[[1, 1], [1, 3], [3, 1]],
                         resize_smallest_side_of_image_to=350)

    def name(self) -> str:
        return "streched_anchor_box_ratios"

if __name__ == "__main__":
    configuration = StrechedAnchorBoxRatiosConfig()
    print(configuration.summary())
