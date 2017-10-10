import math

from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class OriginalPascalVocConfig(FasterRcnnConfiguration):
    """The original configuration that was provided along the keras_frcnn sample implementation """

    def __init__(self):
        super().__init__('resnet50',
                         anchor_box_scales=[128, 256, 512],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(2), 2 / math.sqrt(2)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=600,
                         use_horizontal_flip_augmentation=False,
                         use_vertical_flip_augmentation=False,
                         use_90_degree_rotation_augmentation=False,
                         verbose=True,
                         image_channel_mean=[103.939, 116.779, 123.68],
                         image_scaling_factor=1.0,
                         number_of_ROIs_at_once=32,
                         rpn_stride=16,
                         are_classes_balanced=False,
                         std_scaling=4.0,
                         classifier_regr_std=[8.0, 8.0, 4.0, 4.0],
                         rpn_min_overlap=0.3,
                         rpn_max_overlap=0.7,
                         classifier_min_overlap=0.1,
                         classifier_max_overlap=0.5)

    def name(self) -> str:
        return "original_pascal_voc"


if __name__ == "__main__":
    configuration = OriginalPascalVocConfig()
    print(configuration.summary())
