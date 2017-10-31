from abc import abstractmethod
from typing import List
import math


class FasterRcnnConfiguration(object):
    def __init__(self,
                 anchor_box_scales: List[int] = [128, 256, 512],
                 anchor_box_ratios: List[List[int]] = [[1, 1],
                                                       [1 / math.sqrt(2), 2 / math.sqrt(2)],
                                                       [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                 resize_smallest_side_of_image_to: int = 600,
                 use_horizontal_flip_augmentation: bool = False,
                 use_vertical_flip_augmentation: bool = False,
                 use_90_degree_rotation_augmentation: bool = False,
                 verbose: bool = True,
                 image_channel_mean: List[float] = [103.939, 116.779, 123.68],
                 image_scaling_factor: float = 1.0,
                 number_of_ROIs_at_once: int = 32,
                 rpn_stride: int = 16,
                 are_classes_balanced=False,
                 std_scaling: float = 4.0,
                 classifier_regr_std=[8.0, 8.0, 4.0, 4.0],
                 rpn_min_overlap: float = 0.3,
                 rpn_max_overlap: float = 0.7,
                 classifier_min_overlap: float = 0.1,
                 classifier_max_overlap: float = 0.5):
        self.verbose = verbose

        # setting for data augmentation
        self.use_horizontal_flips = use_horizontal_flip_augmentation
        self.use_vertical_flips = use_vertical_flip_augmentation
        self.rot_90 = use_90_degree_rotation_augmentation

        # anchor box scales
        self.anchor_box_scales = anchor_box_scales

        # anchor box ratios
        self.anchor_box_ratios = anchor_box_ratios

        # size to resize the smallest side of the image
        self.resize_smallest_side_of_image_to = resize_smallest_side_of_image_to

        # image channel-wise mean to subtract
        self.img_channel_mean = image_channel_mean
        self.img_scaling_factor = image_scaling_factor

        # number of ROIs at once
        self.num_rois = number_of_ROIs_at_once

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = rpn_stride

        self.balanced_classes = are_classes_balanced

        # scaling the stdev
        self.std_scaling = std_scaling
        self.classifier_regr_std = classifier_regr_std

        # overlaps for RPN
        self.rpn_min_overlap = rpn_min_overlap
        self.rpn_max_overlap = rpn_max_overlap

        # overlaps for classifier ROIs
        self.classifier_min_overlap = classifier_min_overlap
        self.classifier_max_overlap = classifier_max_overlap

        self.class_mapping = {}

    @abstractmethod
    def name(self) -> str:
        """ Returns the name of this configuration """
        pass

    def summary(self) -> str:
        """ Returns the string that summarizes this configuration """

        summary = "Summary of {0}\n==============================================\n".format(self.name())
        summary += "Resizing images, so smallest side has {0} pixel\n".format(self.resize_smallest_side_of_image_to)
        summary += "Anchor box scales: {0}, Anchor box ratios: {1}\n".format(self.anchor_box_scales,
                                                                             self.anchor_box_ratios)
        summary += "Data-augmentation: Horizontal flip: {0}, vertical flip: {1}, 90° rotation: {2}\n".format(
            self.use_horizontal_flips,
            self.use_vertical_flips,
            self.rot_90)
        summary += "Image channel mean {0}, Image scaling factor {1}, STD scaling {2}, classifier regr std {3}\n" \
            .format(self.img_channel_mean, self.img_scaling_factor, self.std_scaling, self.classifier_regr_std)
        summary += "Number of ROIs at once: {0}, RPN Stride: {1}\n".format(self.num_rois, self.rpn_stride)
        summary += "Is the dataset balanced: {0}\n".format(self.balanced_classes)
        summary += "Region Proposal Network overlap: Minimum {0}, Maximum {1}\n".format(self.rpn_min_overlap,
                                                                                        self.rpn_max_overlap)
        summary += "Classifier overlap: Minimum {0}, Maximum {1}\n".format(self.classifier_min_overlap,
                                                                           self.classifier_max_overlap)

        return summary
