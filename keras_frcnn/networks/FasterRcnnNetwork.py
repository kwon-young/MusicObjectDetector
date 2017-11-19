from abc import abstractmethod
from typing import Tuple

from keras import Input
from keras.engine import Layer


class FasterRcnnNetwork(object):
    """Abstract base class for different network architectures used in Faster R-CNN"""

    @abstractmethod
    def name(self) -> str:
        """ Returns the name of this configuration """
        pass

    @abstractmethod
    def get_weight_path(self) -> str:
        """Returns the path to the file that contains the pre-trained weights of this network architecture
        TODO: Refactor to better name """
        pass

    @abstractmethod
    def get_img_output_length(self, width: int, height: int) -> Tuple[int, int]:
        """Returns the image output size with respect to the input size and the given
        network architecture. E.g. if the image has size 200x400 and the network has two
        pooling layers with stride 2x2, the output size will be 50x100.
        TODO: Refactor to better name
        """
        pass

    @abstractmethod
    def nn_base(self, input_tensor: Input = None, trainable: bool = False) -> Layer:
        """Returns the networks base layers, that are used for feature extraction
        and shared for both the Region Proposal Network, as well as the classifier network"""
        pass

    @abstractmethod
    def rpn(self, base_layers: Layer, num_anchors: int):
        """Returns the Region Proposal network, which takes the base layers for feature
        extraction and adds a few more layers that generate region proposals"""
        pass

    @abstractmethod
    def classifier(self, base_layers: Layer, input_rois: Input, num_rois: int, nb_classes: int = 21,
                   trainable: bool = False):
        """Returns the classifier network, which takes the base layers for feature extraction
        as well as the region-proposal from the Region Proposal Network and adds a few more
        layers that actually perform classification and bounding box regression"""
