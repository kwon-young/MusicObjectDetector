import math
from keras import backend as K

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class AutomaticConfig(FasterRcnnConfiguration):
    """A faster-R-CNN configuration that detects the parameters, by inspection of the
    ground truth dataset and deriving meaningful defaults for anchor box scales, ratios and image sizes"""

    def __init__(self):
        # TODO: Implement meaningful statistics calculation
        pass

    def name(self) -> str:
        return "automatic"


if __name__ == "__main__":
    configuration = AutomaticConfig()
    print(configuration.summary())
