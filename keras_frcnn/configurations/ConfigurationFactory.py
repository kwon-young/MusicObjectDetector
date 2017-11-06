from typing import List

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration
from keras_frcnn.configurations.ManyAnchorBoxRatiosConfig import ManyAnchorBoxRatiosConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisLargeStrideConfig import \
    ManyAnchorBoxScalesManyRoisLargeStrideConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisMediumStrideConfig import \
    ManyAnchorBoxScalesManyRoisMediumStrideConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisSmallStrideConfig import \
    ManyAnchorBoxScalesManyRoisSmallStrideConfig
from keras_frcnn.configurations.SmallAnchorBoxScalesManyRoisConfig import SmallAnchorBoxScalesManyRoisConfig
from keras_frcnn.configurations.SmallImagesConfig import SmallImagesConfig
from keras_frcnn.configurations.StretchedAnchorBoxRatiosConfig import StretchedAnchorBoxRatiosConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesConfig import ManyAnchorBoxScalesConfig
from keras_frcnn.configurations.OriginalPascalVocConfig import OriginalPascalVocConfig
from keras_frcnn.configurations.SmallAnchorBoxScalesConfig import SmallAnchorBoxScalesConfig
from keras_frcnn.configurations.VggSmallImagesConfig import VggSmallImagesConfig


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str) -> FasterRcnnConfiguration:

        configurations = ConfigurationFactory.get_all_configurations()

        for i in range(len(configurations)):
            if configurations[i].name() == name:
                return configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations() -> List[FasterRcnnConfiguration]:
        configurations = [OriginalPascalVocConfig(),
                          ManyAnchorBoxScalesConfig(),
                          StretchedAnchorBoxRatiosConfig(),
                          SmallAnchorBoxScalesConfig(),
                          ManyAnchorBoxScalesManyRoisSmallStrideConfig(),
                          ManyAnchorBoxScalesManyRoisMediumStrideConfig(),
                          ManyAnchorBoxScalesManyRoisLargeStrideConfig(),
                          SmallImagesConfig(),
                          ManyAnchorBoxRatiosConfig(),
                          VggSmallImagesConfig(),
                          SmallAnchorBoxScalesManyRoisConfig()]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations()
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
