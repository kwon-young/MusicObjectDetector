from typing import List

from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration
from keras_frcnn.Configurations.ManyAnchorBoxScalesConfig import ManyAnchorBoxScalesConfig
from keras_frcnn.Configurations.OriginalPascalVocConfig import OriginalPascalVocConfig


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
                          ManyAnchorBoxScalesConfig()]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations()
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
