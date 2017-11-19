import unittest

from keras_frcnn.configurations.ConfigurationFactory import ConfigurationFactory
from keras_frcnn.networks.NetworkFactory import NetworkFactory


class NetworkFactoryTest(unittest.TestCase):
    def test_get_all_networks(self):
        # Arrange
        network_factory = NetworkFactory()

        # Act
        networks = network_factory.get_all_configurations()

        # Assert
        self.assertIsNotNone(networks)
        configuration_names = [c.name() for c in networks]
        self.assertIn("ResNet50", configuration_names)
        self.assertIn("Vgg16", configuration_names)


if __name__ == '__main__':
    unittest.main()
