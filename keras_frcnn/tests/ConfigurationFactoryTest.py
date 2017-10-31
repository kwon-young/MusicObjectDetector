import unittest

from keras_frcnn.configurations.ConfigurationFactory import ConfigurationFactory


class TrainingDatasetProviderTest(unittest.TestCase):
    def test_get_all_configurations(self):
        # Arrange
        configuration_factory = ConfigurationFactory()

        # Act
        configurations = configuration_factory.get_all_configurations()

        # Assert
        self.assertIsNotNone(configurations)
        configuration_names = [c.name() for c in configurations]
        self.assertIn("many_anchor_box_ratios", configuration_names)
        self.assertIn("original_pascal_voc", configuration_names)


if __name__ == '__main__':
    unittest.main()
