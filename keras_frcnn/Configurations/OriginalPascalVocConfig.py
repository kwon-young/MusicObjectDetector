from keras_frcnn.Configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class OriginalPascalVocConfig(FasterRcnnConfiguration):
    """The original configuration that was provided along the keras_frcnn sample implementation """

    def __init__(self):
        super().__init__('resnet50', [128, 256, 512], [[1, 1], [1, 2], [2, 1]], 600, False, False, False, True,
                         [103.939, 116.779, 123.68], 1.0, 32, 16, False, 4.0, [8.0, 8.0, 4.0, 4.0], 0.3, 0.7, 0.1, 0.5)

    def name(self) -> str:
        return "original_pascal_voc"


if __name__ == "__main__":
    configuration = OriginalPascalVocConfig()
    print(configuration.summary())
