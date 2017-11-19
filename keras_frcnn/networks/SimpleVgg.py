from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, TimeDistributed

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.networks.FasterRcnnNetwork import FasterRcnnNetwork


class SimpleVgg(FasterRcnnNetwork):
    """Simplified VGG16 model for Keras.
    # Reference
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    """

    def name(self) -> str:
        return "SimpleVgg"

    def get_weight_path(self):
        " We have no pretrained values for this model, so we return None "
        return None

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # We only have two pooling layers, so image size is reduced by factor 4
            return input_length // 4

        return get_output_length(width), get_output_length(height)

    def nn_base(self, input_tensor=None, trainable=False):
        # Determine proper input shape
        input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)

        # Block 4
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)

        return x

    def rpn(self, base_layers, num_anchors):
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
                base_layers)

        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(
                x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                        name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]

    def classifier(self, base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
        pooling_regions = 7

        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)

        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                    name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                                   name='dense_regress_{}'.format(nb_classes))(out)

        return [out_class, out_regr]
