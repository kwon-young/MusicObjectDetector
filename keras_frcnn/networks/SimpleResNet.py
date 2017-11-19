from keras import backend as K
from keras.layers import Activation, AveragePooling2D, Convolution2D, Dense, Flatten, Input, MaxPooling2D, \
    TimeDistributed, ZeroPadding2D

from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.networks.ResNet50 import ResNet50


class SimpleResNet(ResNet50):
    """ Simplified ResNet model for Keras.
        # Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        Adapted from code contributed by BigMoyan.
    """

    def name(self) -> str:
        return "SimpleResNet"

    def get_weight_path(self):
        return None

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

        bn_axis = 3

        x = ZeroPadding2D((3, 3))(img_input)

        x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
        x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1), trainable=trainable)
        x = self.conv_block(x, 3, [128, 128, 256], stage=3, block='a', trainable=trainable)
        x = self.conv_block(x, 3, [256, 256, 512], stage=4, block='a', trainable=trainable)

        return x

    def classifier_layers(self, x, input_shape, trainable=False):
        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
        # (hence a smaller stride in the region that follows the ROI pool)
        x = self.conv_block_td(x, 3, [256, 256, 1024], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                               trainable=trainable)

        x = self.identity_block_td(x, 3, [256, 256, 1024], stage=5, block='b', trainable=trainable)
        x = self.identity_block_td(x, 3, [256, 256, 1024], stage=5, block='c', trainable=trainable)
        x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

        return x

    def classifier(self, base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
        # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 512)

        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
        out = self.classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

        out = TimeDistributed(Flatten())(out)

        out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                    name='dense_class_{}'.format(nb_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                                   name='dense_regress_{}'.format(nb_classes))(out)
        return [out_class, out_regr]
