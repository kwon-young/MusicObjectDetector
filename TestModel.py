from __future__ import division

import os
import pickle
import time
import traceback
from argparse import ArgumentParser

import cv2
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

from keras_frcnn import roi_helpers
from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration

parser = ArgumentParser()

parser.add_argument("-p", "--testdata_path", dest="testdata_path", help="Path to test data.")
parser.add_argument("-n", "--num_rois",
                    type=int,
                    dest="num_rois",
                    help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_argument("--config_path",
                    dest="config_path",
                    help="Location to read the metadata related to the training (generated when training).",
                    default="config.pickle")
parser.add_argument("--model_path",
                    dest="model_path",
                    help="Path to the *.hdf5 file that contains the saved model that should be loaded for inference")
parser.add_argument("--model_name", type=str, default="resnet50",
                    help="The model used for training the network. Currently one of [vgg, resnet50]")
parser.add_argument("-v", "--verbose", dest="verbose", help="Prints a verbose output while detecting objects.",
                    action="store_true", default=False)
parser.add_argument("--non_max_suppression_max_boxes",
                    type=int,
                    dest="non_max_suppression_max_boxes",
                    help="Number of boxes to keep from non-maximum suppressions", default=300)
parser.add_argument("--non_max_suppression_overlap_threshold",
                    type=float,
                    dest="non_max_suppression_overlap_threshold",
                    help="Overlap threshold for non-maximum suppressions (between 0.0 - 1.0)", default=0.7)
parser.add_argument("--classification_accuracy_threshold",
                    type=float,
                    dest="classification_accuracy_threshold",
                    help="Threshold to accept classifications as hits (between 0.0 - 1.0)", default=0.4)

options, unparsed = parser.parse_known_args()

if not options.testdata_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_path
verbose = options.verbose
model_path = options.model_path
model_name = options.model_name
path_to_test_images = options.testdata_path
num_rois = int(options.num_rois)
non_max_suppression_overlap_threshold = options.non_max_suppression_overlap_threshold
non_max_suppression_max_boxes = int(options.non_max_suppression_max_boxes)
classification_accuracy_threshold = options.classification_accuracy_threshold

if model_name not in ['resnet50', 'vgg']:
    raise ValueError(
        "Currently only resnet50 and vgg are supported model names, but {0} was provided".format(model_name))

with open(config_output_filename, 'rb') as f_in:
    C: FasterRcnnConfiguration = pickle.load(f_in)

if model_name == 'resnet50':
    import keras_frcnn.networks.resnet as nn
elif model_name == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False


def format_img_size(img, C: FasterRcnnConfiguration):
    """ formats the image size based on config """
    img_min_side = float(C.resize_smallest_side_of_image_to)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
np.random.seed(1)  # For creating reproducible random-colors
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(num_rois)

if model_name == 'resnet50':
    num_features = 1024
elif model_name == 'vgg':
    num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

if verbose:
    test_images = sorted(os.listdir(path_to_test_images))
else:
    test_images = tqdm(sorted(os.listdir(path_to_test_images)), desc="Detecting music objects")

for img_name in test_images:
    try:
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue

        if verbose:
            print("Processing {0}".format(img_name))

        starting_time = time.time()
        filepath = os.path.join(path_to_test_images, img_name)

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C,
                                   overlap_thresh=non_max_suppression_overlap_threshold,
                                   max_boxes=non_max_suppression_max_boxes)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // num_rois + 1):
            ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                classifications = P_cls[0, ii, :]
                max_classification = np.max(classifications)
                most_likely_class = np.argmax(classifications)
                background_class = (P_cls.shape[2] - 1)
                if max_classification < classification_accuracy_threshold:
                    continue

                if most_likely_class == background_class:
                    continue

                cls_name = class_mapping[most_likely_class]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = most_likely_class
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except Exception as ex:
                    traceback.print_exc()
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(max_classification)

        all_detected_objects = []
        detected_instances = 0

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                        overlap_thresh=non_max_suppression_overlap_threshold,
                                                                        max_boxes=non_max_suppression_max_boxes)
            for jk in range(new_boxes.shape[0]):
                detected_instances += 1
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_detected_objects.append((key, "{0:0.2f}".format(100 * new_probs[jk])))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                #              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                #              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                # cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        if verbose:
            print("Detected {0} instances from {1} classes in {2:.1f}s".format(detected_instances,
                                                                               len(bboxes.keys()),
                                                                               time.time() - starting_time))
            print(all_detected_objects)
            print("")
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        file_name_without_extension = os.path.splitext(os.path.basename(img_name))[0]
        cv2.imwrite('./image_results/{0}_detect_{1}-rois_{2}-boxes_{3}-overlap_{4}-accuracy-threshold.png'
                    .format(file_name_without_extension, num_rois, non_max_suppression_max_boxes,
                            non_max_suppression_overlap_threshold, classification_accuracy_threshold), img)

    except Exception as ex:
        print("Error while detecting objects in {0}: {1}".format(img_name, ex))
