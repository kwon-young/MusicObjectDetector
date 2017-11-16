import argparse
import datetime
import os
import pickle
import pprint
import random
import shutil
import time

import numpy as np
import tensorflow
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import generic_utils
from omrdatasettools.converters.ImageInverter import ImageInverter
from omrdatasettools.downloaders.CvcMuscimaDatasetDownloader import CvcMuscimaDatasetDownloader, CvcMuscimaDataset
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn import faster_rcnn_losses, data_generators
from keras_frcnn.configurations.ConfigurationFactory import ConfigurationFactory
from keras_frcnn.muscima_image_cutter import delete_unused_images, cut_images
from keras_frcnn.muscima_pp_cropped_image_parser import get_data
from keras_frcnn.reporting import TelegramNotifier, GoogleSpreadsheetReporter

try:
    from keras_frcnn import data_generators_fast

    use_fast_data_generators = True
except:
    print("Could not import fast data_generators. Falling back to slow data_generator. See README for more information")
    use_fast_data_generators = False


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tensorflow.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def train_model(dataset_directory: str, model_name: str, delete_and_recreate_dataset_directory: bool,
                configuration_name: str, output_weight_path: str, configuration_filename: str, number_of_epochs: int,
                early_stopping: int, learning_rate_reduction_patience: int,
                learning_rate_reduction_factor: float, non_max_suppression_overlap_threshold: float,
                non_max_suppression_max_boxes: int, input_weight_path: str = None):
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")
    muscima_cropped_directory = os.path.join(dataset_directory, "muscima_pp_cropped_images")

    if not dataset_directory:  # if filename is not given
        parser.error('Error: path to training data must be specified. Pass --path to command line')

    if model_name == 'vgg':
        from keras_frcnn import vgg as nn
    elif model_name == 'resnet50':
        from keras_frcnn.networks import resnet as nn
    elif model_name == 'simple_resnet':
        from keras_frcnn.networks import simple_resnet as nn
    else:
        print('Not a valid model')
        raise ValueError

    try:
        all_images, classes_count, class_mapping = get_data(muscima_cropped_directory)
        data_loaded = True
    except:
        print("Could not load dataset. Automatically downloading and recreating dataset.")
        data_loaded = False
        delete_and_recreate_dataset_directory = True

    if delete_and_recreate_dataset_directory:
        print("Deleting dataset directory {0}".format(dataset_directory))
        if os.path.exists(dataset_directory):
            shutil.rmtree(dataset_directory)

        downloader = MuscimaPlusPlusDatasetDownloader(muscima_pp_raw_dataset_directory)
        downloader.download_and_extract_dataset()

        downloader = CvcMuscimaDatasetDownloader(muscima_image_directory, CvcMuscimaDataset.StaffRemoval)
        downloader.download_and_extract_dataset()

        delete_unused_images(muscima_image_directory)

        inverter = ImageInverter()
        # We would like to work with black-on-white images instead of white-on-black images
        inverter.invert_images(muscima_image_directory, "*.png")

        shutil.copy("Staff-Vertical-Positions.txt", dataset_directory)

        cut_images(muscima_image_directory, os.path.join(dataset_directory, "Staff-Vertical-Positions.txt"),
                   muscima_cropped_directory, muscima_pp_raw_dataset_directory)

    # pass the settings from the command line, and persist them in the config object
    C = ConfigurationFactory.get_configuration_by_name(configuration_name)
    C.model_path = output_weight_path
    start_time = time.time()

    if not data_loaded:
        all_images, classes_count, class_mapping = get_data(muscima_cropped_directory)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    # inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    print('Hyperparameters: {0} RoIs generated per run with {1} boxes remaining from non-max suppression and using '
          'non-max suppression threshold of {2:.2f}'.format(C.num_rois, non_max_suppression_max_boxes,
                                                            non_max_suppression_overlap_threshold))

    config_output_filename = configuration_filename

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))

    random.seed(1)
    random.shuffle(all_images)

    # num_imgs = len(all_images)

    train_imgs = [s for s in all_images if s['imageset'] == 'train']
    val_imgs = [s for s in all_images if s['imageset'] == 'val']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    if not use_fast_data_generators:
        print("Using standard data_generator")
        data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,
                                                       mode='train')
        data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode='val')
    else:
        print("Using fast data_generator")
        data_gen_train = data_generators_fast.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,
                                                            mode='train')
        data_gen_val = data_generators_fast.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,
                                                          mode='val')

    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    start_of_training = datetime.date.today()
    tensorboard_callback = TensorBoard(
        log_dir="./logs/{0}_{1}/".format(start_of_training, configuration_name))
    tensorboard_callback.set_model(model_all)

    try:
        print('Loading weights from {0}'.format(input_weight_path))
        model_rpn.load_weights(input_weight_path, by_name=True)
        model_classifier.load_weights(input_weight_path, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adadelta()
    optimizer_classifier = Adadelta()
    model_rpn.compile(optimizer=optimizer, loss=[faster_rcnn_losses.rpn_loss_cls(num_anchors),
                                                 faster_rcnn_losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[faster_rcnn_losses.class_loss_cls,
                                   faster_rcnn_losses.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer=Adadelta(), loss='mae')

    epoch_length = 1000
    validation_epoch_length = len(val_imgs)
    validation_interval = 1

    losses = np.zeros((epoch_length, 5))
    losses_val = np.zeros((validation_epoch_length, 5))

    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    best_loss_training = np.inf
    best_loss_epoch = 0
    best_total_loss_validation = np.Inf
    best_loss_rpn_cls = np.inf
    best_loss_rpn_regr = np.inf
    best_loss_class_cls = np.inf
    best_loss_class_regr = np.inf
    best_class_acc = 0.0

    model_classifier.summary()
    print(C.summary())

    print('Starting training')

    train_names = ['train_loss_rpn_cls', 'train_loss_rpn_reg', 'train_loss_class_cls', 'train_loss_class_reg',
                   'train_total_loss', 'train_acc']
    val_names = ['val_loss_rpn_cls', 'val_loss_rpn_reg', 'val_loss_class_cls', 'val_loss_class_reg', 'val_total_loss',
                 'val_acc']
    epochs_without_improvement = 0

    for epoch_num in range(number_of_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, number_of_epochs))

        for iter_num in range(epoch_length):
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        '\nAverage number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True,
                                           overlap_thresh=non_max_suppression_overlap_threshold,
                                           max_boxes=non_max_suppression_max_boxes)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1,
                               [('rpn_cls', np.mean(losses[:iter_num + 1, 0])),
                                ('rpn_regr', np.mean(losses[:iter_num + 1, 1])),
                                ('detector_cls', np.mean(losses[:iter_num + 1, 2])),
                                ('detector_regr', np.mean(losses[:iter_num + 1, 3]))])
            except Exception as e:
                print('Exception during training: {}'.format(e))
                continue

        # Calculate losses after the specified number of iterations
        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])

        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
        rpn_accuracy_for_epoch = []

        if C.verbose:
            print('[INFO TRAINING]')
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {}'.format(loss_class_cls))
            print('Loss Detector regression: {}'.format(loss_class_regr))
            print('Elapsed time: {}'.format(time.time() - start_time))
            print("Best loss for training: {0:.3f}".format(best_loss_training))

        curr_total_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        val_start_time = time.time()
        write_log(tensorboard_callback, train_names,
                  [loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_total_loss, class_acc],
                  epoch_num)

        if curr_total_loss < best_loss_training:
            model_path = C.model_path[:-5] + "_training.hdf5"
            if C.verbose:
                print('Total training loss decreased from {0:.3f} to {1:.3f}, saving weights to {2}'
                      .format(best_loss_training, curr_total_loss, model_path))
            best_loss_training = curr_total_loss
            model_all.save_weights(model_path)

        #############
        # VALIDATION
        #############
        if (epoch_num + 1) % validation_interval != 0:
            continue

        progbar = generic_utils.Progbar(validation_epoch_length)
        for iter_num in range(validation_epoch_length):
            try:
                X, Y, img_data = next(data_gen_val)

                loss_rpn = model_rpn.test_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True,
                                           overlap_thresh=non_max_suppression_overlap_threshold,
                                           max_boxes=non_max_suppression_max_boxes)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2,
                                                                replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.test_on_batch([X, X2[:, sel_samples, :]],
                                                            [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses_val[iter_num, 0] = loss_rpn[1]
                losses_val[iter_num, 1] = loss_rpn[2]

                losses_val[iter_num, 2] = loss_class[1]
                losses_val[iter_num, 3] = loss_class[2]
                losses_val[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1, [('rpn_cls', np.mean(losses_val[:iter_num + 1, 0])),
                                              ('rpn_regr', np.mean(losses_val[:iter_num + 1, 1])),
                                              ('detector_cls', np.mean(losses_val[:iter_num + 1, 2])),
                                              ('detector_regr', np.mean(losses_val[:iter_num + 1, 3]))])
            except Exception as e:
                print('Exception during validation: {}'.format(e))
                continue

        # Computer aggregated losses
        loss_rpn_cls = np.mean(losses_val[:, 0])
        loss_rpn_regr = np.mean(losses_val[:, 1])
        loss_class_cls = np.mean(losses_val[:, 2])
        loss_class_regr = np.mean(losses_val[:, 3])
        class_acc = np.mean(losses[:, 4])

        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
        rpn_accuracy_for_epoch = []
        curr_total_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

        write_log(tensorboard_callback, val_names,
                  [loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_total_loss, class_acc],
                  epoch_num)

        if C.verbose:
            print('[INFO VALIDATION]')
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {}'.format(loss_class_cls))
            print('Loss Detector regression: {}'.format(loss_class_regr))
            print("Current validation loss: {0:.3f}, Best validation loss: {1:.3f} at epoch: {2}"
                  .format(curr_total_loss, best_total_loss_validation, best_loss_epoch))
            print('Elapsed time: {}'.format(time.time() - val_start_time))

        if curr_total_loss < best_total_loss_validation:
            if C.verbose:
                print('Total validation loss decreased from {0:.3f} to {1:.3f}, saving weights to {2}'
                      .format(best_total_loss_validation, curr_total_loss, C.model_path))
            best_total_loss_validation = curr_total_loss
            best_loss_rpn_cls = loss_rpn_cls
            best_loss_rpn_regr = loss_rpn_regr
            best_loss_class_cls = loss_class_cls
            best_loss_class_regr = loss_class_regr
            best_class_acc = class_acc
            best_loss_epoch = epoch_num
            model_all.save_weights(C.model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += validation_interval

        if epochs_without_improvement > early_stopping:
            print("Early stopping training after {0} epochs without improvement on validation set"
                  .format(epochs_without_improvement))
            break

        if epochs_without_improvement > learning_rate_reduction_patience:
            current_learning_rate = K.get_value(model_classifier.optimizer.lr)
            new_learning_rate = current_learning_rate * learning_rate_reduction_factor
            print("Not improved validation accuracy for {0} epochs. Reducing learning rate from {1} to {2}".format(
                learning_rate_reduction_patience, current_learning_rate, new_learning_rate))
            K.set_value(model_classifier.optimizer.lr, new_learning_rate)
            K.set_value(model_rpn.optimizer.lr, new_learning_rate)
            K.set_value(model_all.optimizer.lr, new_learning_rate)

    end_time = time.time()
    execution_time_in_seconds = round(end_time - start_time)
    print("Execution time: {0:.1f}s".format(end_time - start_time))

    notification_message = "Training on {0} dataset with model {1} and configuration {2} finished. " \
                           "Val. accuracy: {3:0.5f}%".format("muscima_pp",
                                                             model_name,
                                                             configuration_name,
                                                             best_class_acc * 100)
    TelegramNotifier.send_message_via_telegram(notification_message)

    today = "{0:02d}.{1:02d}.{2}".format(start_of_training.day, start_of_training.month, start_of_training.year)
    GoogleSpreadsheetReporter.append_result_to_spreadsheet(dataset_size=len(all_images),
                                                           model_name=model_name,
                                                           configuration_name=configuration_name,
                                                           data_augmentation="",
                                                           early_stopping=early_stopping,
                                                           reduction_patience=learning_rate_reduction_patience,
                                                           learning_rate_reduction_factor=learning_rate_reduction_factor,
                                                           optimizer="Adadelta",
                                                           initial_learning_rate=1.0,
                                                           non_max_suppression_overlap_threshold=non_max_suppression_overlap_threshold,
                                                           non_max_suppression_max_boxes=non_max_suppression_max_boxes,
                                                           validation_accuracy=best_class_acc,
                                                           validation_total_loss=best_total_loss_validation,
                                                           best_loss_rpn_cls=best_loss_rpn_cls,
                                                           best_loss_rpn_regr=best_loss_rpn_regr,
                                                           best_loss_class_cls=best_loss_class_cls,
                                                           best_loss_class_regr=best_loss_class_regr,
                                                           date=today,
                                                           datasets="muscima_pp",
                                                           execution_time_in_seconds=execution_time_in_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", type=str, dest="train_path", help="Path to training data.", default="data")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="The model used for training the network. Currently one of [vgg, resnet50, simple_resnet]")
    parser.add_argument("--recreate_dataset_directory", dest="delete_and_recreate_dataset_directory",
                        help="Deletes and recreates the dataset directory",
                        action="store_true", default=False)
    parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=2000)
    parser.add_argument("--configuration_name", type=str, dest="config_name",
                        help="Name of the hyperparameter configuration to use", default="many_anchor_box_scales")
    parser.add_argument("--config_filename", type=str, dest="config_filename",
                        help="Location to store all the metadata related to the training (to be used when testing).",
                        default="config.pickle")
    parser.add_argument("--output_weight_path", type=str, dest="output_weight_path", help="Output path for weights.",
                        default='model_frcnn.hdf5')
    parser.add_argument("--input_weight_path", type=str, dest="input_weight_path",
                        help="Input path for weights. If not specified, will try to load default weights provided by keras.")
    parser.add_argument("--non_max_suppression_max_boxes",
                        type=int,
                        dest="non_max_suppression_max_boxes",
                        help="Number of boxes to keep from non-maximum suppressions", default=300)
    parser.add_argument("--non_max_suppression_overlap_threshold",
                        type=float,
                        dest="non_max_suppression_overlap_threshold",
                        help="Overlap threshold for non-maximum suppressions (between 0.0 - 1.0)", default=0.7)

    options, unparsed = parser.parse_known_args()

    dataset_directory = options.train_path
    model_name = options.model_name
    configuration_name = options.config_name
    output_weight_path = options.output_weight_path
    configuration_filename = options.config_filename
    number_of_epochs = options.num_epochs
    input_weight_path = options.input_weight_path
    early_stopping = 20
    learning_rate_reduction_patience = 8
    learning_rate_reduction_factor = 0.5
    non_max_suppression_overlap_threshold = options.non_max_suppression_overlap_threshold
    non_max_suppression_max_boxes = int(options.non_max_suppression_max_boxes)

    train_model(dataset_directory, model_name, options.delete_and_recreate_dataset_directory, configuration_name,
                output_weight_path, configuration_filename, number_of_epochs, early_stopping,
                learning_rate_reduction_patience, learning_rate_reduction_factor, non_max_suppression_overlap_threshold,
                non_max_suppression_max_boxes, input_weight_path)
