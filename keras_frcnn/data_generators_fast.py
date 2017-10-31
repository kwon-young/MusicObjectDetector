import random
import threading
from typing import List

import cv2
import numpy
import numpy as np
from tqdm import tqdm

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration
from keras_frcnn.SampleSelector import SampleSelector
from keras_frcnn.py_faster_rcnn.utils.bbox import bbox_overlaps
from . import data_augment


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


def calc_rpn_slow(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture

    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        # if img_data['bboxes'][bbox_num]['class'] != 'bg':

                        # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                        if curr_iou > best_iou_for_bbox[bbox_num]:
                            best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                            best_iou_for_bbox[bbox_num] = curr_iou
                            best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                            best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                        # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                        if curr_iou > C.rpn_max_overlap:
                            bbox_type = 'pos'
                            num_anchors_for_bbox[bbox_num] += 1
                            # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                            if curr_iou > best_iou_for_loc:
                                best_iou_for_loc = curr_iou
                                best_regr = (tx, ty, tw, th)

                        # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                        if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                            # gray zone between neg and pos
                            if bbox_type != 'pos':
                                bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


# get fast anchor tatget
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def get_anchors(C, width, height, resized_width, resized_height, img_length_calc_function):
    downscale = float(C.rpn_stride)
    anchor_scales = np.asarray(C.anchor_box_scales) / downscale
    anchor_ratios = [0.5, 1, 2]
    num_anchors = len(anchor_scales) * len(anchor_ratios)
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)
    base_anchor = np.array([1, 1, downscale, downscale]) - 1
    ratio_anchors = _ratio_enum(base_anchor, anchor_ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales)
                         for i in range(ratio_anchors.shape[0])])
    shift_x = np.arange(output_width) * downscale
    shift_y = np.arange(output_height) * downscale
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    all_anchors = np.expand_dims(anchors, axis=0) + np.expand_dims(shifts, axis=0).transpose((1, 0, 2))
    all_anchors = np.reshape(all_anchors, (-1, 4))
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < resized_width) &  # width
        (all_anchors[:, 3] < resized_height)  # height
    )[0]
    all_anchors = np.concatenate((all_anchors, np.zeros((all_anchors.shape[0], 1))), axis=-1)
    all_anchors[inds_inside, -1] = 1
    all_anchors = np.reshape(all_anchors, (output_height, output_width, anchors.shape[0], 5))
    return all_anchors


def calc_rpn_fast(C, img_data, width, height, resized_width, resized_height, anchors):
    num_regions = 256

    num_bboxes = len(img_data['bboxes'])

    # calculate the output map size based on the network architecture

    all_anchors = np.reshape(anchors, (-1, 5))
    valid_idxs = np.where(all_anchors[:, -1] == 1)[0]

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((all_anchors.shape[0], 1))
    y_is_box_valid = np.zeros((all_anchors.shape[0], 1))
    y_rpn_regr = np.zeros((all_anchors.shape[0], 4))

    valid_anchors = all_anchors[valid_idxs, :]
    valid_rpn_overlap = np.zeros((valid_anchors.shape[0], 1))
    valid_is_box_valid = np.zeros((valid_anchors.shape[0], 1))
    valid_rpn_regr = np.zeros((valid_anchors.shape[0], 4))

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    valid_overlap = np.zeros((valid_anchors.shape[0], num_bboxes))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['y1'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['x2'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
    # for i in range(valid_anchors.shape[0]):
    # 	valid_overlap[i,bbox_num] = iou(gta[bbox_num, :], valid_anchors[i,:4])
    valid_overlap = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),
                                  np.ascontiguousarray(gta, dtype=np.float))
    # find every anchor close to which bbox
    argmax_overlaps = valid_overlap.argmax(axis=1)
    max_overlaps = valid_overlap[np.arange(len(valid_idxs)), argmax_overlaps]
    # find which anchor closest to every bbox
    gt_argmax_overlaps = valid_overlap.argmax(axis=0)
    gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
    gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
    valid_rpn_overlap[gt_argmax_overlaps] = 1
    valid_rpn_overlap[max_overlaps > C.rpn_max_overlap] = 1
    # get positives labels
    fg_inds = np.where(valid_rpn_overlap == 1)[0]
    if len(fg_inds) > num_regions / 2:
        able_inds = np.random.choice(fg_inds, size=num_regions / 2, replace=False)
        valid_is_box_valid[able_inds] = 1
    else:
        valid_is_box_valid[fg_inds] = 1
    # get positives regress
    fg_inds = np.where(valid_is_box_valid == 1)[0]
    for i in range(len(fg_inds)):
        anchor_box = valid_anchors[fg_inds[i], :4]
        gt_box = gta[argmax_overlaps[fg_inds[i]], :]
        cx = (gt_box[0] + gt_box[2]) / 2.0
        cy = (gt_box[1] + gt_box[3]) / 2.0
        cxa = (anchor_box[0] + anchor_box[2]) / 2.0
        cya = (anchor_box[1] + anchor_box[3]) / 2.0

        tx = (cx - cxa) / (anchor_box[2] - anchor_box[0])
        ty = (cy - cya) / (anchor_box[3] - anchor_box[1])
        tw = np.log((gt_box[2] - gt_box[0]) / (anchor_box[2] - anchor_box[0]))
        th = np.log((gt_box[3] - gt_box[1]) / (anchor_box[3] - anchor_box[1]))
        valid_rpn_regr[fg_inds[i], :] = [tx, ty, tw, th]

    bg_inds = np.where(valid_rpn_overlap == 0)[0]
    if len(bg_inds) > num_regions - np.sum(valid_is_box_valid == 1):
        able_inds = np.random.choice(bg_inds, size=num_regions - np.sum(valid_is_box_valid == 1), replace=False)
        valid_is_box_valid[able_inds] = 1
    else:
        valid_is_box_valid[bg_inds] = 1

    # transform to the original overlap and validbox
    y_rpn_overlap[valid_idxs, :] = valid_rpn_overlap
    y_is_box_valid[valid_idxs, :] = valid_is_box_valid
    y_rpn_regr[valid_idxs, :] = valid_rpn_regr

    y_rpn_overlap = np.reshape(y_rpn_overlap, (anchors.shape[0], anchors.shape[1], anchors.shape[2])).transpose(
        (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.reshape(y_is_box_valid, (anchors.shape[0], anchors.shape[1], anchors.shape[2])).transpose(
        (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.reshape(y_rpn_regr, (anchors.shape[0], anchors.shape[1], anchors.shape[2] * 4)).transpose((2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_anchor_gt(all_img_data: List, classes_count: dict, C: FasterRcnnConfiguration, img_length_calc_function,
                  mode: str = 'train'):
    image_anchors = {}
    for img_data in tqdm(all_img_data, desc="Pre-computing anchors for resized images"):
        (width, height) = (img_data['width'], img_data['height'])
        (resized_width, resized_height) = get_new_img_size(width, height, C.resize_smallest_side_of_image_to)
        anchors = get_anchors(C, width, height, resized_width, resized_height, img_length_calc_function)
        image_anchors[img_data['filepath']] = anchors

    sample_selector = SampleSelector(classes_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:
                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.resize_smallest_side_of_image_to)

                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    # start_time = time.time()
                    #y_rpn_cls, y_rpn_regr = calc_rpn_slow(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)

                    anchors = image_anchors[img_data['filepath']]
                    y_rpn_cls, y_rpn_regr = calc_rpn_fast(C, img_data_aug, width, height, resized_width, resized_height,
                                                            anchors)
                    # if not np.array_equal(y_rpn_cls, y_rpn_cls2):
                    #     print("Arrays 1 not equal - this might be an error")
                    # if not np.array_equal(y_rpn_regr, y_rpn_regr2):
                    #     print("Arrays 2 not equal - this might be an error")

                    # duration = time.time() - start_time
                    # print (duration)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue
