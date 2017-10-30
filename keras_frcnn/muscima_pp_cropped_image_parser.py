import os
from typing import List, Tuple

import cv2
import numpy as np


def get_data(muscima_pp_cropped_images_directory: str, visualise: bool = False) -> Tuple[List[dict], dict, dict]:
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    annotation_file = os.path.join(muscima_pp_cropped_images_directory, "Annotations.txt")

    with open(annotation_file, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, left, top, right, bottom, class_name) = line_split
            filename = os.path.join(muscima_pp_cropped_images_directory, filename)
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print("Found class name with special name bg. Will be treated as a background region (this is "
                          "usually for hard negative mining).")
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'train'
                else:
                    all_imgs[filename]['imageset'] = 'val'

            all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': left, 'x2': right, 'y1': top, 'y2': bottom})

            if visualise:
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255))
                cv2.imshow('img', img)
                cv2.waitKey(0)

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


if __name__ == "__main__":
    all_data, classes_count, class_mapping = get_data("../data/muscima_pp_cropped_images", False)

    number_of_bounding_boxes = sum(classes_count.values())
    print("Found {0} samples with {1} bounding-boxes belonging to {2} classes".format(len(all_data),
                                                                                      number_of_bounding_boxes,
                                                                                      len(classes_count)))
