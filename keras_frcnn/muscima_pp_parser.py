import os
import re
from typing import Tuple

import cv2
import numpy as np
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm


def get_data(muscima_image_directory, muscima_pp_raw_data_directory: str, visualize=False) -> Tuple[list, dict, dict]:
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    image_generator = MuscimaPlusPlusImageGenerator()
    all_xml_files = image_generator.get_all_xml_file_paths(muscima_pp_raw_data_directory)

    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)
        doc = crop_objects[0].doc
        result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
        writer = result.group("writer")
        page = result.group("page")

        # image_path = os.path.join(image_directory, "w-{0}".format(writer), "p0{0}.png".format(page))
        image_path = os.path.join(muscima_image_directory, "CvcMuscima-Distortions", "ideal", "w-{0}".format(writer), "image", "p0{0}.png".format(page))

        img = cv2.imread(image_path)
        for crop_object in crop_objects:
            class_name = crop_object.clsname
            (top, left, bottom, right) = crop_object.bounding_box
            x1, y1, x2, y2 = left, top, right, bottom

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            if image_path not in all_imgs:
                all_imgs[image_path] = {}

                (rows, cols) = img.shape[:2]
                all_imgs[image_path]['filepath'] = image_path
                all_imgs[image_path]['width'] = cols
                all_imgs[image_path]['height'] = rows
                all_imgs[image_path]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[image_path]['imageset'] = 'trainval'
                else:
                    all_imgs[image_path]['imageset'] = 'test'

            all_imgs[image_path]['bboxes'].append({'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
            if visualize:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

        if visualize:
            cv2.imshow('img', img)
            cv2.waitKey(0)

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    return all_data, classes_count, class_mapping


if __name__ == "__main__":
    all_data, classes_count, class_mapping = get_data(
        # "C:\\Users\\Alex\\Repositories\\CVC-MUSCIMA\\CVCMUSCIMA_WI\\CVCMUSCIMA_WI\\PNG_GT_Gray\\",
        "C:\\Users\\\Alex\\Repositories\\CVC-MUSCIMA\\CVCMUSCIMA_SR",
        "../data/muscima_pp_raw", False)

    number_of_bounding_boxes = sum(classes_count.values())
    print("Found {0} samples with {1} bounding-boxes belonging to {2} classes".format(len(all_data),
                                                                                      number_of_bounding_boxes,
                                                                                      len(classes_count)))
