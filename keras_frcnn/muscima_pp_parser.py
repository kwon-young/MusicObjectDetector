from typing import Tuple

import cv2
import numpy as np
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator


def get_data(image_path, raw_data_directory: str) -> Tuple[list, dict, dict]:
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    visualise = True

    image_generator = MuscimaPlusPlusImageGenerator()
    all_xml_files = image_generator.get_all_xml_file_paths(raw_data_directory)[0:1]
    crop_objects = image_generator.load_crop_objects_from_xml_files(all_xml_files)

    print('Parsing annotation files')

    filename = image_path
    img = cv2.imread(filename)

    for crop_object in crop_objects:
        class_name = crop_object.clsname
        (x1, y1, x2, y2) = crop_object.bounding_box

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping)

        if filename not in all_imgs:
            all_imgs[filename] = {}

            (rows, cols) = img.shape[:2]
            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            if np.random.randint(0, 6) > 0:
                all_imgs[filename]['imageset'] = 'trainval'
            else:
                all_imgs[filename]['imageset'] = 'test'

        all_imgs[filename]['bboxes'].append(
            {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

        if visualise:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            crop_object.mask

    if visualise:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    return all_data, classes_count, class_mapping


if __name__ == "__main__":
    # get_data("annotationsample.txt", "../data/muscima_pp_raw")
    get_data(
        "C:\\Users\\Alex\\Repositories\\CVC-MUSCIMA\\CVCMUSCIMA_SR\\CvcMuscima-Distortions\\ideal\\w-10\\image\\p010.png",
        #"C:\\Users\\Alex\\Repositories\\MusicObjectDetector\\data\\CVCMUSCIMA_WI\\CVCMUSCIMA_WI\\PNG_GT_Gray\\w-01\\p010.png",
        "../data/muscima_pp_raw")
