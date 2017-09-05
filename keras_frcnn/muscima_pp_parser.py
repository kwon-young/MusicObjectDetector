import os
import re
from glob import glob
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageDraw
from muscima.cropobject import CropObject
from omrdatasettools.converters.ImageInverter import ImageInverter
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
        image_path = os.path.join(muscima_image_directory, "CvcMuscima-Distortions", "ideal", "w-{0}".format(writer),
                                  "image", "p0{0}.png".format(page))

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


def cut_images(muscima_image_directory: str, staff_vertical_positions_file: str, output_path: str,
               muscima_pp_raw_data_directory: str, ):
    image_paths = [y for x in os.walk(muscima_image_directory) for y in glob(os.path.join(x[0], '*.png'))]
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path,"Annotations.txt"), "w") as annotations_file:

        image_generator = MuscimaPlusPlusImageGenerator()
        all_xml_files = image_generator.get_all_xml_file_paths(muscima_pp_raw_data_directory)

        crop_object_annotations: List[Tuple[str, str, List[CropObject]]] = []

        for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
            crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)
            doc = crop_objects[0].doc
            result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
            writer = result.group("writer")
            page = result.group("page")
            crop_object_annotations.append(('w-' + writer, 'p' + page.zfill(3), crop_objects))

        with open(staff_vertical_positions_file) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        annotations = [x.strip().split(';') for x in content]

        images_to_cut: List[Tuple[str, str, str, str]] = []

        for image_path in image_paths:
            result = re.match(r".*(?P<writer>w-\d+).*(?P<page>p\d+).png", image_path)
            writer = result.group("writer")
            page = result.group("page")
            coordinates = None
            for annotation in annotations:
                if annotation[0] == writer and annotation[1] == page:
                    coordinates = annotation[2]
                    break

            if coordinates is not None:
                images_to_cut.append((image_path, writer, page, coordinates))

        for image_to_cut in tqdm(images_to_cut, desc="Cutting images"):
            path, writer, page, coordinates = image_to_cut
            staff_line_pairs = coordinates.split(',')
            image = Image.open(path, "r")
            width = image.width
            crop_objects_of_image: List[CropObject] = None
            for crop_object_annotation in crop_object_annotations:
                if writer == crop_object_annotation[0] and page == crop_object_annotation[1]:
                    crop_objects_of_image = crop_object_annotation[2]
                    break

            i = 1
            for pair in staff_line_pairs:
                y_top, y_bottom = map(int, pair.split(':'))
                previous_width = 0
                overlap = 100
                for crop_width in range(500, 3501, 500):
                    objects_of_cropped_image: List[Tuple[str, Tuple[int, int, int, int]]] = []
                    if crop_width > width:
                        crop_width = width
                    image_crop_bounding_box = (previous_width, y_top, crop_width, y_bottom)

                    file_name = "{0}_{1}_{2}.png".format(writer, page, i)

                    for crop_object in crop_objects_of_image:
                        if bounding_box_in(image_crop_bounding_box, crop_object.bounding_box):
                            top, left, bottom, right = crop_object.bounding_box
                            translated_bounding_box = (
                                top - y_top, left - previous_width, bottom - y_top, right - previous_width)
                            trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                            objects_of_cropped_image.append((crop_object.clsname, translated_bounding_box))
                            annotations_file.write("{0},{1},{2},{3},{4},{5}\n".format(file_name,
                                                                                       trans_left,
                                                                                       trans_top,
                                                                                       trans_right,
                                                                                       trans_bottom,
                                                                                       crop_object.clsname))

                    cropped_image = image.crop(image_crop_bounding_box).convert('RGB')

                    #draw_bounding_boxes(cropped_image, objects_of_cropped_image)

                    output_file = os.path.join(output_path, file_name)
                    cropped_image.save(output_file)
                    previous_width = crop_width - overlap
                    i += 1


def bounding_box_in(image_crop_bounding_box: Tuple[int, int, int, int],
                    crop_object_bounding_box: Tuple[int, int, int, int]) -> bool:
    image_left, image_top, image_right, image_bottom = image_crop_bounding_box
    object_top, object_left, object_bottom, object_right = crop_object_bounding_box
    if object_left >= image_left and object_right <= image_right \
            and object_top >= image_top and object_bottom <= image_bottom:
        return True
    return False


def draw_bounding_boxes(cropped_image, objects_of_cropped_image):
    draw = ImageDraw.Draw(cropped_image)
    red = (255, 0, 0)
    for object_in_image in objects_of_cropped_image:
        top, left, bottom, right = object_in_image[1]
        draw.rectangle((left, top, right, bottom), fill=None, outline=red)


if __name__ == "__main__":
    # inverter = ImageInverter()
    # inverter.invert_images("../data/", "*.png")

    cut_images("../data/cvcmuscima_staff_removal", "../data/Staff-Vertical-Positions.txt",
               "../data/muscima_pp_cropped_images", "../data/muscima_pp_raw")

    exit()
    all_data, classes_count, class_mapping = get_data(
            # "C:\\Users\\Alex\\Repositories\\CVC-MUSCIMA\\CVCMUSCIMA_WI\\CVCMUSCIMA_WI\\PNG_GT_Gray\\",
            "C:\\Users\\\Alex\\Repositories\\CVC-MUSCIMA\\CVCMUSCIMA_SR",
            "../data/muscima_pp_raw", False)

    number_of_bounding_boxes = sum(classes_count.values())
    print("Found {0} samples with {1} bounding-boxes belonging to {2} classes".format(len(all_data),
                                                                                      number_of_bounding_boxes,
                                                                                      len(classes_count)))
