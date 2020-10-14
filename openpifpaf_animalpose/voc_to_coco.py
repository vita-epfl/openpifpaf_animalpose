"""
Prepare the dataset to be converted from VOC 2 COCO
This code bridges with the repository https://github.com/yukkyo/voc2coco
that requires annotation lists and train val split and annotation directories

The original datasets include images and annotations divided in 2 parts.
- part1 consists of images from VOC2012 and custom annotations
- part 2 consists of custom images and annotations
"""

import os
import glob
import argparse
import time
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

CATEGORIES = ('cat', 'cow' 'dog', 'sheep', 'horse')


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data/animalpose',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data/animalpose/annotations',
                        help='where to save xml annotations and output json ')
    args = parser.parse_args()
    return args


class VocToCoco:

    json_file = {}
    map_cat = {cat: el+1 for el, cat in enumerate(CATEGORIES)}

    def __init__(self, dir_dataset, dir_out, args):
        """
        :param dir_dataset: Original dataset directory
        :param dir_out: Processed dataset directory
        """
        self.dir_dataset = dir_dataset
        self.dir_out = dir_out

    def process(self):
        # for phase, im_paths in self.splits.items():  # Train and Val
        cnt_images = 0
        cnt_instances = 0
        cnt_kps = [0] * 66
        self.initiate_json()  # Initiate json file at each phase
        cnt = 0
        for folder in glob.glob(os.path.join(self.dir_dataset, 'part*')):
            dir_ann = os.path.join(folder, 'annotations')
            for cat in CATEGORIES:
                paths = glob.glob(os.path.join(dir_ann, cat + os.sep + '*.xml'))
                for xml_path in paths:
                    im_path, im_id = self._extract_filename(xml_path)
                    im_size = self._process_image(im_path, im_id, cat)
                    cnt_images += 1
                    self._process_annotation(xml_path, im_size, im_id)
                    cnt_instances += 1

    def _process_image(self, im_path, im_id, cat):
        """Update image field in json file"""
        file_name = os.path.split(im_path)[1]
        im_name = os.path.splitext(file_name)[0]

        # Assert for different categorization

        im = Image.open(im_path)
        width, height = im.size

        self.json_file["images"].append({
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height})
        return (width, height), im_name, im_id

    def _process_annotation(self, xml_path, im_id, im_size):
        """Process single instance"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        box_obj = root.findall('visible_bounds')
        assert len(box_obj) <= 1, "multiple elements in a single annotation file not supported"

        x_min = round(float((box_obj[0].attrib['xmin']))) - 1

        width = round(float(box_obj[0].attrib['width']))
        height = round(float(box_obj[0].attrib['height']))
        try:
            y_min = round(float(box_obj[0].attrib['ymin'])) - 1
        except KeyError:
            y_min = round(float(box_obj[0].attrib['xmax'])) - 1
        box = [x_min, y_min, width, height]

        kp_obj = root.findall('keypoints')
        assert len(kp_obj) <= 1, "multiple elements in a single annotation file not supported"
        kps_list = kp_obj[0].findall('keypoint')
        try:
            assert 17 <= len(kps_list) <= 20, "number of keypoints not recognized"
        except AssertionError:
            aa = 5

        kps, num = self._process_keypoint(kps_list)

        self.json_file["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': im_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})
        return None


    def _process_keypoint(self, kps_list):
        for kp in kps_list:
            aa = 5
        return None, None


    def initiate_json(self):
        """
        Initiate Json for training and val phase
        """
        self.json_file["info"] = dict(url="https://github.com/vita-epfl/openpifpaf",
                                      date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                                      description="Conversion of AnimalPose dataset into MS-COCO format")

        self.json_file["categories"] = [dict(name='car',
                                             id=1,
                                             skeleton=[],
                                             supercategory='car',
                                             keypoints=[])]
        self.json_file["images"] = []
        self.json_file["annotations"] = []


    def _extract_filename(self, xml_path):
        """
        Manage all the differences between the 2 annotated parts and all the exceptions of Part 2
        """
        path = os.path.normpath(xml_path)
        sub_dirs = path.split(os.sep)
        cat = sub_dirs[-2]
        folder = sub_dirs[-4]
        im_dir = os.path.join(*sub_dirs[:-3], 'images')
        assert folder in ('part1', 'part2')

        if folder == 'part1':
            splits = os.path.splitext(sub_dirs[-1])[0].split(sep='_')
            basename = splits[0] + '_' + splits[1]
            ext = '.jpg'
            im_path = os.path.join(im_dir, basename + ext)
            im_id = int(splits[1])
        else:
            basename = os.path.splitext(sub_dirs[-1])[0]
            num = int(basename[2:])
            im_id = int(str(self.map_cat[cat]) + basename[2:])
            if cat == 'sheep' and num == 65:
                ext = '.png'
            elif cat == 'sheep' and (num <= 97 or num >= 190):
                ext = '.jpg'
            else:
                ext = '.jpeg'
            im_path = os.path.join(im_dir, cat, basename + ext)

        return im_path, im_id

def get_coco_annotation_from_obj(obj, label2id):

    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def main():
    args = cli()
    voc_coco = VocToCoco(args.dir_data, args.dir_out, args)
    voc_coco.process()


if __name__ == "__main__":
    main()
