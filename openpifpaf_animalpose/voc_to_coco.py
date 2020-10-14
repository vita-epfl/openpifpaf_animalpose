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

from .constants import CATEGORIES, ANIMAl_KEYPOINTS, ALTERNATIVE_NAMES

def dataset_mappings():
    """Map the two names to 0 n-1"""
    map_n = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(ANIMAl_KEYPOINTS, range(len(ANIMAl_KEYPOINTS))):
        map_n[i] = j
    for i, j in zip(ALTERNATIVE_NAMES, range(len(ALTERNATIVE_NAMES))):
        map_n[i] = j
    map_vis = {'1': 2, '0': 0}
    return map_n, map_vis


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
    map_names, map_vis = dataset_mappings()
    n_kps = len(ANIMAl_KEYPOINTS)
    cnt_kps = [0] * n_kps

    def __init__(self, dir_dataset, dir_out):
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
        self.cnt_kps = [0] * len(ANIMAl_KEYPOINTS)
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
        """Extract single keypoint from XML"""
        cnt = 0
        kps_out = np.zeros((self.n_kps, 3))
        for kp in kps_list:
            n = self.map_names[kp.attrib['name']]
            if n < 100:
                kps_out[n, 0] = float(kp.attrib['x'])
                kps_out[n, 1] = float(kp.attrib['y'])
                kps_out[n, 2] = self.map_vis[kp.attrib['visible']]
                cnt += 1
                self.cnt_kps[n] += 1
        kps_out = list(kps_out.reshape((-1,)))
        return kps_out, cnt

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

def main():
    args = cli()
    voc_coco = VocToCoco(args.dir_data, args.dir_out)
    voc_coco.process()


if __name__ == "__main__":
    main()
