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


def _augment_xml(path_xml, dir_im, dir_out):
    """Add the image path in the xml and save a copy of it"""
    tree = ET.parse(path_xml)
    root = tree.getroot()
    for el in root.iter():
        print(el)
    print(root[0])
    aa = 5


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
            dir_im, dir_ann = os.path.join(folder, 'images'), os.path.join(folder, 'annotations')
            im_ext = 'jpg' if folder[-5:] == 'part1' else 'jpeg'  # voc official images or custom images
            for cat in CATEGORIES:
                paths = glob.glob(os.path.join(dir_ann, cat + '/*.xml'))
                for xml_path in paths:
                    basename = os.path.basename(xml_path).split(sep='.')[0]
                    im_path = os.path.join(dir_im, cat, basename + '.' + im_ext)
                    im_info = self._process_image(im_path, cat)
                    self._process_annotation(xml_path)

    def _process_image(self, im_path, cat):
        """Update image field in json file"""
        file_name = os.path.split(im_path)[1]
        im_name = os.path.splitext(file_name)[0]
        try:
            im_id = int(im_name.split(sep='_')[1])
        except IndexError:
            im_id = int(str(self.map_cat[cat]) + (im_name[2:]))
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


    def _process_annotation(self, data, xml_path, im_id, im_size, cnt_kps):
        """Process single instance"""
        all_kps = np.array(data)  # [#, x, y]

        # Enlarge box
        box_tight = [np.min(all_kps[:, 1]), np.min(all_kps[:, 2]), np.max(all_kps[:, 1]), np.max(all_kps[:, 2])]
        w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
        x_o = max(box_tight[0] - (w / 10), 0)
        y_o = max(box_tight[1] - (h / 10), 0)
        x_i = min(x_o + (w / 4) + w, im_size[0])
        y_i = min(y_o + (h / 4) + h, im_size[1])
        box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

        kps, num = self._transform_keypoints(all_kps)
        txt_id = os.path.splitext(txt_path.split(sep='_')[-1])[0]
        car_id = int(str(im_id) + str(int(txt_id)))  # include at the end of the number the specific annotation id
        self.json_file["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': car_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})
        # Stats
        for num in data[0]:
            cnt_kps[num] += 1
        return cnt_kps


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


def main():
    args = cli()
    voc_coco = VocToCoco(args.dir_data, args.dir_out, args)
    voc_coco.process()


if __name__ == "__main__":
    main()
