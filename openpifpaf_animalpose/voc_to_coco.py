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
import json
import random
from collections import defaultdict
import shutil
from shutil import copyfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .constants import _CATEGORIES, ANIMAL_KEYPOINTS, ALTERNATIVE_NAMES, ANIMAL_SKELETON


def dataset_mappings():
    """Map the two names to 0 n-1"""
    map_n = defaultdict(lambda: 100)  # map to 100 the keypoints not used
    for i, j in zip(ANIMAL_KEYPOINTS, range(len(ANIMAL_KEYPOINTS))):
        map_n[i] = j
    for i, j in zip(ALTERNATIVE_NAMES, range(len(ALTERNATIVE_NAMES))):
        map_n[i] = j
    return map_n


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_data', default='data/animalpose',
                        help='dataset directory')
    parser.add_argument('--dir_out', default='data/animalpose',
                        help='where to save xml annotations and output json ')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    parser.add_argument('--split_images', action='store_true',
                        help='Whether to copy images into train val split folder')
    parser.add_argument('--histogram', action='store_true',
                        help='Whether to show keypoints histogram')
    args = parser.parse_args()
    return args


class VocToCoco:

    json_file = {}
    map_cat = {cat: el+1 for el, cat in enumerate(_CATEGORIES)}
    map_names = dataset_mappings()
    n_kps = len(ANIMAL_KEYPOINTS)
    cnt_kps = [0] * n_kps

    def __init__(self, dir_dataset, dir_out, args):
        """
        :param dir_dataset: Original dataset directory
        :param dir_out: Processed dataset directory
        """
        self.dir_dataset = dir_dataset
        self.dir_out_im = os.path.join(dir_out, 'images')
        self.dir_out_ann = os.path.join(dir_out, 'annotations')
        assert os.path.isdir(self.dir_out_im), "Images output directory not found"
        assert os.path.isdir(self.dir_out_ann), "Annotations directory not found"
        self.sample = args.sample
        self.split_images = args.split_images
        self.histogram = args.histogram

    def process(self):
        splits = self._split_train_val()
        all_xml_paths = []
        for phase in ('train', 'val'):
            metadata = splits[phase]
            if self.sample:
                metadata = metadata[:50]
            if self.split_images:
                make_new_directory(os.path.join(self.dir_out_im, phase))
            cnt_images = 0
            cnt_instances = 0
            self.cnt_kps = [0] * len(ANIMAL_KEYPOINTS)
            self.initiate_json()  # Initiate json file at each phase

            for im_meta in metadata:
                self._process_image(im_meta[0], im_meta[1])
                cnt_images += 1
                for xml_path in self._find_annotations(im_meta):
                    self._process_annotation(xml_path, im_meta[1])
                    cnt_instances += 1
                    all_xml_paths.append(xml_path)

                # Split the image in a new folder
                if self.split_images:
                    dst = os.path.join(self.dir_out_im, phase, os.path.split(im_meta[0])[-1])
                    copyfile(im_meta[0], dst)

                # Count
                if (cnt_images % 1000) == 0:
                    text = ' and copied to new directory' if self.split_images else ''
                    print(f'Parsed {cnt_images} images' + text)

            # Save
            name = 'animal_keypoints_' + str(self.n_kps) + '_'
            if self.sample:
                name = name + 'sample_'

            path_json = os.path.join(self.dir_out_ann, name + phase + '.json')
            with open(path_json, 'w') as outfile:
                json.dump(self.json_file, outfile)
            print(f'Phase:{phase}')
            print(f'Average number of keypoints labelled: {sum(self.cnt_kps) / cnt_instances:.1f} / {self.n_kps}')
            print(f'Saved {cnt_instances} instances over {cnt_images} images ')
            print(f'JSON PATH:  {path_json}')
            if self.histogram:
                histogram(self.cnt_kps)

    def _process_image(self, im_path, im_id):
        """Update image field in json file"""
        file_name = os.path.split(im_path)[1]
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

    def _process_annotation(self, xml_path, im_id):
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
            if n < 100 and kp.attrib['visible'] == '1':
                kps_out[n, 0] = float(kp.attrib['x'])
                kps_out[n, 1] = float(kp.attrib['y'])
                kps_out[n, 2] = 2
                cnt += 1
                self.cnt_kps[n] += 1
        kps_out = list(kps_out.reshape((-1,)))
        return kps_out, cnt

    def _split_train_val(self, val_n=800):
        """
        Random train val split: create im_meta:
        im_path
        im_id,
        cat
        folder
        """
        np.random.seed(1)
        im_data = []
        for folder in glob.glob(os.path.join(self.dir_dataset, 'part*')):
            folder_name = os.path.basename(folder)
            dir_ann = os.path.join(folder, 'annotations')
            for cat in _CATEGORIES:
                xml_paths = glob.glob(os.path.join(dir_ann, cat + os.sep + '*.xml'))
                for xml_path in xml_paths:
                    im_path, im_id = self._extract_filename(xml_path)
                    im_data.append((im_path, im_id, cat, folder_name))
        cnt_ann = len(im_data)
        im_data = list(set(im_data))  # Remove duplicates
        cnt_im = len(im_data)
        train_n = cnt_im - val_n
        random.shuffle(im_data)
        splits = {'train': im_data[:train_n], 'val': im_data[train_n:]}
        print(f'Split {train_n} into training images and {val_n} validation ones')
        print(f'Read {cnt_ann}  annotations')
        return splits

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
            im_id = int(str(int(splits[0])) + str(int(splits[1])))
        else:
            basename = os.path.splitext(sub_dirs[-1])[0]
            num = int(basename[2:])
            im_id = int(str(999) + str(self.map_cat[cat]) + basename[2:])
            if cat == 'sheep' and num == 65:
                ext = '.png'
            elif cat == 'sheep' and (num <= 97 or num >= 190):
                ext = '.jpg'
            else:
                ext = '.jpeg'
            im_path = os.path.join(im_dir, cat, basename + ext)
        assert isinstance(im_id, int), "im id is not numeric"
        return im_path, im_id

    def _find_annotations(self, meta):
        root = os.path.join(self.dir_dataset, meta[3], 'annotations', meta[2],
                            os.path.splitext(os.path.basename(meta[0]))[0])
        xml_paths = glob.glob(root + '[_,.]*xml')  # Avoid duplicates of the form cow13 cow130
        assert xml_paths, "No annotations, expected at least one"
        return xml_paths

    def initiate_json(self):
        """
        Initiate Json for training and val phase
        """
        self.json_file["info"] = dict(url="https://github.com/vita-epfl/openpifpaf",
                                      date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                                      description="Conversion of AnimalPose dataset into MS-COCO format")

        self.json_file["categories"] = [dict(name='animal',
                                             id=1,
                                             skeleton=ANIMAL_SKELETON,
                                             supercategory='animal',
                                             keypoints=[])]
        self.json_file["images"] = []
        self.json_file["annotations"] = []


def histogram(cnt_kps):
    bins = np.arange(len(cnt_kps))
    data = np.array(cnt_kps)
    plt.figure()
    plt.bar(bins, data)
    plt.xticks(np.arange(len(cnt_kps), step=5))
    plt.show()
    plt.close()


def make_new_directory(dir_out):
    """Remove the output directory if already exists (avoid residual txt files)"""
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    print("!Created empty output directory!".format(dir_out))


def main():
    args = cli()
    voc_coco = VocToCoco(args.dir_data, args.dir_out, args)
    voc_coco.process()


if __name__ == "__main__":
    main()
