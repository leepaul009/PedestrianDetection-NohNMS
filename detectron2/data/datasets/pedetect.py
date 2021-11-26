# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved
"""
@contact:
quanzhe.li@cn.bosch.com
"""
import logging
import os
import numpy as np
import pandas as pd

import cv2
import datetime
from scipy import io
import json

from fvcore.common.timer import Timer
from traitlets.traitlets import default
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

from collections import defaultdict


logger = logging.getLogger(__name__)

__all__ = ["register_pedetect_instances", "load_pedetect"]

PEDETECT_CATEGORY = ('Pedestrian',)

def load_pedetect(anno_file, image_dir, is_train=True):
    with open(anno_file, 'r') as f:
        json_data = json.load(f)
    images      = json_data['images']
    categories  = json_data['categories']
    annotations = json_data['annotations']
    logger.info("Loaded {} images in Ped from {}".format(len(images), anno_file))

    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    for ann in annotations:
        imgToAnns[ann['image_id']].append(ann)
        catToImgs[ann['category_id']].append(ann['image_id'])

    # filter
    valid_classes = [ c['id'] for c in categories if c['name'] in PEDETECT_CATEGORY]
    ids_with_ann = set(_['image_id'] for _ in annotations)
    ids_in_cat = set()
    for c_id in valid_classes:
        ids_in_cat |= set(catToImgs[c_id])
    ids_in_cat &= ids_with_ann

    dataset_dicts = []
    ignore_instances = 0
    instances = 0

    bad_images = ['1502445483179.jpg', '1502433988638.jpg']
    for img in images:
        if is_train and img['id'] not in ids_in_cat:
            continue
        if is_train and min(img['width'], img['height']) < 32:
            continue
        ## bad image
        if is_train and img['file_name'] in bad_images:
            continue
        record = {}
        ### VERY IMPORTANT!!!
        record['image_id']  = img['id'] # used fro evaluation
        record['ID']        = len(dataset_dicts)
        record['file_name'] = os.path.join(image_dir, img['file_name'])
        
        objs = []
        exist_valid_bbox = False
        for ann in imgToAnns[ img['id'] ]:
            # if ann.get('ignore', False): # if ann['ignore'] == 1, continue
            #     continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img['height']) - max(y1, 0))
            if is_train and inter_w * inter_h == 0:
                continue
            if is_train and ann['area'] <= 0 or w < 1 or h < 1:
                continue
            # if ann['category_id'] not in valid_classes:
            #     continue
            obj              = {}
            obj["bbox"]      = ann["bbox"]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            ### VERY IMPORTANT!!!
            if is_train:
                if ann['category_id'] in valid_classes:
                    if ann['ignore'] == 0 and ann['iscrowd'] == 0:
                        obj['category_id'] = ann['category_id'] - 1
                        exist_valid_bbox = True
                    else: 
                        # pass 'ignore|iscrowd' bbox in train
                        continue
                else:
                    obj['category_id'] = -1
                    ignore_instances += 1
                
                segm = ann.get("segmentation", None)
                if segm:
                    if not isinstance(segm, dict):
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            continue
                    obj['segmentation'] = segm
            else:
                ### for evaluation, do not filter 'ignore|iscrowd' bbox
                ### for evaluation, do not filter invalid class
                if ann['category_id'] in valid_classes:
                    obj['category_id'] = ann['category_id'] - 1
                else:
                    obj['category_id'] = -1
                    # ignore_instances += 1
                exist_valid_bbox = True
            
            instances +=1
            obj['vis_ratio'] = 1
            objs.append(obj)
        record['annotations'] = objs
        # make sure to see all the images in evaluation
        if is_train and not exist_valid_bbox:
            continue
        dataset_dicts.append(record)

    
    logger.info("Remained image {}. Loaded {} instances and {} ignore instances"
        .format(len(dataset_dicts), instances, ignore_instances))
    return dataset_dicts


def load_ped_test(anno_file, image_dir):

    df = pd.read_csv(anno_file)
    dataset_dicts = []

    for i in range(len(df)):
        record = {}
        record["image_id"]  = df.loc[i, "image_id"] + 1
        # record["width"]     = df.loc[i, "width"]
        # record["height"]    = df.loc[i, "height"]
        file_name = df.loc[i, "file_name"]
        record["file_name"] = os.path.join(image_dir, file_name)
        record["ID"] = file_name.split('.')[0] # image name
        dataset_dicts.append(record)

    return dataset_dicts

def register_pedetect_instances(name, metadata, anno_file, image_dir, val_json_files):

    if "test" in name:
        DatasetCatalog.register(name, lambda: load_ped_test(anno_file, image_dir))
        return
    is_train = True if "train" in name else False
    DatasetCatalog.register(name, lambda: load_pedetect(anno_file, image_dir, is_train))

    if not isinstance(val_json_files, list):
        val_json_files = [val_json_files]

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    # json_file is used in evaluation, ex. crowdhuman_evaluation.py line 107
    MetadataCatalog.get(name).set(
        json_file=val_json_files,
        anno_file=anno_file,
        image_dir=image_dir,
        evaluator_type="ped",
        **metadata,
    )




def clip_bbox(bbox, box_size):
    height, width = box_size
    bbox[0] = np.clip(bbox[0], 0, width)
    bbox[1] = np.clip(bbox[1], 0, height)
    bbox[2] = np.clip(bbox[2], 0, width)
    bbox[3] = np.clip(bbox[3], 0, height)
    return bbox


def outside(bbox, height, width):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if cx < 0 or cx > width or cy < 0 or cy > height:
        return True
    else:
        return False
