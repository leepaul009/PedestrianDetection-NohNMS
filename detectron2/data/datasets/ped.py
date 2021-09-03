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
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse crowdhuman-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["register_ped_instances", "load_ped"]

# load 2 cls:
def load_ped(anno_file, image_dir):
    """
    Return dataset dict.
    """
    with open(anno_file, "r") as f:
        annos = json.load(f)
    image_num = len(annos)
    logger.info("Loaded {} images in Ped from {}".format(image_num, anno_file))

    dataset_dicts = []
    ignore_instances = 0
    instances = 0

    for img_id, anno in enumerate(annos):

        record = {}
        record["image_id"] = img_id + 1 # corresp.t. 
        record["ID"] = anno["file_name"].split('.')[0]
        record["file_name"] = os.path.join(image_dir, anno["file_name"])

        objs = []

        has_ped = False
        for gt_box in anno["annotations"]:
            if gt_box["bbox"][2] < 0 or gt_box["bbox"][3] < 0:
                continue
            obj = {}
            obj["bbox"] = gt_box["bbox"]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            ### VERY IMPORTANT!!!
            if gt_box["category_id"] == 1 and gt_box["ignore"] == 0:
                obj["category_id"] = 0
                has_ped = True
            elif gt_box["category_id"] == 2 and gt_box["ignore"] == 0:
                obj["category_id"] = 1
                has_ped = True
            else:
                obj["category_id"] = -1
                ignore_instances += 1
            instances += 1

            obj["vis_ratio"] = 1
            objs.append(obj)
        
        record["annotations"] = objs
        if has_ped:
            dataset_dicts.append(record)

    logger.info(
        "Loaded {} instances and {} ignore instances in CrowdHuman from {}".format(
            instances, ignore_instances, anno_file
        )
    )

    return dataset_dicts

# load 1 cls:
def load_ped_cls_1(anno_file, image_dir):
    with open(anno_file, "r") as f:
        annos = json.load(f)
    image_num = len(annos)
    logger.info("Loaded {} images in Ped from {}".format(image_num, anno_file))

    dataset_dicts = []
    ignore_instances = 0
    instances = 0

    for img_id, anno in enumerate(annos):
        record = {}
        record["image_id"] = img_id + 1 # corresp.t. 
        record["ID"] = anno["file_name"].split('.')[0]
        record["file_name"] = os.path.join(image_dir, anno["file_name"])
        objs = []
        has_ped = False
        for gt_box in anno["annotations"]:
            if gt_box["bbox"][2] < 0 or gt_box["bbox"][3] < 0:
                continue
            obj = {}
            obj["bbox"] = gt_box["bbox"]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            ### VERY IMPORTANT!!!
            if gt_box["category_id"] == 1 and gt_box["ignore"] == 0:
                obj["category_id"] = 0
                has_ped = True
            else:
                obj["category_id"] = -1
                ignore_instances += 1
            instances += 1

            obj["vis_ratio"] = 1
            objs.append(obj)
        
        record["annotations"] = objs
        if has_ped:
            dataset_dicts.append(record)
    logger.info(
        "Loaded {} instances and {} ignore instances in CrowdHuman from {}".format(
            instances, ignore_instances, anno_file
        )
    )
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

def load_ped_with_original_input(anno_file, image_dir):
    """
    Return dataset dict.
    """
    # anno_lines = open(anno_file, "r").readlines()
    # annos = [json.loads(line.strip()) for line in anno_lines]

    # image_num = len(annos)
    # logger.info("Loaded {} images in CrowdHuman from {}".format(image_num, anno_file))
    with open(anno_file, "r") as f:
        jdata = json.load(f)
    images      = jdata['images']
    categories  = jdata['categories']
    annotations = jdata['annotations']

    image_num = len(images)
    logger.info("Loaded {} images in CrowdHuman from {}".format(image_num, anno_file))

    dataset_dicts = []
    ignore_instances = 0
    instances = 0
    anno_id = 0
    # for img_id, anno in enumerate(annos):
    for img_id, image in enumerate(images):
        # record = {}
        # record["image_id"] = img_id + 1
        # record["ID"] = anno["ID"]
        # record["file_name"] = os.path.join(image_dir, anno["ID"] + ".jpg")
        record = {}
        record["image_id"] = img_id
        record["ID"] = image["file_name"]
        record["file_name"] = os.path.join(image_dir, image["file_name"])

        objs = []
        # for gt_box in anno["gtboxes"]:
        #     if gt_box["fbox"][2] < 0 or gt_box["fbox"][3] < 0:
        #         continue
        #     obj = {}
        #     obj["bbox"] = gt_box["fbox"]
        #     obj["bbox_mode"] = BoxMode.XYWH_ABS
        #     if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) != 0:
        #         obj["category_id"] = -1
        #         ignore_instances += 1
        #     else:
        #         obj["category_id"] = 0
        #     instances += 1

        #     vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(
        #         (gt_box["fbox"][2] * gt_box["fbox"][3])
        #     )
        #     obj["vis_ratio"] = vis_ratio

        #     objs.append(obj)

        while(anno_id < len(annotations)):
            gt_box = annotations[anno_id]
            if gt_box["image_id"] == img_id:
                obj = {}
                obj["bbox"] = gt_box["bbox"]
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if gt_box["category_id"] != 1 or gt_box["ignore"] != 0:
                    obj["category_id"] = -1
                    ignore_instances += 1
                else:
                    obj["category_id"] = 0
                instances += 1

                obj["vis_ratio"] = 1 # visible_box/full_box
                objs.append(obj)
                anno_id += 1
            else:
                break

        record["annotations"] = objs
        dataset_dicts.append(record)

    logger.info(
        "Loaded {} instances and {} ignore instances in CrowdHuman from {}".format(
            instances, ignore_instances, anno_file
        )
    )

    return dataset_dicts

def register_ped_instances(name, metadata, anno_file, image_dir, val_json_files):
    """
    Register CityPersons dataset.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """

    # if name(dateset name) have key word "test":
    if "test" in name:
        DatasetCatalog.register(name, lambda: load_ped_test(anno_file, image_dir))
        return

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ped(anno_file, image_dir))
    # DatasetCatalog.register(name, lambda: load_ped_cls_1(anno_file, image_dir))

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
    if "val" in name:
        for val_json_file in val_json_files:
            if not os.path.exists(val_json_file):
                is_clip = "clip" in val_json_file
                convert_to_coco_dict(anno_file, image_dir, val_json_file, is_clip=is_clip)
                # convert_to_coco_dict_1_cls(anno_file, image_dir, val_json_file, is_clip=is_clip)

def convert_to_coco_dict(anno_file, image_dir, json_file, is_clip=True):
    from tqdm import tqdm

    # anno_lines = open(anno_file, "r").readlines()
    # annos = [json.loads(line.strip()) for line in anno_lines]
    with open(anno_file, "r") as f:
        annos = json.load(f)

    print("Converting dataset dicts into COCO format")

    images = []
    annotations = []
    outside_num, clip_num = 0, 0
    for img_id, anno in tqdm(enumerate(annos)):
        # filename = os.path.join(image_dir, anno["ID"] + ".jpg")
        filename = os.path.join(image_dir, anno["file_name"])
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        # "id" corresp.t. record["image_id"] of load_ped
        image = {"id": img_id + 1, "file_name": filename, "height": height, "width": width}
        images.append(image)

        for gt_box in anno["annotations"]:
            annotation = {}
            x1, y1, w, h = gt_box["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            annotation["id"] = len(annotations) + 1
            annotation["image_id"] = image["id"]

            annotation["area"] = gt_box["bbox"][2] * gt_box["bbox"][3]
            
            ### VERY IMPORTANT!!!
            categories_invalid = gt_box["category_id"] not in [1, 2]
            # if gt_box["category_id"] != 1 or gt_box["category_id"] != 2 or gt_box["ignore"] == 1:
            if categories_invalid or gt_box["ignore"] == 1:
                annotation["ignore"] = 1
            elif outside(bbox, height, width):
                annotation["ignore"] = 1
                outside_num += 1
            elif is_clip and (
                (bbox[0] < 0) or (bbox[1] < 0) or (bbox[2] > width) or (bbox[3] > height)
            ):
                bbox = clip_bbox(bbox, [height, width])
                clip_num += 1
                annotation["ignore"] = 0
            else:
                annotation["ignore"] = 0

            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]

            ### VERY IMPORTANT!!!
            annotation["category_id"] = 1 if gt_box["category_id"] == 1 else 2
            # annotation["category_id"] = gt_box["category_id"]
            # annotation["category_id"] = 1
            annotation["bbox"] = [round(float(x), 3) for x in bbox]
            annotation["height"] = annotation["bbox"][3]
            # vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(annotation["area"])
            annotation["vis_ratio"] = 1 # vis_ratio
            annotation["iscrowd"] = 0
            annotations.append(annotation)

    print("outside num: {}, clip num: {}".format(outside_num, clip_num))
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated CrowdHuman json file for Detectron2.",
    }

    # categories = [{"id": 1, "name": "pedestrian"}]
    # [{'id': 1, 'name': 'Pedestrian'}, {'id': 2, 'name': 'Cyclist'}, {'id': 3, 'name': 'Car'}, {'id': 4, 'name': 'Truck'}, {'id': 5, 'name': 'Van'}]
    categories = [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "cyclist"}]

    coco_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    try:
        json.dump(coco_dict, open(json_file, "w"))
    except:
        print("json dump falied in crowdhuman convert processing.")
        from IPython import embed

        embed()



def convert_to_coco_dict_1_cls(anno_file, image_dir, json_file, is_clip=True):
    from tqdm import tqdm

    # anno_lines = open(anno_file, "r").readlines()
    # annos = [json.loads(line.strip()) for line in anno_lines]
    with open(anno_file, "r") as f:
        annos = json.load(f)

    print("Converting dataset dicts into COCO format")

    images = []
    annotations = []
    outside_num, clip_num = 0, 0
    for img_id, anno in tqdm(enumerate(annos)):
        # filename = os.path.join(image_dir, anno["ID"] + ".jpg")
        filename = os.path.join(image_dir, anno["file_name"])
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        image = {"id": img_id + 1, "file_name": filename, "height": height, "width": width}
        images.append(image)

        for gt_box in anno["annotations"]:
            annotation = {}
            x1, y1, w, h = gt_box["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            annotation["id"] = len(annotations) + 1
            annotation["image_id"] = image["id"]

            annotation["area"] = gt_box["bbox"][2] * gt_box["bbox"][3]
            if gt_box["category_id"] != 1 or gt_box["ignore"] == 1:
                annotation["ignore"] = 1
            elif outside(bbox, height, width):
                annotation["ignore"] = 1
                outside_num += 1
            elif is_clip and (
                (bbox[0] < 0) or (bbox[1] < 0) or (bbox[2] > width) or (bbox[3] > height)
            ):
                bbox = clip_bbox(bbox, [height, width])
                clip_num += 1
                annotation["ignore"] = 0
            else:
                annotation["ignore"] = 0

            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]

            annotation["category_id"] = 1
            annotation["bbox"] = [round(float(x), 3) for x in bbox]
            annotation["height"] = annotation["bbox"][3]
            # vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(annotation["area"])
            annotation["vis_ratio"] = 1 # vis_ratio
            annotation["iscrowd"] = 0
            annotations.append(annotation)

    print("outside num: {}, clip num: {}".format(outside_num, clip_num))
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated CrowdHuman json file for Detectron2.",
    }

    categories = [{"id": 1, "name": "pedestrian"}]

    coco_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    try:
        json.dump(coco_dict, open(json_file, "w"))
    except:
        print("json dump falied in crowdhuman convert processing.")
        from IPython import embed

        embed()


def convert_to_coco_dict_backup(anno_file, image_dir, json_file, is_clip=True):
    from tqdm import tqdm

    anno_lines = open(anno_file, "r").readlines()
    annos = [json.loads(line.strip()) for line in anno_lines]

    print("Converting dataset dicts into COCO format")

    images = []
    annotations = []
    outside_num, clip_num = 0, 0
    for img_id, anno in tqdm(enumerate(annos)):
        filename = os.path.join(image_dir, anno["ID"] + ".jpg")
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        image = {"id": img_id + 1, "file_name": filename, "height": height, "width": width}
        images.append(image)

        for gt_box in anno["gtboxes"]:
            annotation = {}
            x1, y1, w, h = gt_box["fbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            annotation["id"] = len(annotations) + 1
            annotation["image_id"] = image["id"]

            annotation["area"] = gt_box["fbox"][2] * gt_box["fbox"][3]
            if gt_box["tag"] != "person" or gt_box["extra"].get("ignore", 0) == 1:
                annotation["ignore"] = 1
            elif outside(bbox, height, width):
                annotation["ignore"] = 1
                outside_num += 1
            elif is_clip and (
                (bbox[0] < 0) or (bbox[1] < 0) or (bbox[2] > width) or (bbox[3] > height)
            ):
                bbox = clip_bbox(bbox, [height, width])
                clip_num += 1
                annotation["ignore"] = 0
            else:
                annotation["ignore"] = 0

            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]

            annotation["category_id"] = 1
            annotation["bbox"] = [round(float(x), 3) for x in bbox]
            annotation["height"] = annotation["bbox"][3]
            vis_ratio = (gt_box["vbox"][2] * gt_box["vbox"][3]) / float(annotation["area"])
            annotation["vis_ratio"] = vis_ratio
            annotation["iscrowd"] = 0
            annotations.append(annotation)

    print("outside num: {}, clip num: {}".format(outside_num, clip_num))
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated CrowdHuman json file for Detectron2.",
    }

    categories = [{"id": 1, "name": "pedestrian"}]

    coco_dict = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    try:
        json.dump(coco_dict, open(json_file, "w"))
    except:
        print("json dump falied in crowdhuman convert processing.")
        from IPython import embed

        embed()


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
