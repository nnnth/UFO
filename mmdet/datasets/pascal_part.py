# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO
import os 

@DATASETS.register_module()
class PascalPartDataset(CocoDataset):
    METAINFO = {
        'classes':
        {1: ('aeroplane', 'body'), 2: ('aeroplane', 'wing'), 3: ('aeroplane', 'tail'), 4: ('aeroplane', 'wheel'), 5: ('bicycle', 'wheel'), 6: ('bicycle', 'handlebar'), 
     7: ('bicycle', 'saddle'), 8: ('bird', 'beak'), 9: ('bird', 'head'), 10: ('bird', 'eye'), 11: ('bird', 'leg'), 12: ('bird', 'foot'), 13: ('bird', 'wing'), 
     14: ('bird', 'neck'), 15: ('bird', 'tail'), 16: ('bird', 'torso'), 17: ('bottle', 'body'), 18: ('bottle', 'cap'), 19: ('bus', 'license plate'), 20: ('bus', 'headlight'), 
     21: ('bus', 'door'), 22: ('bus', 'mirror'), 23: ('bus', 'window'), 24: ('bus', 'wheel'), 25: ('car', 'license plate'), 26: ('car', 'headlight'), 27: ('car', 'door'), 
     28: ('car', 'mirror'), 29: ('car', 'window'), 30: ('car', 'wheel'), 31: ('cat', 'head'), 32: ('cat', 'leg'), 33: ('cat', 'ear'), 34: ('cat', 'eye'), 35: ('cat', 'paw'),
     36: ('cat', 'neck'), 37: ('cat', 'nose'), 38: ('cat', 'tail'), 39: ('cat', 'torso'), 40: ('cow', 'head'), 41: ('cow', 'leg'), 42: ('cow', 'ear'), 43: ('cow', 'eye'), 
     44: ('cow', 'neck'), 45: ('cow', 'horn'), 46: ('cow', 'muzzle'), 47: ('cow', 'tail'), 48: ('cow', 'torso'), 49: ('dog', 'head'), 50: ('dog', 'leg'), 51: ('dog', 'ear'), 
     52: ('dog', 'eye'), 53: ('dog', 'paw'), 54: ('dog', 'neck'), 55: ('dog', 'nose'), 56: ('dog', 'muzzle'), 57: ('dog', 'tail'), 58: ('dog', 'torso'), 59: ('horse', 'head'), 
     60: ('horse', 'leg'), 61: ('horse', 'ear'), 62: ('horse', 'eye'), 63: ('horse', 'neck'), 64: ('horse', 'muzzle'), 65: ('horse', 'tail'), 66: ('horse', 'torso'), 
     67: ('motorbike', 'wheel'), 68: ('motorbike', 'handlebar'), 69: ('motorbike', 'headlight'), 70: ('motorbike', 'saddle'), 71: ('person', 'hair'), 72: ('person', 'head'), 
     73: ('person', 'ear'), 74: ('person', 'eye'), 75: ('person', 'nose'), 76: ('person', 'neck'), 77: ('person', 'mouth'), 78: ('person', 'arm'), 79: ('person', 'hand'), 
     80: ('person', 'leg'), 81: ('person', 'foot'), 82: ('person', 'torso'), 83: ('potted plant', 'plant'), 84: ('potted plant', 'pot'), 85: ('sheep', 'head'), 86: ('sheep', 'leg'), 
     87: ('sheep', 'ear'), 88: ('sheep', 'eye'), 89: ('sheep', 'neck'), 90: ('sheep', 'horn'), 91: ('sheep', 'muzzle'), 92: ('sheep', 'tail'), 93: ('sheep', 'torso')},
        'palette':
        None
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = COCO(local_path)

        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            # coco_url is used in LVISv1 instead of file_name
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            # train/val split in specified in url
            raw_img_info['file_name'] = os.path.join(
                            "VOCdevkit", "VOC2010", "JPEGImages", raw_img_info['file_name']
                        )
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis
        print("pascal part", len(data_list))
        return data_list
