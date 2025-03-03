import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask

from .refer import REFER
from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class ReferSegDataset(BaseDataset):
    ignore_label = 255

    def __init__(
        self,
        data_root,
        num_classes_per_sample: int = 3,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        split='train',
        **kwargs
    ):
        self.refer_seg_data = refer_seg_data
        self.num_classes_per_sample = num_classes_per_sample
        self.split = split

        super().__init__(
            data_root=data_root,
            **kwargs,
        )

    def load_data_list(self):
        DATA_DIR = self.data_root
        self.refer_seg_ds_list = self.refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']

        data_list = []
        pre_num = 0
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

             
            refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
 
            annotations = refer_api.Anns
            imgs = refer_api.Imgs
            for ref in refs_train:
                sentences = ref['sentences']
                ann_id = ref['ann_id']
                img_id = ref['image_id']
                ann = annotations[ann_id]
                img = imgs[img_id]
                
                if ds == "refclef":
                    img_path = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", img["file_name"]
                    )
                else:
                    img_path = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", img["file_name"]
                    )
                
                for sent in sentences:
                    data_info = {
                        'img_path': img_path,
                        'image_id': img_id,
                        'ann_id': ann_id,
                        'text': sent['sent'],
                        'height':img['height'],
                        'width': img['width'],
                        'mask': ann['segmentation'],
                    }
                    data_list.append(data_info)

            
            print(
                "dataset {} (refs {}) (train split) has {} annotations.".format(
                    ds,
                    splitBy,
                    len(data_list) - pre_num,
                )
            )
            pre_num = len(data_list)
        return data_list