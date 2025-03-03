import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json

from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset
import pycocotools.mask as maskUtils

def singleMask2rle(mask):
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

@DATASETS.register_module()
class ReasonSegDataset(BaseDataset):
    ignore_label = 255

    def __init__(
        self,
        data_root,
        num_classes_per_sample: int = 3,
        reason_seg_data="ReasonSeg|train",
        explanatory=-1,
        select_short=False,
        select_long=False,
        select_first=False,
        **kwargs
    ):
        self.reason_seg_data = reason_seg_data
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.explanatory = explanatory
        self.select_short = select_short
        self.select_long = select_long
        self.select_first = select_first

        super().__init__(
            data_root=data_root,
            **kwargs,
        )

    def load_data_list(self):
        base_image_dir = self.data_root

        reason_seg_data, splits = self.reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        data_list = []
        for img_path, json_path in zip(images,jsons):
            img_data = cv2.imread(img_path)
            height, width = img_data.shape[:2]
            mask, sents, is_sentence = get_mask_from_json(json_path, height, width)
            if self.explanatory != -1:
                img_name = img_path.split("/")[-1]
                explain_text = self.img_to_explanation[img_name]['outputs']
            else:
                explain_text = None

            if self.select_first:
                sents = [sents[0]]
            for sent in sents:
                data_info = {
                    'img_path':img_path,
                    'mask_path':json_path,
                    'text':sent,
                    'height': height,
                    'width': width,
                    'is_sentence':is_sentence,
                    'explain_text':explain_text,
                }
                if not self.select_short and not self.select_long:
                    data_list.append(data_info)
                elif self.select_short:
                    if not is_sentence:
                        data_list.append(data_info)
                else:
                    if is_sentence:
                        data_list.append(data_info)
        return data_list
        