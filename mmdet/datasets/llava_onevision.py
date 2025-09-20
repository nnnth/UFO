import json
import os
import random

import cv2
import torch
import torch.nn.functional as F

from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset
import jsonlines

@DATASETS.register_module()
class LLaVAOneVision(BaseDataset):
    ignore_label = 255

    def __init__(
        self,
        data_root='data/llava_onevision/',
        img_dir='data/',
        vqa_data="train.jsonl",
        **kwargs
    ):
        self.vqa_data = vqa_data
        self.img_dir = img_dir
        super().__init__(
            data_root=data_root,
            **kwargs,
        )
        
    def load_data_list(self):
        DATA_DIR = self.data_root
        self.vqa_image_root = self.img_dir

        vqa_data = []
        with open(os.path.join(DATA_DIR, self.vqa_data)) as f:
            for item in jsonlines.Reader(f):
                vqa_data.append(item)
        
        data_list = []
        for item in vqa_data:
            if 'image' not in item:
                continue

            img_path = os.path.join(self.vqa_image_root, item["image"])
            if 'gif' in img_path or 'GIF' in img_path:
                continue

            # constrain turns
            conversations = item['conversations']
            if len(conversations) == 1:
                continue
            if conversations[0]['from'] == 'gpt':
                continue
            if len(conversations) > 6:
                conversations = conversations[:6]
            
            # constrain lens
            concat_str = ''
            for k, conv in enumerate(conversations):
                concat_str += conv['value']
            if len(concat_str) > 1200:
                continue

            check_answer = False 
            for k, conv in enumerate(conversations):
                if conv['from'] == 'gpt':
                    if len(conv['value']) == 0:
                        check_answer = True 
                        print("jump no answer",conversations)
                        break
                if k == 0 and '<image>' not in conv['value']:
                    conversations[k]['value'] = '<image>\n' + conversations[k]['value']
            if check_answer:
                continue
 
            data_info = {
                'img_path': img_path,
                'img_id': item['id'],
                'conversations': conversations
            }
            data_list.append(data_info)
        
        return data_list