import json
import os
import random

import cv2
import torch
import torch.nn.functional as F

from mmdet.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class LLaVA665K(BaseDataset):
    ignore_label = 255

    def __init__(
        self,
        data_root='data/llava-665k/',
        img_dir='data/',
        vqa_data="llava_v1_5_mix665k.json",
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
        with open(os.path.join(DATA_DIR, self.vqa_data)) as f:
            vqa_data = json.load(f)
        data_list = []
        ignore_count = 0
        for item in vqa_data:
            if 'image' not in item:
                ignore_count += 1
                continue
            
        
            img_path = os.path.join(self.vqa_image_root, item["image"])
            img_path = img_path.replace('gqa','gqa_vqa')
            img_path = img_path.replace('VG_100K_2','images')
            img_path = img_path.replace('VG_100K','images')

            if 'ocr_vqa' in img_path:
                if img_path in [
                    'data/ocr_vqa/images/1609496760.jpg',
                    'data/ocr_vqa/images/1580405568.jpg',
                    'data/ocr_vqa/images/986042501.jpg',
                    'data/ocr_vqa/images/1844006360.jpg',
                    'data/ocr_vqa/images/1512198471.jpg',
                    'data/ocr_vqa/images/1517152860.jpg',
                    'data/ocr_vqa/images/761181865.jpg'
                ]:
                    print(f'not exits {img_path}')
                    continue
            conversations = item['conversations']

            check_answer = False 
            for conv in conversations:
                if conv['from'] == 'gpt':
                    if len(conv['value']) == 0:
                        check_answer = True 
                        print("jump no answer",conversations)
                        break
            if check_answer:
                continue
 
            # if 'What vitamin is this vegetable associated with' not in debug_str:
            #     continue

            # conversations = remove_image_token(conversations)
            data_info = {
                'img_path': img_path,
                'img_id': item['id'],
                'conversations': conversations
            }
            data_list.append(data_info)

        print(f'ignore {ignore_count}')
        print("vqa_data: ", len(data_list))

        return data_list