import pandas as pd 
import pycocotools.mask as maskUtils
import mmcv
import os
import numpy as np
from tqdm import tqdm
import pickle
import cv2
stage = 'train'
open_dir = 'data/OpenImages'
meta_path = f'{open_dir}/annotations/{stage}-image-metas.pkl'
with open(meta_path,'rb') as f:
    img_metas = pickle.load(f)
    img_metas_dict = {}
    for meta in img_metas:
        file_id =os.path.basename(meta['filename']).split('.')[0]
        img_metas_dict[file_id] = meta
    print(len(img_metas),len(img_metas_dict))
    img_metas = img_metas_dict

save_path = f'{open_dir}/annotations/{stage}-image-metas-dict.pkl'
with open(save_path,'wb') as f:
    pickle.dump(img_metas,f)

    