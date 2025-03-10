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
seg_prefix = f'{open_dir}/segmentation/{stage}'
def singleMask2rle(mask):
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
csv_path = f'{open_dir}/annotations/{stage}-annotations-object-segmentation.csv'

meta_path = f'{open_dir}/annotations/{stage}-image-metas.pkl'
with open(meta_path,'rb') as f:
    img_metas = pickle.load(f)
    img_metas_dict = {}
    for meta in img_metas:
        file_id =os.path.basename(meta['filename']).split('.')[0]
        img_metas_dict[file_id] = meta
    print(len(img_metas),len(img_metas_dict))
    img_metas = img_metas_dict

df = pd.read_csv(csv_path)
sort_df = df.sort_values(by='ImageID')
# import ipdb
# ipdb.set_trace()
rle_lst = []
print("all df",len(sort_df))
count = 0
for idx, line in tqdm(sort_df.iterrows()):
    # if line[1] != 'fffc2f36b181a4fb' and line[1] != 'fff2268a1b921e8e':
    #     continue
    if count < 1000000:
        count += 1
        continue
    try:
        mask_path = os.path.join(seg_prefix,line[0])
        if not os.path.exists(mask_path):
            print(mask_path)
            continue
        mask = mmcv.imread(mask_path)
    except:
        print(mask_path)
        print("fail read")
        continue
    continue
    # resize 1024
    image_id = line[1]
    meta = img_metas[image_id]

    img_shape = meta['ori_shape']
    # if image_id == 'fffc2f36b181a4fb' or image_id == 'fff2268a1b921e8e':
    #     import ipdb
    #     ipdb.set_trace()
    mask = cv2.resize(mask,img_shape[:2][::-1])
    assert mask.shape[0] == img_shape[0] and mask.shape[1] == img_shape[1]
    mask = (mask[...,0] > 1).astype(np.uint8)
    rle = singleMask2rle(mask)
    rle_lst.append(rle)
# exit()
# np.save(rle_lst,f'{stage}_rle.npy')
sort_df.insert(sort_df.shape[1],'rle',rle_lst)
# import ipdb
# ipdb.set_trace()
sort_df.to_csv(f'{open_dir}/annotations/{stage}-annotations-object-segmentation_sort_resize.csv',index=False)
