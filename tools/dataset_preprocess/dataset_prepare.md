# Training Dataset
```
UFO
|──data
|  |──ade
|  |  |──ADEChallengeData2016
|  |  |  |──annorations
|  |  |  |  |──training & validation
|  |  |  |──images
|  |  |  |  |──training & validation
|  |  |  |──objectInfo150.txt
|  |  |  |──sceneCategories.txt
|  |──coco
|  |  |──annotations
|  |  |  |──*.json
|  |  |──train2017
|  |  |  |──*.jpg
|  |  |──val2017
|  |  |  |──*.jpg
|  |──coco_2014
|  |  |──annotations
|  |  |  |──*.json
|  |  |  |──coco_karpathy_test.json
|  |  |  |──coco_karpathy_train.json
|  |  |  |──coco_karpathy_val_gt.json
|  |  |  |──coco_karpathy_val.json
|  |  |──train2014
|  |  |  |──*.jpg
|  |  |──val2014
|  |  |  |──*.jpg
|  |  |──refcoco
|  |  |  |──*.p
|  |  |──refcoco+
|  |  |  |──*.p
|  |  |──refcocog
|  |  |  |──*.p
|  |──OpenImages
|  |  |──annotations (follow https://github.com/open-mmlab/mmdetection/tree/main/configs/openimages)
|  |  |  |──train-annotations-object-segmentation_sort_resize.csv
|  |  |  |──val-annotations-object-segmentation_sort_resize.csv
|  |  |──OpenImages
|  |  |  |──train
|  |  |  |  |──*.jpg
|  |  |  |──validation
|  |  |  |  |──*.jpg
|  |  |  |──test
|  |  |  |  |──*.jpg
|  |  |──segmentation
|  |  |  |──train
|  |  |  |  |──*.png
|  |  |  |──validation
|  |  |  |  |──*.png
|  |  |  |──test
|  |  |  |  |──*.png

|  |──Objects365
|  |  |──Obj365_v2
|  |  |  |──annotations
|  |  |  |  |──*.jpg
|  |  |  |──train
|  |  |  |  |──patch0
|  |  |  |  |──patch1
             ...
|  |  |  |──val
|  |  |  |  |──patch0
|  |  |  |  |──patch1
             ...
|  |  |  |──annotations
|  |  |  |  |──zhiyuan_objv2_train.json 
|  |  |  |  |──zhiyuan_objv2_val.json
             ...
|  |──lvis_v1
|  |  |──annotations
|  |  |  |──*.json
|  |  |──train2017
|  |  |  |──*.jpg
|  |  |──val2017
|  |  |  |──*.jpg

|  |──coco_stuff164k
|  |  |──annotations
|  |  |  |──train2017
|  |  |  |  |——000000250893_labelTrainIds.png 
|  |  |  |  |——000000250893.png 
|  |  |  |——val2017
|  |  |  |  |——000000229601_labelTrainIds.png
|  |  |  |  |——000000229601.png
|  |  |──images
|  |  |  |──train2017
|  |  |  |  |——*.png 
|  |  |  |——val2017
|  |  |  |  |——*.png 

|  |──nuimages
|  |  |──annotations
|  |  |  |── nuimages_v1.0-train.json
|  |  |  |── nuimages_v1.0-val.json
|  |  |──calibrated
|  |  |──samples
|  |  |  |── CAM_BACK
|  |  |  |—— CAM_BACK_LEFT
|  |  |  |—— CAM_BACK_RIGHT
|  |  |  |—— CAM_FRONT
|  |  |  |—— CAM_FRONT_LEFT
|  |  |  |—— CAM_FRONT_RIGHT
|  |  |──v1.0-mini
|  |  |──v1.0-test
|  |  |──v1.0-train
|  |  |──v1.0-val

|  |──nuimages_seg
|  |  |──annotations
|  |  |  |── training
|  |  |  |   |── *.png
|  |  |  |── validation
|  |  |──images
|  |  |  |── training
|  |  |  |   |── *.jpg
|  |  |  |── validation

│   │── pascal_part
│   │   ├── train.json
│   │   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   ├── SegmentationClassContext
│   │   │   │   ├── ImageSets
│   │   │   │   │   ├── SegmentationContext
│   │   │   │   │   │   ├── train.txt
│   │   │   │   │   │   ├── val.txt
│   │   │   │   ├── trainval_merged.json

│   │── paco_lvis
│   │   ├── annotations
│   │   │   ├── paco_lvis_v1_train.json
│   │   │   ├── paco_lvis_v1_val.json
│   │   │   ├── paco_lvis_v1_test.json
│   │   ├── train2017
│   │   │   ├── *.jpg
│   │   ├── val2017
│   │   │   ├── *.jpg

│   ├── mapillary
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   │   ├── validation
│   │   │   ├── images
|   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons

│   ├── refclef
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   |   ├── images
│   │   ├── refs(unc).p
│   │   ├── instances.json



```

## COCO 2017
python tools/misc/download_dataset.py --dataset-name coco2017

## ADE20K
mim download mmsegmentation --dataset ade20k

## COCO Caption
python tools/misc/download_dataset.py --dataset-name coco2014
- karpathy download link: https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml

```shell
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json
```

## OpenImages
1. You need to download and extract Open Images dataset.

2. The Open Images dataset does not have image metas (width and height of the image),
   which will be used during training and testing (evaluation). We suggest to get test image metas before
   training/testing by using `tools/misc/get_image_metas.py`.

   **Usage**

   ```shell
   python tools/misc/get_image_metas.py ${CONFIG} \
   --dataset ${DATASET TYPE} \  # train or val or test
   --out ${OUTPUT FILE NAME}
   ```

3. The directory should be like this:

   ```none
   mmdetection
   ├── mmdet
   ├── tools
   ├── configs
   ├── data
   │   ├── OpenImages
   │   │   ├── annotations
   │   │   │   ├── bbox_labels_600_hierarchy.json
   │   │   │   ├── class-descriptions-boxable.csv
   │   │   │   ├── oidv6-train-annotations-bbox.scv
   │   │   │   ├── validation-annotations-bbox.csv
   │   │   │   ├── validation-annotations-human-imagelabels-boxable.csv
   │   │   │   ├── validation-image-metas.pkl      # get from script
   │   │   ├── challenge2019
   │   │   │   ├── challenge-2019-train-detection-bbox.txt
   │   │   │   ├── challenge-2019-validation-detection-bbox.txt
   │   │   │   ├── class_label_tree.np
   │   │   │   ├── class_sample_train.pkl
   │   │   │   ├── challenge-2019-validation-detection-human-imagelabels.csv       # download from official website
   │   │   │   ├── challenge-2019-validation-metas.pkl     # get from script
   │   │   ├── OpenImages
   │   │   │   ├── train           # training images
   │   │   │   ├── test            # testing images
   │   │   │   ├── validation      # validation images
   ```
## Object365
1. You need to download and extract Objects365 dataset. Users can download Objects365 V2 by using `tools/misc/download_dataset.py`.

   **Usage**

   ```shell
   python tools/misc/download_dataset.py --dataset-name objects365v2 \
   --save-dir ${SAVING PATH} \
   --unzip \
   --delete  # Optional, delete the download zip file
   ```

   **Note:** There is no download link for Objects365 V1 right now. If you would like to download Objects365-V1, please visit [official website](http://www.objects365.org/) to concat the author.

2. The directory should be like this:

   ```none
   mmdetection
   ├── mmdet
   ├── tools
   ├── configs
   ├── data
   │   ├── Objects365
   │   │   ├── Obj365_v1
   │   │   │   ├── annotations
   │   │   │   │   ├── objects365_train.json
   │   │   │   │   ├── objects365_val.json
   │   │   │   ├── train        # training images
   │   │   │   ├── val          # validation images
   │   │   ├── Obj365_v2
   │   │   │   ├── annotations
   │   │   │   │   ├── zhiyuan_objv2_train.json
   │   │   │   │   ├── zhiyuan_objv2_val.json
   │   │   │   ├── train        # training images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   │   │   │   ├── val          # validation images
   │   │   │   │   ├── patch0
   │   │   │   │   ├── patch1
   │   │   │   │   ├── ...
   ```
## LVIS 1.0
  ```
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
  ```
## COCO Stuff 164k

For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.

```shell
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

## nuImages
Download samples and metadata from https://www.nuscenes.org/nuimages#download
tar -zxvf nuimages-v1.0-all-samples.tgz 

tar -zxvf nuimages-v1.0-all-metadata.tgz 
```shell
python -u tools/dataset_converters/nuimage_converter.py --data-root ${DATA_ROOT} --version ${VERSIONS} \
                                                    --out-dir ${OUT_DIR} --nproc ${NUM_WORKERS} --extra-tag ${TAG}
```

- `--data-root`: the root of the dataset, defaults to `./data/nuimages`.
- `--version`: the version of the dataset, defaults to `v1.0-mini`. To get the full dataset, please use `--version v1.0-train v1.0-val v1.0-mini`
- `--out-dir`: the output directory of annotations and semantic masks, defaults to `./data/nuimages/annotations/`.
- `--nproc`: number of workers for data preparation, defaults to `4`. Larger number could reduce the preparation time as images are processed in parallel.
- `--extra-tag`: extra tag of the annotations, defaults to `nuimages`. This can be used to separate different annotations processed in different time for study.
```shell
python -u tools/dataset_converters/nuimage_converter.py --data-root ./data/nuimages --version v1.0-train v1.0-val v1.0-mini \
                                                    --out-dir ./data/nuimages/annotations/ --nproc 8 
```

## nuImages Segmentation
python tools/dataset_preprocess/link_nuimage_seg.py

## OpenImages Instance Segmentation
```
cd data/OpenImages
mkdir segmentation
cd segmentation 
mkdir train
mkdir validation
mkdir test
python tools/dataset_preprocess/download_open_inseg.py
```
sort instances by ImageID:
```
python tools/dataset_preprocess/sort_instances.py
```
this command will create `train-annotations-object-segmentation_sort_resize.csv` and `validation-annotations-object-segmentation_sort_resize.csv` in `data/OpenImages/annotations`.

# PACO-LVIS, PASCAL-Part
We prepare data following [LISA](https://github.com/dvlab-research/LISA?tab=readme-ov-file#training-data-preparation)

# refcoco, refcoco+,refcocog
```
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
```
- use images from coco 2014

- use refs(unc).p for refcoco+ and refcoco

- use refs(umd).p for refcocog

# refclef
```
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
```
download image data from https://www.imageclef.org/SIAPRdata



 