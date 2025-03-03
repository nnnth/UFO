# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset
from .coco_caption import COCOCaption
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import MultiImageMixDataset, ConcatDataset
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset
from .ade import ADE20KDataset
from .basesegdataset import BaseSegDataset
from .refcoco import RefCOCO
from .coco_stuff import COCOStuffDataset
from .nuimage import NuimageDataset
from .openimages_inseg import OpenImagesDatasetInseg
from .nuimage_seg import NuimageSegDataset
from .mapillary import MapillaryDataset_v2
from .refer_seg import ReferSegDataset
from .reason_seg import ReasonSegDataset
from .paco_lvis import PACOLVISDataset
from .pascal_part import PascalPartDataset
from .llava_665k import LLaVA665K
__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset', 'ADE20KDataset', 'BaseSegDataset','COCOCaption','RefCOCO',
    'COCOStuffDataset','NuimageDataset','OpenImagesDatasetInseg','NuimageSegDataset','MapillaryDataset_v2',
    'ReferSegDataset', 'ReasonSegDataset',
    'PACOLVISDataset', 'PascalPartDataset',
    'LLaVA665K'
]
