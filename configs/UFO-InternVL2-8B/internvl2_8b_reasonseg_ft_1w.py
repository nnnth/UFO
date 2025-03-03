_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
base_img_size = 448
load_from = './ufo-internvl2-8b-instruction.pth'
# hyper parameter for each tasks
insseg_cfgs = dict(
    grid_resolution_perwin=[10, 10],
    samples_grids_eachwin=40,
    grid_interpolate=True)

semseg_cfgs = dict(
    grid_resolution_perwin=[10, 10],
    samples_grids_eachwin=40,
    grid_interpolate=True)

vqa_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False)

referseg_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False)

model = dict(
    type='UFO_InternVL',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 
    'caption', 'grounding','refer_segmentation','vqa', 'refer_caption','reason_segmentation'],
    use_checkpoints=True,
    lora_r=8,
    lora_alpha=2*8,
    tokenizer=dict(type='AutoTokenizer', 
        name_or_path='./ckpt/InternVL2-8B',
        trust_remote_code=True,
        use_fast=False),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=255,
        pad_size_divisor=448),
    backbone=dict(
        type='AutoModel',
        name_or_path='./ckpt/InternVL2-8B',
        trust_remote_code=True,
        attn_implementation='eager',
        ),
    head_list=dict( 
        # non parametric task-specific heads
        instance_segmentation_head=dict(type='UFOInternVLInsSegHead',
                train_cfg=dict(
                    assigner=dict(
                        type='HungarianAssigner',
                        match_costs=[
                            dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),
                            dict(type='MaskCost', weight=100.0),
                            ])),
                test_cfg=dict(max_per_img=100),
                mask_loss_weight=1.0,
                cls_loss_weight=1.0,
                nms=dict(
                    score_thr=0.1,
                    filter_thr=0.05,
                    kernel='gaussian',  # gaussian/linear
                    sigma=2.0,
                ),
                repeat_times=3,
                sample_prob=1.0,
                mask_token_id=92553,
        ),
        semantic_segmentation_head=dict(type='UFOInternVLSemSegHead',                    
                    train_cfg=dict(
                        assigner=dict(
                            type='HungarianAssigner',
                            match_costs=[
                                dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),
                                dict(type='MaskRandomCost', weight=100.0),])),
                    test_cfg=dict(max_per_img=100),
                    mask_loss_weight=1.0,
                    cls_loss_weight=1.0,
                    repeat_times=3,
                    sample_prob=1.0,
                    mask_token_id=92553,
        ),
        vqa_head=dict(type='UFOInternVLVQAHead',
                            template_name='internlm2-chat'),
        refer_segmentation_head=dict(type='UFOInternVLReferSegHead',
                            mask_token_id=92553,
                            cls_loss_weight=1.0,),
        reason_segmentation_head=dict(type='UFOInternVLReasonSegHead',
                            mask_token_id=92553,),
        ),
    )

# instance segmentation pipeline
paco_lvis_insseg_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',
                                            dataset_name='paco_lvis', 
                                            head_cfg=dict(num_classes=531,
                                                          num_vocal=(base_img_size*2 + 1) + 531 + 1,
                                                          num_bins=base_img_size*2,
                                                          max_length=30),
                                            git_cfg=insseg_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),
]
pascal_part_insseg_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',
                                            dataset_name='pascal_part', 
                                            head_cfg=dict(num_classes=93,
                                                          num_vocal=(base_img_size*2 + 1) + 93 + 1,
                                                          num_bins=base_img_size*2,
                                                          max_length=30),
                                            git_cfg=insseg_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),
]


# semantic segmentation
cocostuff_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            dataset_name='coco_stuff',
                                            head_cfg=dict(num_classes=171,
                                                            num_vocal=172,
                                                            max_lenght=30,
                                                            ignore_index=255),
                                            git_cfg=semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(448, 448)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(448 * x * 0.1), int(448 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg','dataset_name'))]

mapillary_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            dataset_name='mapillary',
                                            head_cfg=dict(num_classes=124,
                                                            num_vocal=125,
                                                            max_lenght=30,
                                                            ignore_index=255),
                                            git_cfg=semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(448, 448)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(448 * x * 0.1), int(448 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg','dataset_name'))]

ade20k_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations', reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            dataset_name='ade20k',
                                            head_cfg=dict(num_classes=150,
                                                            num_vocal=151,
                                                            max_lenght=30,
                                                            ignore_index=255),
                                            git_cfg=semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(448, 448)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(448 * x * 0.1), int(448 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name'))]

ade20k_semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='SegLoadAnnotations', reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            dataset_name='ade20k',
                                            head_cfg=dict(num_classes=150,
                                                            num_vocal=151,
                                                            max_length=30,
                                                            ignore_index=255),
                                            git_cfg=semseg_cfgs)),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg','dataset_name'))]

# refer segmentation pipeline
referseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsReferSeg'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='refer_segmentation', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=referseg_cfgs)),
   dict(type='Resize',
        scale=(448, 448),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_masks',],
        meta_keys=['image_id','img_shape', 'scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]
referseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',
        scale=(448, 448),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='LoadAnnotationsReferSeg'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='refer_segmentation', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=referseg_cfgs)),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_masks', ],
        meta_keys=['image_id','img_shape','ori_shape','scale_factor','task_name', 'head_cfg', 'git_cfg','img_path'],
    ),
]

# vqa pipeline
vqa_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='vqa', 
                            head_cfg=dict(num_classes=896+1,
                                            max_length=20),
                            git_cfg=referseg_cfgs)),
    dict(type='Resize', scale=(448, 448), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs',
        algorithm_keys=['conversations'],
        meta_keys=['image_id','img_shape', 'scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]

# reason segmentation pipeline
reasonseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsReasonSeg',ignore=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='reason_segmentation', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=referseg_cfgs)),
    dict(type='Resize',
        scale=(448, 448),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_masks',],
        meta_keys=['image_id','img_shape', 'scale_factor','task_name', 'head_cfg', 'git_cfg','is_sentence','explain_text'],
    ),
]

reasonseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',
        scale=(448, 448),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='LoadAnnotationsReasonSeg'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='reason_segmentation', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=referseg_cfgs)),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_masks', ],
        meta_keys=['image_id','img_shape','ori_shape','scale_factor','task_name', 'head_cfg', 'git_cfg','img_path','is_sentence','explain_text'],
    ),
]

insseg_ratio = 1/4
semseg_ratio = 1/4
referseg_ratio = 5/24
vqa_ratio = 1/4
reasonseg_ratio = 1/24

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=4, source_ratio=[insseg_ratio*1/2, insseg_ratio*1/2, 
                                                                                    semseg_ratio*1/3, semseg_ratio*1/3, semseg_ratio*1/3,  
                                                                                    referseg_ratio,
                                                                                    vqa_ratio,
                                                                                    reasonseg_ratio], 
                if_group=[True,True,False,False,False,False,False,False,False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
            # instance segmentation: paco-lvis, pascal-part
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                    type='PACOLVISDataset',
                    data_root='data/paco_lvis/',
                    ann_file='annotations/paco_lvis_v1_train.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=paco_lvis_insseg_train_pipeline,
                    backend_args=backend_args)),
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                    type='PascalPartDataset',
                    data_root='data/pascal_part/',
                    ann_file='train.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=pascal_part_insseg_train_pipeline,
                    backend_args=backend_args)),
            # semantic segmentation coco_stuff164k, Mapillary V2, ade20k
            dict(type='COCOStuffDataset',
                data_root='data/coco_stuff164k',
                data_prefix=dict(
                    img_path='images/train2017', seg_map_path='annotations/train2017'),
                return_classes=False,
                pipeline=cocostuff_semseg_train_pipeline),
            dict(type='MapillaryDataset_v2',
                data_root='data/mapillary/',
                data_prefix=dict(
                    img_path='training/images', seg_map_path='training/v2.0/labels'),
                return_classes=False,
                pipeline=mapillary_semseg_train_pipeline),
            dict(type='ADE20KDataset',
                data_root='data/ade/ADEChallengeData2016',
                data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                return_classes=False,
                pipeline=ade20k_semseg_train_pipeline),

            # refer segmentation 
            dict(type='ReferSegDataset',
                data_root='data/refer_seg',
                split='train',
                pipeline=referseg_train_pipeline),

            # VQA
            dict(type='LLaVA665K',
                pipeline=vqa_train_pipeline),
            
            dict(type='ReasonSegDataset',
                data_root='data/',
                reason_seg_data="ReasonSeg|train",
                explanatory=-1,
                pipeline=reasonseg_train_pipeline)

            ]),
            
    )

test_pipeline = referseg_test_pipeline
val_dataloader = dict(batch_size=4,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|test",
            explanatory=-1,
            pipeline=reasonseg_test_pipeline))
test_dataloader = val_dataloader
extra_val_dataloaders = [
    dict(batch_size=4,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|val",
            explanatory=-1,
            pipeline=reasonseg_test_pipeline)),
    dict(batch_size=4,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|test",
            explanatory=-1,
            select_short=True,
            pipeline=reasonseg_test_pipeline)),
    dict(batch_size=4,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|test",
            explanatory=-1,
            select_long=True,
            pipeline=reasonseg_test_pipeline)),
]
import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=16,)

val_evaluator = dict(type='IoUMetricBinaryBest', iou_metrics=['mIoU'])

test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='IoUMetricBinaryBest', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinaryBest', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinaryBest', iou_metrics=['mIoU']),
]

# learning policy
max_iters=10000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
val_cfg = dict(type='MultiSourceValLoop', extra_dataloaders=extra_val_dataloaders, extra_evaluators=extra_val_evaluators)
test_cfg = val_cfg

param_scheduler = [
    dict(type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1000,
        by_epoch=False),
    dict(
          type='CosineAnnealingLR',
          T_max=max_iters,
          eta_min=0,
          begin=1000,
          end=max_iters,
          by_epoch=False,)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=10))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
