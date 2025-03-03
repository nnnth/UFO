_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
base_img_size = 448
# hyper parameter for each tasks
det_cfgs = dict(
    grid_resolution_perwin=[10, 10],
    samples_grids_eachwin=40,
    grid_interpolate=True)

insseg_cfgs = dict(
    grid_resolution_perwin=[10, 10],
    samples_grids_eachwin=40,
    grid_interpolate=True)

semseg_cfgs = dict(
    grid_resolution_perwin=[10, 10],
    samples_grids_eachwin=40,
    grid_interpolate=True)

grounding_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False)

caption_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False)

model = dict(
    type='UFO_InternVL_Full',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 
    'caption', 'grounding'],
    use_checkpoints=True,
    train_mlp=True,
    train_vit=True,
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
        detection_head=dict(type='UFOInternVLDetHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),])),
            test_cfg=dict(max_per_img=100),
            nms=dict(type='soft_nms', iou_threshold=0.5),
            repeat_times=3,
            sample_prob=0.0,
        ),
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
                sample_prob=0.0,
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
                    sample_prob=0.0,
                    mask_token_id=92553,
        ),
        caption_head=dict(type='UFOInternVLCaptionHead'),
        grounding_head=dict(type='UFOInternVLGroundHead'),
        ),
    )

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# det pipeline
coco_det_load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            dataset_name='coco',
                                            head_cfg=dict(num_classes=80,
                                                          num_vocal=(base_img_size*2 + 1) + 80 + 1,
                                                          num_bins=base_img_size*2,
                                                          max_length=30),
                                            git_cfg=det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5)),

]
coco_det_train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

coco_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection',
                                            dataset_name='coco', 
                                            head_cfg=dict(num_classes=80,
                                                          num_vocal=(base_img_size*2 + 1) + 80 + 1,
                                                          num_bins=base_img_size*2,
                                                          max_length=30),
                                            git_cfg=det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]


# instance segmentation pipeline 
coco_insseg_load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            dataset_name='coco',
                                            head_cfg=dict(num_classes=80,
                                                          num_vocal=(base_img_size*2 + 1) + 80 + 1,
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
]
coco_insseg_train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),
    ]

coco_insseg_test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            dataset_name='coco',
                                            head_cfg=dict(num_classes=80,
                                                          num_vocal=(base_img_size*2 + 1) + 80 + 1,
                                                          num_bins=base_img_size*2,
                                                          max_length=30,),
                                            git_cfg=insseg_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]


# semantic segmentation
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
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(base_img_size * x * 0.1), int(base_img_size * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(base_img_size, base_img_size), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name'))]

ade20k_semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
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

# grounding pipeline
grounding_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=grounding_cfgs)),
    dict(type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                backend='cv2')
        ],
        prob=0.5),
    dict(type='mmdet.RandomCrop',
        crop_type='relative_range',
        crop_size=(0.8, 0.8),
        allow_negative_crop=False),
    dict(type='RandomChoiceResize',
        scales=[(448, 448)],
        keep_ratio=False),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes',],
        meta_keys=['image_id','img_shape', 'scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]
grounding_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=grounding_cfgs)),
    dict(type='Resize',
        scale=(448, 448),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', ],
        meta_keys=['image_id','img_shape','scale_factor','task_name', 'head_cfg', 'git_cfg','img_path'],
    ),
]

caption_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=92554,
                                                            num_vocal=92554,
                                                            max_length=30,
                                                            ignore_index=-100,
                                                            beam_num=2),
                                            git_cfg=caption_cfgs)),
    dict(type='CleanCaption', keys='gt_caption',remove_chars=[],lowercase=False),
    dict(type='Resize', scale=(448, 448), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', algorithm_keys=['gt_caption'], meta_keys=['image_id','img_shape', 'task_name', 'head_cfg', 'git_cfg'],),
]

caption_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=92554,
                                                            num_vocal=92554,
                                                            max_length=20,
                                                            ignore_index=-100,
                                                            beam_num=3,
                                                            temperature=0.7,
                                                            alpha=0.75),
                                            git_cfg=caption_cfgs)),
    dict(type='Resize', scale=(448, 448), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'img_shape', 'task_name', 'head_cfg', 'git_cfg']),
]



train_dataloader = dict(
    batch_size=3,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=3, source_ratio=[
                                                                                    3/10,3/10,4/30,4/90,4/90,4/90,4/30,
                                                                                    ], 
                 if_group=[True,True,False,False,False,False,False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
            dict(
                type='MultiImageMixDataset',
                dataset=dict(type='CocoDataset',
                    data_root='data/coco/',
                    ann_file='annotations/instances_train2017.json',
                    data_prefix=dict(img='train2017/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=coco_det_load_pipeline,
                    backend_args=backend_args),
                pipeline=coco_det_train_pipeline),
            dict(type='MultiImageMixDataset',
                dataset=dict(type='CocoDataset',
                    data_root='data/coco/',
                    ann_file='annotations/instances_train2017.json',
                    data_prefix=dict(img='train2017/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=coco_insseg_load_pipeline,
                    backend_args=backend_args),
                pipeline=coco_insseg_train_pipeline),
            dict(type='ADE20KDataset',
                data_root='data/ade/ADEChallengeData2016',
                data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                return_classes=False,
                pipeline=ade20k_semseg_train_pipeline),

            # grounding dataset
            dict(type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco/instances.json',
                split_file='refcoco/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(
                type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco+/instances.json',
                split_file='refcoco+/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(
                type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcocog/instances.json',
                split_file='refcocog/refs(umd).p',
                split='train',
                pipeline=grounding_train_pipeline),

            dict(type='COCOCaption',
                data_root='data/coco_2014',
                ann_file='annotations/coco_karpathy_train.json',
                pipeline=caption_train_pipeline)
            ]),
            
    )

test_pipeline = ade20k_semseg_test_pipeline
val_dataloader =  dict(batch_size=1, 
        num_workers=1,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='ADE20KDataset',
            data_root='data/ade/ADEChallengeData2016',
            data_prefix=dict(
                img_path='images/validation',
                seg_map_path='annotations/validation'),
            return_classes=True,
            pipeline=ade20k_semseg_test_pipeline))
test_dataloader = val_dataloader

extra_val_dataloaders = [
    dict(batch_size=3,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root='data/coco/',
            ann_file='annotations/instances_val2017.json',
            data_prefix=dict(img='val2017/'),
            test_mode=True,
            return_classes=True,
            pipeline=coco_det_test_pipeline,
            backend_args=None)),
    dict(
        batch_size=3,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root='data/coco/',
            ann_file='annotations/instances_val2017.json',
            data_prefix=dict(img='val2017/'),
            test_mode=True,
            return_classes=True,
            pipeline=coco_insseg_test_pipeline,
            backend_args=backend_args)),
    dict(batch_size=16,
        num_workers=8,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='val',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='COCOCaption',
            data_root='data/coco_2014',
            ann_file='annotations/coco_karpathy_test.json',
            pipeline=caption_test_pipeline))
]
strategy = dict(
    type='DeepSpeedStrategy',
    bf16=dict(
        enabled=True,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=[0],
    zero_optimization=dict(
        stage=1,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=500000000,
        reduce_bucket_size=500000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False
        ),
    gradient_clipping=1.0,

)
import torch
# optimizer
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    # accumulative_counts=16,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.language_model.output.weight': dict(lr_mult=1.0),
            'backbone.language_model.model.norm.weight': dict(lr_mult=1.0),
            'backbone.language_model.model.tok_embeddings.weight': dict(lr_mult=1.0),
            'backbone.language_model.model.layers.0': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.1': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.2': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.3': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.4': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.5': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.6': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.7': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.8': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.9': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.10': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.11': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.12': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.13': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.14': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.15': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.16': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.17': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.18': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.19': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.20': dict(lr_mult=0.1),
            'backbone.language_model.model.layers.21': dict(lr_mult=0.175),
            'backbone.language_model.model.layers.22': dict(lr_mult=0.25),
            'backbone.language_model.model.layers.23': dict(lr_mult=0.325),
            'backbone.language_model.model.layers.24': dict(lr_mult=0.4),
            'backbone.language_model.model.layers.25': dict(lr_mult=0.475),
            'backbone.language_model.model.layers.26': dict(lr_mult=0.55),
            'backbone.language_model.model.layers.27': dict(lr_mult=0.625),
            'backbone.language_model.model.layers.28': dict(lr_mult=0.7),
            'backbone.language_model.model.layers.29': dict(lr_mult=0.775),
            'backbone.language_model.model.layers.30': dict(lr_mult=0.85),
            'backbone.language_model.model.layers.31': dict(lr_mult=0.925),
        })
        )
runner_type = 'FlexibleRunner'

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='CocoMetric',
        ann_file='data/coco/' + 'annotations/instances_val2017.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args),
    dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        backend_args=None,
        format_only=False,
        metric=[
            'segm',
        ],
        type='CocoMetric'),
    
    dict(type='VisualGroundingMetric'),
    dict(type='COCOCaption',
        ann_file='data/coco_2014/annotations/coco_karpathy_test_gt.json',)
]

# learning policy
max_iters=300000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=20000)
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

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (3 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1))
log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
