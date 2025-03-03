_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
base_img_size = 1344
# hyper parameter for each tasks
det_cfgs = dict(
    grid_resolution_perwin=[8, 8],
    samples_grids_eachwin=1,
    grid_interpolate=True)

insseg_cfgs = dict(
    grid_resolution_perwin=[8, 8],
    samples_grids_eachwin=1,
    grid_interpolate=True)


model = dict(
    type='UFO_InternVL_Full',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 
    'caption', 'grounding'],
    use_checkpoints=True,
    pretrain_path='./ufo-internvl2-8b-multi-1344.pth',
    repeat_grid=True,
    repeat_num=3,
    keep_num=300,
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
            test_cfg=dict(max_per_img=300),
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
                test_cfg=dict(max_per_img=300),
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



train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=1, source_ratio=[
                                                                                    1/2, 1/2
                                                                                    ], 
                 if_group=[True,True], shuffle=True),
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
            ]),
            
    )

test_pipeline = coco_det_test_pipeline
val_dataloader =  dict(batch_size=3,
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
            backend_args=None))
test_dataloader = val_dataloader

extra_val_dataloaders = [
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
]
strategy = dict(
    type='DeepSpeedStrategy',
    gradient_accumulation_steps=3,
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
        stage=2,
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
    optimizer=dict(type='AdamW', lr=0.000000002, weight_decay=0.01),
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

val_evaluator = dict(type='CocoMetric',
        ann_file='data/coco/' + 'annotations/instances_val2017.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args)
test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        backend_args=None,
        format_only=False,
        metric=[
            'segm',
        ],
        type='CocoMetric'),
]

# learning policy
max_iters=1
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1)
val_cfg = dict(type='MultiSourceValLoop', extra_dataloaders=extra_val_dataloaders, extra_evaluators=extra_val_evaluators)
test_cfg = val_cfg

param_scheduler = [
    dict(type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1,
        by_epoch=False)
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
