_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth'

base_img_size = 1120
# hyper parameters for each tasks
det_cfgs = dict(
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    global_only_image=True)

insseg_cfgs = dict(
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    global_only_image=True)

semseg_cfgs = dict(
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    global_only_image=True)

caption_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False,
    global_only_image=False)

grounding_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False,
    global_only_image=False)

model = dict(
    type='UFO_ViT',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 'caption', 'grounding'],
    use_checkpoints=True,
    mean_output=True,
    mean_layes=[32,33,34,35,36,37],
    tokenizer=dict(type='BlipTokenizer', name_or_path='./ckpt/bert-base-uncased'),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=255,
        pad_size_divisor=224),
    backbone=dict(
        type='ViTUFO',
        arch='huge',
        img_size=base_img_size,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],  # win, global
        drop_path_rate=0.4,
        mean_layers=[32,33,34,35,36,37],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),
        text_cfg=dict(type='bert-base', hidden_size=1280, 
                      pretrain_path='./ckpt/bert_embed_huge.pt',vocab_size=30525),),
    head_list=dict(
        # non parametric task-specific heads
        detection_head=dict(type='UFOViTDetHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),])),
            test_cfg=dict(max_per_img=100),
            nms=dict(type='soft_nms', iou_threshold=0.5),
            repeat_times=3,
            # beam_num=3,
            # temperature=1.5,
            # alpha=10.0,
        ),
        instance_segmentation_head=dict(type='UFOViTInsSegHead',
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
                    # beam_num=3,
                    # temperature=1.5,
                    # alpha=10.0,
                    ),
        semantic_segmentation_head=dict(type='UFOViTSemSegHead',                    
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
                    ),
        caption_head=dict(type='UFOViTCaptionHead'),
        grounding_head=dict(type='UFOViTGroundHead')),
)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
## pipeline for detection
det_load_pipeline = [
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
det_train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

det_test_pipeline = [
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name')),]
## pipeline for instance segmentation
insseg_load_pipeline = [
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
insseg_train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name')),
    ]

insseg_test_pipeline = [
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name')),]
## pipeline for semantic segmentation
semseg_train_pipeline = [
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
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name'))]
semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 672), keep_ratio=False),
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
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg', 'dataset_name'))]
# pipeline for image caption
caption_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=30525,
                                                            num_vocal=30525,
                                                            max_length=20,
                                                            ignore_index=-100,
                                                            beam_num=2),
                                            git_cfg=caption_cfgs)),
    dict(type='RandomResizedCrop', scale=224, interpolation='bicubic', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='CleanCaption', keys='gt_caption'),
    dict(type='PackInputs', algorithm_keys=['gt_caption'], meta_keys=['image_id','img_shape', 'task_name', 'head_cfg', 'git_cfg'],),
]

caption_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=30525,
                                                            num_vocal=30525,
                                                            max_length=20,
                                                            ignore_index=-100,
                                                            beam_num=2,
                                                            temperature=1.0,
                                                            alpha=0.7),
                                            git_cfg=caption_cfgs)),
    dict(type='Resize', scale=(224, 224), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'img_shape', 'task_name', 'head_cfg', 'git_cfg']),
]
# pipeline for visual grounding
grounding_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=448+1,
                                            num_vocal=448+1,
                                            num_bins=448,
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
        scales=[(224, 224)],
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
                            head_cfg=dict(num_classes=448+1,
                                            num_vocal=448+1,
                                            num_bins=448,
                                            max_length=20),
                            git_cfg=grounding_cfgs)),
    dict(type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', ],
        meta_keys=['image_id','img_shape','scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=1, source_ratio=[0.2, 0.2, 0.2, 0.2, 
            0.2/3., 0.2/3., 0.2/3.], if_group=[True, True, False, False,False, False, False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette'],
        datasets=[
            dict(
                type='MultiImageMixDataset',
                dataset=dict(type='CocoDataset',
                    data_root='data/coco/',
                    ann_file='annotations/instances_train2017.json',
                    data_prefix=dict(img='train2017/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=True,
                    pipeline=det_load_pipeline,
                    backend_args=backend_args),
                pipeline=det_train_pipeline,
            ),
            dict(
                type='MultiImageMixDataset',
                dataset=dict(type='CocoDataset',
                    data_root='data/coco/',
                    ann_file='annotations/instances_train2017.json',
                    data_prefix=dict(img='train2017/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=True,
                    pipeline=insseg_load_pipeline,
                    backend_args=backend_args),
                pipeline=insseg_train_pipeline,
            ),
            dict(type='ADE20KDataset',
                data_root='data/ade/ADEChallengeData2016',
                data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                return_classes=True,
                pipeline=semseg_train_pipeline),
            dict(type='COCOCaption',
                data_root='data/coco_2014',
                ann_file='annotations/coco_karpathy_train.json',
                pipeline=caption_train_pipeline),
            dict(type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco/instances.json',
                split_file='refcoco/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco+/instances.json',
                split_file='refcoco+/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcocog/instances.json',
                split_file='refcocog/refs(umd).p',
                split='train',
                pipeline=grounding_train_pipeline),
            ]),     
    )

val_dataloader = dict(batch_size=1,
        num_workers=4,
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
            pipeline=det_test_pipeline,
            backend_args=None))
test_dataloader = val_dataloader

# construct extra val dataloaders
extra_val_dataloaders = [
    dict(batch_size=1,
        num_workers=4,
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
            pipeline=insseg_test_pipeline,
            backend_args=None)),
    dict(batch_size=16,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='COCOCaption',
            data_root='data/coco_2014',
            ann_file='annotations/coco_karpathy_test.json',
            pipeline=caption_test_pipeline)),
    dict(batch_size=1, 
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='ADE20KDataset',
            data_root='data/ade/ADEChallengeData2016',
            data_prefix=dict(
                img_path='images/validation',
                seg_map_path='annotations/validation'),
            return_classes=True,
            pipeline=semseg_test_pipeline)),
    dict(batch_size=16,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='val',
            pipeline=grounding_test_pipeline))
]


import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.embed': dict(lr_mult=1.0),
            'backbone.layers.0': dict(lr_mult=0.1),
            'backbone.layers.1': dict(lr_mult=0.1),
            'backbone.layers.2': dict(lr_mult=0.1),
            'backbone.layers.3': dict(lr_mult=0.1),
            'backbone.layers.4': dict(lr_mult=0.1),
            'backbone.layers.5': dict(lr_mult=0.1),
            'backbone.layers.6': dict(lr_mult=0.1),
            'backbone.layers.7': dict(lr_mult=0.1),
            'backbone.layers.8': dict(lr_mult=0.1),
            'backbone.layers.9': dict(lr_mult=0.1),
            'backbone.layers.10': dict(lr_mult=0.1),
            'backbone.layers.11': dict(lr_mult=0.1),
            'backbone.layers.12': dict(lr_mult=0.1),
            'backbone.layers.13': dict(lr_mult=0.1),
            'backbone.layers.14': dict(lr_mult=0.1),
            'backbone.layers.15': dict(lr_mult=0.1),
            'backbone.layers.16': dict(lr_mult=0.1),
            'backbone.layers.17': dict(lr_mult=0.15625),
            'backbone.layers.18': dict(lr_mult=0.2125),
            'backbone.layers.19': dict(lr_mult=0.26875),
            'backbone.layers.20': dict(lr_mult=0.325),
            'backbone.layers.21': dict(lr_mult=0.38125),
            'backbone.layers.22': dict(lr_mult=0.4375),
            'backbone.layers.23': dict(lr_mult=0.49375),
            'backbone.layers.24': dict(lr_mult=0.55),
            'backbone.layers.25': dict(lr_mult=0.60625),
            'backbone.layers.26': dict(lr_mult=0.6625),
            'backbone.layers.27': dict(lr_mult=0.71875),
            'backbone.layers.28': dict(lr_mult=0.7750),
            'backbone.layers.29': dict(lr_mult=0.83125),
            'backbone.layers.30': dict(lr_mult=0.8875),
            'backbone.layers.31': dict(lr_mult=0.94375),
            'backbone.layers.32': dict(lr_mult=1.0),
            'backbone.layers.33': dict(lr_mult=1.0),
            'backbone.layers.34': dict(lr_mult=1.0),
            'backbone.layers.35': dict(lr_mult=1.0),
            'backbone.layers.36': dict(lr_mult=1.0),
            'backbone.layers.37': dict(lr_mult=1.0),
        }))

val_evaluator = dict(type='CocoMetric',
        ann_file='data/coco/' + 'annotations/instances_val2017.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args)

test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='CocoMetric', 
        ann_file='data/coco/annotations/instances_val2017.json',
        backend_args=None,
        format_only=False,
        metric=['segm']),
    dict(type='COCOCaption',
        ann_file='data/coco_2014/annotations/coco_karpathy_test_gt.json',),
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='VisualGroundingMetric'),
]

# learning policy
max_iters=640000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=640000)
val_cfg = dict(type='MultiSourceValLoop', extra_dataloaders=extra_val_dataloaders, extra_evaluators=extra_val_evaluators)
test_cfg = val_cfg
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.1,
         begin=0,
         end=5000,
         by_epoch=False),
    dict(
          type='CosineAnnealingLR',
          T_max=max_iters,
          eta_min=2e-6,
          begin=5000,
          end=max_iters,
          by_epoch=False,)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (3 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,show=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
