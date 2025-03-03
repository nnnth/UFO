_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
base_img_size = 336
load_from = './ufo-llava-1.5-7b-instruction.pth'
# hyper parameter for each tasks
grounding_cfgs = dict(
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False)

model = dict(
    type='UFO_LLaVA',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 
    'caption', 'grounding','refer_segmentation','vqa', 'refer_caption','reason_segmentation'],
    use_checkpoints=True,
    lora_r=8,
    lora_alpha=2*8,
    tokenizer=dict(type='AutoTokenizer', 
        name_or_path='./ckpt/llava-1.5-7b-hf',
        trust_remote_code=True,
        use_fast=False),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=255,
        pad_size_divisor=336),
    backbone=dict(
        type='LlavaForConditionalGeneration',
        name_or_path='./ckpt/llava-1.5-7b-hf',
        trust_remote_code=True,
        attn_implementation='eager',
        ),
    head_list=dict( 
        # non parametric task-specific heads
        grounding_head=dict(type='UFOLLaVAGroundHead'),
        ),
    )

# grounding pipeline
grounding_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=base_img_size*2+1,
                                            num_vocal=base_img_size*2+1,
                                            num_bins=base_img_size*2,
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
        scales=[(336, 336)],
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
                            head_cfg=dict(num_classes=base_img_size*2+1,
                                            num_vocal=base_img_size*2+1,
                                            num_bins=base_img_size*2,
                                            max_length=20),
                            git_cfg=grounding_cfgs)),
    dict(type='Resize',
        scale=(336, 336),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', ],
        meta_keys=['image_id','img_shape','scale_factor','task_name', 'head_cfg', 'git_cfg','img_path'],
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=4, source_ratio=[1/3,1/3,1/3], 
                 if_group=[False,False,False,], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
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
            ]),
            
    )

test_pipeline = grounding_test_pipeline
val_dataloader = dict(batch_size=16,
        num_workers=1,
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
            pipeline=grounding_test_pipeline))
test_dataloader = val_dataloader
extra_val_dataloaders = [
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='testA',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='testB',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco+/instances.json',
            split_file='refcoco+/refs(unc).p',
            split='val',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco+/instances.json',
            split_file='refcoco+/refs(unc).p',
            split='testA',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco+/instances.json',
            split_file='refcoco+/refs(unc).p',
            split='testB',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcocog/instances.json',
            split_file='refcocog/refs(umd).p',
            split='val',  # or 'testB'
            pipeline=grounding_test_pipeline)),
    dict(batch_size=16,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcocog/instances.json',
            split_file='refcocog/refs(umd).p',
            split='test',  # or 'testB'
            pipeline=grounding_test_pipeline)),
]
import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=16,)

val_evaluator = dict(type='VisualGroundingMetric')

test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric'),
    dict(type='VisualGroundingMetric')
]

# learning policy
max_iters=20000
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
