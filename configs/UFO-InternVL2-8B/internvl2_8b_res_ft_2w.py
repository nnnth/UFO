_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
base_img_size = 448
load_from = './ufo-internvl2-8b-instruction.pth'
# hyper parameter for each tasks
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
        refer_segmentation_head=dict(type='UFOInternVLReferSegHead',
                            mask_token_id=92553,
                            cls_loss_weight=1.0,),
        ),
    )

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

train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=8, source_ratio=[1.0], 
                 if_group=[False,], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
            # refer segmentation 
            dict(type='ReferSegDataset',
                data_root='data/refer_seg',
                split='train',
                pipeline=referseg_train_pipeline),

            ]),
            
    )

test_pipeline = referseg_test_pipeline
val_dataloader = dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='val',
            refer_seg_data='refcoco',
            pipeline=referseg_test_pipeline))
test_dataloader = val_dataloader
extra_val_dataloaders = [
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='testA',
            refer_seg_data='refcoco',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='testB',
            refer_seg_data='refcoco',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='val',
            refer_seg_data='refcoco+',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='testA',
            refer_seg_data='refcoco+',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='testB',
            refer_seg_data='refcoco+',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='val',
            refer_seg_data='refcocog',
            pipeline=referseg_test_pipeline)),
    dict(batch_size=8,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReferSegDataset',
            data_root='data/refer_seg',
            split='test',
            refer_seg_data='refcocog',
            pipeline=referseg_test_pipeline)),
]
import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=16,)

val_evaluator = dict(type='IoUMetricBinary', iou_metrics=['mIoU'])

test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
]

# learning policy
max_iters=20000
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

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=10))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
