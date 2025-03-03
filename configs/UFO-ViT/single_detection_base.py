_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

base_img_size = 1120
# hyper parameter for each tasks
det_cfgs = dict(
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    global_only_image=True)

model = dict(
    type='UFO_ViT',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 'caption', 'grounding'],
    use_checkpoints=True,
    mean_output=True,
    mean_layes=[12,13,14,15,16,17],
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
        arch='base',
        img_size=base_img_size,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],  # win, global
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),
        text_cfg=dict(type='bert-base', hidden_size=768, 
                      pretrain_path='./ckpt/bert_embed.pt'),),
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
        )),
    )

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
## pipeline for detection
load_pipeline = [
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=3, source_ratio=[1.], 
                 if_group=[True], shuffle=True),
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
                    pipeline=load_pipeline,
                    backend_args=backend_args),
                pipeline=det_train_pipeline,
            ),
            
            ]),
            
    )

test_pipeline = det_test_pipeline
val_dataloader = dict(batch_size=1,
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
            pipeline=det_test_pipeline,
            backend_args=None))
test_dataloader = val_dataloader
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
            'backbone.layers.6': dict(lr_mult=0.2286),
            'backbone.layers.7': dict(lr_mult=0.3571),
            'backbone.layers.8': dict(lr_mult=0.4858),
            'backbone.layers.9': dict(lr_mult=0.6143),
            'backbone.layers.10': dict(lr_mult=0.7429),
            'backbone.layers.11': dict(lr_mult=0.8714),
            'backbone.layers.12': dict(lr_mult=1.0),
            'backbone.layers.13': dict(lr_mult=1.0),
            'backbone.layers.14': dict(lr_mult=1.0),
            'backbone.layers.15': dict(lr_mult=1.0),
            'backbone.layers.16': dict(lr_mult=1.0),
            'backbone.layers.17': dict(lr_mult=1.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

val_evaluator = dict(type='CocoMetric',
        ann_file='data/coco/' + 'annotations/instances_val2017.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args)
test_evaluator = val_evaluator

# learning policy
max_iters=120000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

param_scheduler = [dict(
          type='CosineAnnealingLR',
          T_max=max_iters,
          eta_min=2e-6,
          begin=0,
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
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)