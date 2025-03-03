_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

base_img_size = 672
# hyper parameter for each tasks
semseg_cfgs = dict(
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
                      pretrain_path='./ckpt/bert_embed.pt',vocab_size=30525),),
    head_list=dict(
        # non parametric task-specific heads
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
                    )),
    )

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
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

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=8, source_ratio=[1.], 
                 if_group=[False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette'],
        datasets=[
            dict(type='ADE20KDataset',
                data_root='data/ade/ADEChallengeData2016',
                data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                return_classes=True,
                pipeline=semseg_train_pipeline),
            ]),
            
    )

test_pipeline = semseg_test_pipeline
val_dataloader = dict(batch_size=4, # batch inference is not supported now.
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
            pipeline=semseg_test_pipeline))
test_dataloader = val_dataloader
import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    # optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
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
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=500))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
