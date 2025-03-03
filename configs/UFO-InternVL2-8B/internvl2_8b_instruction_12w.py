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
        detection_head=dict(type='UFOInternVLDetHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),])),
            test_cfg=dict(max_per_img=100),
            nms=dict(type='soft_nms', iou_threshold=0.5),
            repeat_times=3,
            sample_prob=1.0,
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
        grounding_head=dict(type='UFOInternVLGroundHead'),
        vqa_head=dict(type='UFOInternVLVQAHead',
                            template_name='internlm2-chat'),
        refer_segmentation_head=dict(type='UFOInternVLReferSegHead',
                            mask_token_id=92553,
                            cls_loss_weight=1.0,),
        reason_segmentation_head=dict(type='UFOInternVLReasonSegHead',
                            mask_token_id=92553,),
        ),
    )

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# det pipeline
obj365_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            dataset_name='obj365',
                                            head_cfg=dict(num_classes=365,
                                                          num_vocal=(base_img_size*2 + 1) + 365 + 1,
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

oim_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            dataset_name='oim',
                                            head_cfg=dict(num_classes=601,
                                                          num_vocal=(base_img_size*2 + 1) + 601 + 1,
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

lvisv1_det_train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            dataset_name='lvis',
                                            head_cfg=dict(num_classes=1203,
                                                          num_vocal=(base_img_size*2 + 1) + 1203 + 1,
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

nuimage_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection',
                                            dataset_name='nuimage_det', 
                                            head_cfg=dict(num_classes=10,
                                                        num_vocal=10+1+base_img_size*2+1,
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

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

# instance segmentation pipeline
lvisv1_insseg_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',
                                            dataset_name='lvis', 
                                            head_cfg=dict(num_classes=1203,
                                                          num_vocal=(base_img_size*2 + 1) + 1203 + 1,
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
oim_insseg_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',
                                            dataset_name='oim', 
                                            head_cfg=dict(num_classes=601,
                                                          num_vocal=(base_img_size*2 + 1) + 601 + 1,
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
nuimage_insseg_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            dataset_name='nuimage_ins',
                                            head_cfg=dict(num_classes=10,
                                                        num_vocal=10+1+base_img_size*2+1,
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
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg','dataset_name')),]

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

nuimage_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            dataset_name='nuimage_seg',
                                            head_cfg=dict(num_classes=31,
                                                            num_vocal=32,
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

# grounding pipeline
grounding_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=896+1,
                                            num_vocal=896+1,
                                            num_bins=896,
                                            max_length=20),
                            git_cfg=grounding_cfgs)),
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

# reason segmentation pipeline
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
    # dict(type='CleanCaption', keys='text'),
    dict(type='PackInputs',
        algorithm_keys=['text', 'gt_masks', ],
        meta_keys=['image_id','img_shape','ori_shape','scale_factor','task_name', 'head_cfg', 'git_cfg','img_path','is_sentence','explain_text'],
    ),
]

det_ratio = 1/6
insseg_ratio = 1/6
semseg_ratio = 1/6
referseg_ratio = 1/6
vqa_ratio = 1/6
ref_ground_ratio = 1/6

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=4, source_ratio=[
                                                                                    det_ratio*1/5, det_ratio*1/5, det_ratio*1/5, det_ratio*1/5, det_ratio*1/5,
                                                                                    insseg_ratio*1/6, insseg_ratio*1/6, insseg_ratio*1/6, insseg_ratio*1/6, insseg_ratio*1/6, insseg_ratio*1/6,
                                                                                    semseg_ratio*1/4, semseg_ratio*1/4, semseg_ratio*1/4,  semseg_ratio*1/4, 
                                                                                    referseg_ratio,
                                                                                    vqa_ratio,
                                                                                    ref_ground_ratio*1/4, ref_ground_ratio*1/4, ref_ground_ratio*1/4, ref_ground_ratio*1/4
                                                                                    ], 
                 if_group=[True, True, True, True, True,
                            True, True, True, True, True, True,
                            False,False,False,False,
                            False,
                            False,
                            False,False,False,False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
            # detection: object365, openimage, lvisv1,nuimages, coco, v3det
            dict(type='Objects365V2Dataset',
                data_root='data/Objects365/Obj365_v2/',
                ann_file='annotations/zhiyuan_objv2_train.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                return_classes=False,
                pipeline=obj365_det_train_pipeline,
                backend_args=backend_args),
            dict(type='ClassBalancedDataset',
                oversample_thr=1./601,
                dataset=dict(type='OpenImagesDataset',
                    data_root='data/OpenImages/',
                    ann_file='annotations/oidv6-train-annotations-bbox.csv',
                    data_prefix=dict(img='OpenImages/train/'),
                    label_file='annotations/class-descriptions-boxable.csv',
                    hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
                    meta_file='annotations/train-image-metas.pkl',
                    return_classes=False,
                    pipeline=oim_det_train_pipeline,
                    backend_args=backend_args)),
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                    type='LVISV1Dataset',
                    data_root='data/lvis_v1/',
                    ann_file='annotations/lvis_v1_train.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=lvisv1_det_train_pipeline,
                    backend_args=backend_args)),
            dict(
                type='NuimageDataset',
                ann_file='data/nuimages/' + 'annotations/nuimages_v1.0-train.json',
                data_prefix=dict(img='data/nuimages/'),
                return_classes=False,
                pipeline=nuimage_det_train_pipeline),
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
            # instance segmentation: lvis, openimage, nuimage, coco, paco-lvis, pascal-part
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                    type='LVISV1Dataset',
                    data_root='data/lvis_v1/',
                    ann_file='annotations/lvis_v1_train.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    return_classes=False,
                    pipeline=lvisv1_insseg_train_pipeline,
                    backend_args=backend_args)),
            dict(type='ClassBalancedDataset',
                oversample_thr=1./601,
                dataset=dict(
                    type='OpenImagesDatasetInseg',
                    data_root='data/OpenImages/',
                    ann_file='annotations/train-annotations-object-segmentation_sort_resize.csv',
                    data_prefix=dict(img='OpenImages/train/',seg='segmentation/train'),
                    label_file='annotations/class-descriptions-boxable.csv',
                    hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
                    meta_file='annotations/train-image-metas-dict.pkl',
                    return_classes=False,
                    pipeline=oim_insseg_train_pipeline,
                    backend_args=backend_args)),
            dict(type='NuimageDataset',
                ann_file='data/nuimages/' + 'annotations/nuimages_v1.0-train.json',
                data_prefix=dict(img='data/nuimages/'),
                return_classes=False,
                pipeline=nuimage_insseg_train_pipeline),
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
            # semantic segmentation coco_stuff164k, nuimage_seg, Mapillary V2, ade20k
            dict(type='COCOStuffDataset',
                data_root='data/coco_stuff164k',
                data_prefix=dict(
                    img_path='images/train2017', seg_map_path='annotations/train2017'),
                return_classes=False,
                pipeline=cocostuff_semseg_train_pipeline),
            dict(type='NuimageSegDataset',
                data_root='data/nuimages_seg',
                data_prefix=dict(
                    img_path='images/training', seg_map_path='annotations/training'),
                return_classes=False,
                pipeline=nuimage_semseg_train_pipeline),
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
            dict(type='RefCOCO',
                data_root='data/refclef',
                data_prefix='saiapr_tc-12',
                ann_file='instances.json',
                split_file='refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            ]),
            
    )

test_pipeline = referseg_test_pipeline
val_dataloader = dict(batch_size=4,
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
    dict(batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|val",
            explanatory=-1,
            pipeline=reasonseg_test_pipeline)),
    dict(batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ReasonSegDataset',
            data_root='data/',
            reason_seg_data="ReasonSeg|test",
            explanatory=-1,
            pipeline=reasonseg_test_pipeline)),
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
            split='val',  # or 'testB'
            pipeline=grounding_test_pipeline)),
]
import torch
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=16,
    )

val_evaluator = dict(type='IoUMetricBinary', iou_metrics=['mIoU'])

test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='IoUMetricBinary', iou_metrics=['mIoU']),
    dict(type='VisualGroundingMetric'),
]

# learning policy
max_iters=120000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=10000)
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
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=10))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
