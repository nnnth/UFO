### Training
#### Single Task (UFO-ViT-B)
Detection

```shell
bash tools/dist_train.sh configs/UFO-ViT/single_detection_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Semantic Segmentation

```shell
bash tools/dist_train.sh configs/UFO-ViT/single_semseg_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Instance Segmentation

```shell
bash tools/dist_train.sh configs/UFO-ViT/single_insseg_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Image Caption

```shell
bash tools/dist_train.sh configs/UFO-ViT/single_caption_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

Visual Grounding

```shell
bash tools/dist_train.sh configs/UFO-ViT/single_ground_base.py ${GPU_NUM} --work-dir ${work_dir}
```

#### Multi Task (UFO-ViT)

UFO-ViT-B

```shell
bash tools/dist_train.sh configs/UFO-ViT/multi_fivetask_base.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-ViT-L

```shell
bash tools/dist_train.sh configs/UFO-ViT/multi_fivetask_large.py ${GPU_NUM} --work-dir ${work_dir}
```

UFO-ViT-H

```shell
bash tools/dist_train.sh configs/UFO-ViT/multi_fivetask_huge.py ${GPU_NUM} --work-dir ${work_dir}
```

#### Multi Task (UFO-InternVL2_5-8B)
448x448

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B-multitask/multi_448_30w.py  ${GPU_NUM} --work-dir ${work_dir}
```

896x896

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B-multitask/multi_896_6w.py  ${GPU_NUM} --work-dir ${work_dir}
```

#### Instruction Tuning

UFO-InternVL2_5-8B 448x448

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_instruction_12w.py  ${GPU_NUM} --work-dir ${work_dir}
```

Copy the checkpoint for 448x448 resolution as `ufo-internvl2_5-8b-instruction.pth` in root dir, then run next stage:

UFO-InternVL2_5-8B 896x896

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_instruction_896_4w.py  ${GPU_NUM} --work-dir ${work_dir}
```


UFO-LLaVA-1.5-7B

```shell
bash tools/dist_train.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_instruction_12w.py ${GPU_NUM} --work-dir ${work_dir}
```

#### Specific finetuning
Please download instruction tuning weight from [huggingface](https://huggingface.co/kanashi6/UFO/tree/main) and organize files as follows:

```
UFO
|──ufo-internvl2_5-8b-instruction-896.pth
|——ufo-llava1.5-7b-instruction.pth
```

UFO-InternVL2_5-8B, REC
```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_rec_ft_2w.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-InternVL2_5-8B, RES
```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_res_ft_2w.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-InternVL2_5-8B, ReasonSeg
```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_reasonseg_ft_1w.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, REC
```shell
bash tools/dist_train.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_rec_ft_2w.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, RES
```shell
bash tools/dist_train.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_res_ft_2w.py  ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, ReasonSeg
```shell
bash tools/dist_train.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_reasonseg_ft_1w.py  ${GPU_NUM} --work-dir ${work_dir}
```
#### DeepSpeed Training
If you encounter an out-of-memory issue during experiments, you can use DeepSpeed to reduce GPU memory usage. 

```shell
pip install deepspeed==0.15.4
```

For specific configurations, refer to `llava1.5_7b_instruction_12w_deepspeed.py`. The larger the stage parameter, the less GPU memory is consumed. However, setting stage=3 will significantly slow down the training speed, so stage=1 or stage=2 are recommended.

After training is complete, if you need to merge the sharded checkpoints, follow these steps: 
1. Create a `latest` file in the working directory. The content of this latest file should be the name of the most recent checkpoint folder. 
2. Create a directory to save merged checkpoint. 
3. Run the `zero_to_fp32.py ` script located in your working directory.

### Testing

#### Single Task (UFO-ViT-B)
Detection

```shell
bash tools/dist_test.sh configs/UFO-ViT/single_detection_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Semantic Segmentation

```shell
bash tools/dist_test.sh configs/UFO-ViT/single_semseg_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Instance Segmentation

```shell
bash tools/dist_test.sh configs/UFO-ViT/single_insseg_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Image Caption

```shell
bash tools/dist_test.sh configs/UFO-ViT/single_caption_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

Visual Grounding

```shell
bash tools/dist_test.sh configs/UFO-ViT/single_ground_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```
#### Beam Search
We use beam search for better performance on COCO detection and instance segmentation. To switch on beam search, please remove the comments in config:
```
detection_head=dict(type='UFOViTDetHead',
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),])),
    test_cfg=dict(max_per_img=100),
    nms=dict(type='soft_nms', iou_threshold=0.5),
    repeat_times=3,
    # use beam search here
    # beam_num=3,
    # temperature=1.5,
    # alpha=10.0,
)
```

#### Multi Task (UFO-ViT)

UFO-ViT-B

```shell
bash tools/dist_test.sh configs/UFO-ViT/multi_fivetask_base.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

UFO-ViT-L
```shell
bash tools/dist_test.sh configs/UFO-ViT/multi_fivetask_large.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

UFO-ViT-H

```shell
bash tools/dist_test.sh configs/UFO-ViT/multi_fivetask_huge.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

#### Multi-Task (UFO-InternVL2_5-8B)
As there is a bug in direct evaluation using deepspeed, please set checkpoint path for `pretrain_path` variable in the test config, then execute evaluation in training.

448x448

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B-multitask/multi_448_30w_test.py  ${GPU_NUM} --work-dir ${work_dir}
```

896x896

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B-multitask/multi_896_6w_test.py  ${GPU_NUM} --work-dir ${work_dir}
```

Another option is to setting the zero stage to 0 if no OOM problem, which is mentioned in [issue](https://github.com/nnnth/UFO/issues/42). Thanks for the contribution for [shuzhangcasia](https://github.com/shuzhangcasia).

#### Instruction tuning
UFO-InternVL2_5-8B, REC

```shell
bash tools/dist_test.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_rec_ft_2w.py ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

For res and reasonseg, we use 896x896 for test, which requires deepspeed to avoid memory issues. Similar to the test of Multi-task above, we use evaluation in training. Please set checkpoint path for `pretrain_path` variable in the test config, then execute evaluation in training:

UFO-InternVL2_5-8B, RES

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_res_ft_2w_test.py ${GPU_NUM} --work-dir ${work_dir}
```

UFO-InternVL2_5-8B, ReasonSeg

```shell
bash tools/dist_train.sh configs/UFO-InternVL2_5-8B/internvl2_5_8b_reasonseg_ft_1w.py ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, REC
```shell
bash tools/dist_test.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_rec_ft_2w.py  ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, RES
```shell
bash tools/dist_test.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_res_ft_2w.py  ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```

UFO-LLaVA1.5-7B, ReasonSeg
```shell
bash tools/dist_test.sh configs/UFO-LLaVA1.5-7B/llava1.5_7b_reasonseg_ft_1w.py  ${ckpt_file} ${GPU_NUM} --work-dir ${work_dir}
```