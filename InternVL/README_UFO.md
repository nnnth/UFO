# UFO: A Unified Approach to Fine-grained Visual Perception via Open-ended Language Interface
We have migrated the REC inference from UFO to the InternVL repository. The specific command are as follows:

Download checkpoints from [UFO-InternVL2-8B-instruct](https://huggingface.co/kanashi6/UFO-InternVL2-8B-instruct) or [UFO-InternVL2-8B-rec-ft](https://huggingface.co/kanashi6/UFO-InternVL2-8B-rec-ft), put them in `UFO/InternVL/pretrained`.

(Optional) If you have trained LoRA weights using mmdetection, you can use `UFO/InternVL/internvl_chat/tools/merge_lora_custom.py` to convert them.
```shell
python tools/merge_lora_custom.py ../pretrained/InternVL2-8B ../pretrained/UFO-InternVL2-8B-instruct ../../ckpt/ufo-internvl2-8b-instruction.pth
```

Then run commands to evaluate on RefCOCO series. Following command is for RefCOCO val:

```shell
GPUS=8 bash evalulate_ufo.sh ../pretrained/UFO-InternVL2-8B-instruct refcoco-val
```

**Please note that the current InternVL repository does not support perception tasks beyond REC (such as RES and inferential segmentation). We will migrate the code as soon as possible.**
