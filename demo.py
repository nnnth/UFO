
from mmdet.apis import DetInferencer
from mmdet.apis import MMSegInferencer
import os
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='UFO demo'
    )

    # Required arguments
    parser.add_argument(
        '--img_path',
        type=str,
        required=True,
        help='Path to the input image.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to the checkpoint file.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Directory for the output image.'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Name of the task.'
    )

    # Optional arguments
    parser.add_argument(
        '--text',
        type=str,
        default='',
        help='Input text. Defaults to an empty string.'
    )
    parser.add_argument(
        '--is_sentence',
        type=bool,
        default=None,
        help='Set to True if the input is a complete sentence. Set to False if it is a phrase.'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Print received arguments 
    print("Configuration file path:", args.config)
    print("Checkpoint path:", args.ckpt_path)
    print("Task name:", args.task)
    if args.text != '':
        print("Input text:", args.text)
    if args.is_sentence is not None:
        print("Is complete sentence:", args.is_sentence)
    
    os.makedirs(args.out_dir, exist_ok=True)

    if args.task in ['detection', 'ins_seg']:
        inferencer = DetInferencer(args.config, weights=args.ckpt_path, scope='mmdet')
        output = inferencer(args.img_path, return_vis=True, show=False, out_dir=args.out_dir)
    elif args.task in ['sem_seg']:
        inferencer = MMSegInferencer(args.config, weights=args.ckpt_path, scope='mmdet')
        output = inferencer(args.img_path, return_vis=True, show=False, out_dir=args.out_dir)
    elif args.task in ['rec', 'res', 'reason_seg']:
        assert args.text != '', f'Please input a prompt for the task: {args.task}'
        
        inferencer = DetInferencer(args.config, weights=args.ckpt_path, scope='mmdet')
        if args.task == 'reason_seg':
            assert args.is_sentence is not None, 'Please specify whether the prompt is a complete sentence.'
            output = inferencer(args.img_path, text=args.text, is_sentence=args.is_sentence, return_vis=True, show=False, out_dir=args.out_dir)
        else:
            output = inferencer(args.img_path, text=args.text, return_vis=True, show=False, out_dir=args.out_dir)
    print(output)
    print("Done!")

if __name__ == '__main__':
    main()
