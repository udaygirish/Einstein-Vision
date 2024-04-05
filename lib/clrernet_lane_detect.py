# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys 
sys.path.append("../Lane_Detection/CLRerNet/")
sys.path.append("../Lane_Detection/CLRerNet/libs")

from mmdet.apis import init_detector

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes


def inference_lanes(img_path):
    base_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/Lane_Detection/CLRerNet"
    config_path = f"{base_path}/configs/clrernet/culane/clrernet_culane_dla34.py"
    checkpoint_path = f"{base_path}/pretrained/clrernet_culane_dla34.pth"
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    src, preds = inference_one_image(model, img_path)
    return src, preds
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    src, preds = inference_one_image(model, args.img)
    # show the results
    print("Predicted lanes:", preds)
    dst = visualize_lanes(src, preds, save_path=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
