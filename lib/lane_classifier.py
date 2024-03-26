import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import sys
sys.path.append("../Lane_Detection/Lane_Classsification/")
from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names


# args
# input, threshold, weights, show, no-boxes
# model_file = outputs/training/road_line/model_15.pth
# show - default show_true 
# no-boxes - default store_true (--no_boxes)
# threshold - default 0.5
BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
MODEL_PATH = "Lane_Detection/Lane_Classsification/outputs/training/road_line/model_15.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(weights_path= BASE_PATH + MODEL_PATH):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False,
                                                                  num_classes=91)
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))
    # initialize the model
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt['model'])
    # load the modle on to the computation device and set to eval mode
    model.to(DEVICE).eval()
    
    return model 

def infer_image(model, image_path, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()]) 
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(DEVICE)
    
    masks, boxes, labels = get_outputs(image, model)    
    
    #result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
    
    return masks, boxes, labels

def main():
    model = load_model()
    image_path = BASE_PATH + "P3Data/test_video_frames/frame_0001.png"
    masks, boxes, labels = infer_image(model, image_path)
    print("Masks: ", masks)
    print("Boxes: ", boxes)
    print("Labels: ", labels)

if __name__ == '__main__':
    main()