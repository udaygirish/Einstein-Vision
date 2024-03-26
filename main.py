import os 
import sys 
import json
import glob 
import torch 
from PIL import Image 
import cv2 
import numpy as np
from lib.clrernet_lane_detect import inference_lanes as il_inf
# from lib.depth_anything import load_pipe as load_depth_pipe
# from lib.depth_anything import predict as predict_depth
from lib.zoe_depth import load_model as load_model_depth
from lib.zoe_depth import run_inference as run_inference_depth
from lib.zoe_depth import save_output as save_output_depth
from lib.lane_classifier import load_model as load_model_lane_classifier
from lib.lane_classifier import infer_image as infer_image_lane_classifier
# from lib.pose_2d import get_pose_2d as get_pose_2d
from lib.yolov8_det import load_model as load_model_det
from lib.yolov8_det import predict_image as predict_image_det
from lib.yolov8_pose import load_model as load_model_pose
from lib.yolov8_pose import predict_image as predict_image_pose
from lib.yolov8_seg import load_model as load_model_seg
from lib.yolov8_seg import predict_image as predict_image_seg




def single_image_pipeline(image_path):
    
    # Read image 
    image = Image.open(image_path).convert("RGB")

    # Get lane detection
    lanes = il_inf(image_path)
    print("Lanes: ", lanes)
    
    # Get Object Detection
    model_obj = load_model_det()
    obj_res, boxes_total, classes_total, scores_total, classes_names = predict_image_det(model_obj, image_path)
    
    # Get Pose Detection
    model_pose = load_model_pose()
    pose_res = predict_image_pose(model_pose, image_path)
    
    # Get depth
    model_depth = load_model_depth()
    depth = run_inference_depth(model_depth, image_path)
    
    # Get Segmentation
    model_seg = load_model_seg()
    seg_res = predict_image_seg(model_seg, image_path)
    
    # lane classification
    lane_class = load_model_lane_classifier()
    lane_masks, lane_boxes, lane_labels = infer_image_lane_classifier(lane_class, image_path)
    
    
    # TOtal results
    results = {
        'lanes': lanes,
        'object_detection': {
            'boxes': boxes_total,
            'classes': classes_total,
            'scores': scores_total,
            'classes_names': classes_names
        },
        'pose_detection': pose_res,
        'depth': depth,
        'segmentation': seg_res,
        'lane_masks': lane_masks,
        'lane_boxes': lane_boxes,
        'lane_labels': lane_labels
    }
    return results
    
def main():
    # Function to process a image
    # Get objects and save as JSON
    total_images = glob.glob("../P3Data/test_video_frames/*.png")
    print("Total Images: ", len(total_images))
    total_images = total_images[:1]
    result_dict = {}
    for image_path in total_images:
        result_dict['image_path'] = single_image_pipeline(image_path)
    
    print("====================================")
    print("Results: ", result_dict)
    print("====================================")
    # with open('results.json', 'w') as f:
    #     json.dump(result_dict, f)
    

if __name__ == '__main__':
    main()