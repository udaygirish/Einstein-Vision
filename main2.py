import os 
import sys 
import json
import glob 
import torch 
from PIL import Image 
import cv2 
import numpy as np
from tqdm import tqdm, trange
import time 
import pickle
import sys 
from tqdm import tqdm 

from lib.yolov3d_infer import detect3d # Have to work on the DLANet already registered error
# Solved the above issue as loading yolo3d first works better
# it is following absolute paths 
# also np.float alias issue
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
from lib.lane_class_matcher import lane_class_matcher
from lib.traffic_sign_thresholder import *
from lib.optical_flow import *
from lib.vehicle_indicators import *

from utilities.cv2_utilities import *
from utilities.blender_utils import *
from utilities.three_d_utils import *
from utilities.random_utils import *


# Have to check how the Human in Blender thing works 

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
def load_yolo3d_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def single_image_pipeline(image_path):
    
    # Read image 
    image = Image.open(image_path).convert("RGB")

    # Get lane detection
    src, lanes = il_inf(image_path)
    
    
    # lane classification
    lane_class = load_model_lane_classifier()
    lane_masks, lane_boxes, lane_labels = infer_image_lane_classifier(lane_class, image_path)
    
    
    # # TOtal results
    results = {
        'lanes': lanes,
        'lane_masks': lane_masks,
        'lane_boxes': lane_boxes,
        'lane_labels': lane_labels,
    }

    try :
        final_lanes = lane_class_matcher(results)
        results['final_lanes'] = final_lanes
    except Exception as e:
        results['final_lanes'] = []
        pass

    return results
    
def main():
    # Function to process a image
    # Get objects and save as JSON
    for folderNum in range(1,14) :
        total_images = glob.glob(f"../P3Data/FinalImages/Images_{folderNum}/*.jpg")
        total_images = total_images
        total_images = sorted(total_images)
        result_dict = {}
    
        for i in tqdm(range(len(total_images))): # 11 to 13
            img1_path = total_images[i]
            temp_results = single_image_pipeline(img1_path)
            result_dict[img1_path] = temp_results
        
        
        video_name = f"video{folderNum}"
        # Dump the results to pickle
        with open(BASE_PATH+ "P3Data/laneData/results_{}.pkl".format(video_name), 'wb') as f:
            pickle.dump(result_dict, f)
    
    

if __name__ == '__main__':
    main()