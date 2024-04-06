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
from lib.ocrInference import *
from lib.data_prep import *
from lib.traffic_sign_thresholder import *
from lib.optical_flow import *
from lib.vehicle_indicators import *
from lib.yolo_world_exp import *

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

def load_all_models():
    # Load all the models
    model_obj = load_model_det()
    model_pose = load_model_pose()
    model_depth = load_model_depth()
    model_seg = load_model_seg()
    lane_class = load_model_lane_classifier()
    vehicle_model = load_model_veh_ind()
    oflow_model = load_oflow_model()
    model_yolo_world = load_model_yworld()
    
    return model_obj, model_pose, model_depth, model_seg, lane_class, vehicle_model, oflow_model, model_yolo_world

def single_image_pipeline(model_list, image_path_before, image_path):
    
    
    # Load all models
    model_obj, model_pose, model_depth, model_seg, lane_class, vehicle_model, oflow_model, model_yolo_world = model_list
    
    
    # # Read image 
    image1 = Image.open(image_path_before).convert("RGB")   
    image2 = Image.open(image_path).convert("RGB")

    # # Get lane detection
    src, lanes = il_inf(image_path)
    
    # # Get Object Detection
    obj_res, boxes_total, classes_total, scores_total, classes_names = predict_image_det(model_obj, image_path)
    
    # # Get Pose Detection
    pose_res = predict_image_pose(model_pose, image_path)
    
    # # Get depth
    depth = run_inference_depth(model_depth, image_path)
    
    # # Get Segmentation
    seg_res = predict_image_seg(model_seg, image_path)
    
    # # lane classification
    lane_masks, lane_boxes, lane_labels = infer_image_lane_classifier(lane_class, image_path)
    
    # Currently traffic sign detection is from yolo 
    filtered_boxes, filtered_classes, filtered_scores, filtered_classes_names = get_filter_boxes(boxes_total, classes_total, scores_total, classes_names, ["car", "truck", "bus", "motorbike", "bicycle", "person"])
    
    moving_labels_list = get_movement_classification(oflow_model, image_path_before, image_path, filtered_boxes)
    
    filtered_boxes1, filtered_classes1, filtered_scores1, filtered_classes_names1 = get_filter_boxes(boxes_total, classes_total, scores_total, classes_names, ["car", "truck"])
    
    vehicle_results, vehicle_boxes, vehicle_classes, vehicle_scores, vehicle_classes_names = predict_image_ind_classes(vehicle_model, image_path)
    
    world_results, world_boxes, world_classes, world_scores, world_classes_names = predict_image_yworld(model_yolo_world, image_path)
    # Get YOlo3d 
    # Defining some variables - Shift Later
    BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
    D3_WEIGHTS_PATH = "Object_Detection/YOLO3D/weights/resnet.pkl"
    MODEL_SELECT = "resnet" 
    CALIB_FILE_PATH = "../Object_Detection/YOLO3D/eval/camera_cal/calib_cam_to_cam.txt"
    print("IN YOLO 3d")
    out_Objects_3d = detect3d(BASE_PATH+D3_WEIGHTS_PATH,
                              MODEL_SELECT,
                              [image_path],
                                CALIB_FILE_PATH,
                                show_result=False,
                                save_result=False,
                                output_path=None)
    # # TOtal results
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
        'lane_labels': lane_labels,
        'yolo3d': out_Objects_3d,
        'optical_flow': {
            'moving_labels': moving_labels_list,
            'filtered_boxes': filtered_boxes,
        },
        'object_detection_yworld': {
            'boxes': world_boxes,
            'classes': world_classes,
            'scores': world_scores,
            'classes_names': world_classes_names
        },
        'vehicle_indicators': {
            'vehicle_results': vehicle_results,
            'vehicle_boxes': vehicle_boxes,
            'vehicle_classes': vehicle_classes,
            'vehicle_scores': vehicle_scores,
            'vehicle_classes_names': vehicle_classes_names
        }
    }
    
    final_lanes = lane_class_matcher(results)
    
    results['final_lanes'] = final_lanes
    
    return results
    
def main():
    # Function to process a image
    # Get objects and save as JSON
    total_images = glob.glob("../P3Data/test_video_frames/*.png")
    print("Total Images: ", len(total_images))
    K, R = get_hardcoded_KR()
    total_images = total_images
    total_images = sorted(total_images)
    total_images = total_images[:3]
    print("Total Images: ", len(total_images))
    result_dict = {}
    model_list = load_all_models()
    for i in tqdm(range(len(total_images)-1)):
        img1_path = total_images[i]
        img2_path = total_images[i+1]
        print("Image 1: ", img1_path)
        print("Image 2: ", img2_path)
        temp_results = single_image_pipeline(model_list, img1_path,img2_path)
        results = data_processor(temp_results, img2_path)
        result_dict[img2_path] = temp_results
    
    
    # video_name = "video1"
    # Dump the results to pickle
    # with open(BASE_PATH+ "P3Data/results_{}.pkl".format(video_name), 'wb') as f:
    #     pickle.dump(result_dict, f)
    
    

if __name__ == '__main__':
    main()