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
from lib.traffic_sign_thresholder import traffic_sign_threshold


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
    
    traffic_sign_threshold(image, boxes_total, classes_total, scores_total, classes_names)
    
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
    
    # Temp method to use the 3d json file directly 
    # YOLO_3D_JSON_PATH = "P3Data/frames_test_yolo3d.json"
    
    # out_Objects_3d = load_yolo3d_json(BASE_PATH+YOLO_3D_JSON_PATH)
    # frame_name = image_path.split("/")[-1].split(".")[0]
    # frame_no  = int(frame_name.split("_")[-1])
    # out_Objects_3d = out_Objects_3d[frame_no-1]
    
    # print("Type of all the outputs")
    
    print(type(lanes))
    print(type(obj_res))
    print(type(pose_res))
    print(type(depth))
    print(type(seg_res))
    print(type(lane_masks))
    print(type(lane_boxes))
    print(type(lane_labels))
    
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
        'lane_labels': lane_labels,
        'yolo3d': out_Objects_3d
    }
    final_lanes = lane_class_matcher(results)
    
    results['final_lanes'] = final_lanes
    return results
    
def main():
    # Function to process a image
    # Get objects and save as JSON
    total_images = glob.glob("../P3Data/new_test/*.jpg")
    print("Total Images: ", len(total_images))
    total_images = total_images[:1]
    total_images = sorted(total_images)
    #total_images = total_images[200:201]
    result_dict = {}
    for image_path in total_images:
        temp_results = single_image_pipeline(image_path)
        print("Processing Image: ", image_path)
        print("=====================================")
        print("POSE DETECTION OUTPUT")
        print(temp_results['pose_detection'])
        print("=====================================")
        print(temp_results['yolo3d'])
        result_dict[image_path] = temp_results
    
    
    # Dump the results to pickle
    with open(BASE_PATH+ "P3Data/results.pkl", 'wb') as f:
        pickle.dump(result_dict, f)
    
    

if __name__ == '__main__':
    main()