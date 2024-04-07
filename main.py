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
from config.blender_config import YOLOCLASSES_TO_BLENDER, Blender_rot_scale

from utilities.cv2_utilities import *
from utilities.blender_utils import *
from utilities.three_d_utils import *
from utilities.random_utils import *
from lib.load_3d_poses import load_pose_data, get_pose_details


# Have to check how the Human in Blender thing works 

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
def load_yolo3d_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_all_models():
    # Load all the models
    #model_obj = load_model_det()
    model_obj = None
    #model_pose = load_model_pose()
    model_pose = None
    model_depth = load_model_depth()
    #model_seg = load_model_seg()
    model_seg = None
    lane_class = load_model_lane_classifier()
    vehicle_model = load_model_veh_ind()
    oflow_model = load_oflow_model()
    model_yolo_world = load_model_yworld()
    model_yolo_world_min = load_model_yworld(classes = ["car", "truck", "bus", "motorbike", "bicycle", "person"])
    
    return model_obj, model_pose, model_depth, model_seg, lane_class, vehicle_model, oflow_model, model_yolo_world, model_yolo_world_min

def single_image_pipeline(model_list, image_path_before, image_path):
    
    
    # Load all models
    model_obj, model_pose, model_depth, model_seg, lane_class, vehicle_model, oflow_model, model_yolo_world, model_yolo_world_min = model_list
    
    
    # # Read image 
    image1 = Image.open(image_path_before).convert("RGB")   
    image2 = Image.open(image_path).convert("RGB")

    # # Get lane detection
    src, lanes = il_inf(image_path)
    
    # # Get Object Detection
    #obj_res, boxes_total, classes_total, scores_total, classes_names = predict_image_det(model_obj, image_path)
    obj_res, boxes_total, classes_total, scores_total, classes_names = None, None, None, None, None
    
    # # Get Pose Detection
    #pose_res = predict_image_pose(model_pose, image_path)
    pose_res = None
    
    # # Get depth
    depth = run_inference_depth(model_depth, image_path)
    
    # # Get Segmentation
    # seg_res = predict_image_seg(model_seg, image_path)
    seg_res = None
    
    world_results, world_boxes, world_classes, world_scores, world_classes_names = predict_image_yworld(model_yolo_world, image_path)
    
    world_results_min, world_boxes_min, world_classes_min, world_scores_min, world_classes_names_min = predict_image_yworld(model_yolo_world_min, image_path)
    # # lane classification
    lane_masks, lane_boxes, lane_labels = infer_image_lane_classifier(lane_class, image_path)
    
    vehicle_results, vehicle_boxes, vehicle_classes, vehicle_scores, vehicle_classes_names = predict_image_ind_classes(vehicle_model, image_path)
    # Currently traffic sign detection is from yolo 
    #filtered_boxes, filtered_classes, filtered_scores, filtered_classes_names = get_filter_boxes(world_boxes_min, world_classes_min, world_scores_min, world_classes_min , ["car", "truck", "bus", "motorbike", "bicycle", "person"])
    
    moving_labels_list = get_movement_classification(oflow_model, image_path_before, image_path, world_boxes_min)
    
    #filtered_boxes1, filtered_classes1, filtered_scores1, filtered_classes_names1 = get_filter_boxes(world_boxes, world_classes, world_scores, world_classes_names, ["car", "truck"])
    
    
    
    
    
    
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
            'filtered_boxes': world_boxes_min,
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
    
def main(vid_id):
    # Function to process a image
    # Get objects and save as JSON
    total_images = glob.glob("../P3Data/FinalImages/Images_{}/*.jpg".format(vid_id))
    print("Total Images: ", len(total_images))
    K, R = get_hardcoded_KR()
    total_images = total_images
    total_images = sorted(total_images)
    
    # Load pose data
    pose_data_path = "../P3Data/Pose_Outputs/scene_{}/output.pkl".format(vid_id)
    
    pose_data = load_pose_data(pose_data_path)
    #total_images = total_images[95:125]
    print("Total Images: ", len(total_images))
    result_dict = {}
    model_list = load_all_models()
    for i in tqdm(range(len(total_images)-1)):
        img1_path = total_images[i]
        img2_path = total_images[i+1]
        print("===================================="*3)
        print("Image 1: ", img1_path)
        print("Image 2: ", img2_path)
        print("===================================="*3)
        temp_results = single_image_pipeline(model_list, img1_path,img2_path)
        results_f_3d = data_processor(temp_results, img2_path)
        
        # Get person details and mesh and add them here 
        person_index = int(img2_path.split("/")[-1].split(".")[0].split("_")[-1])
        print(person_index)
        
        pose_details , pose_bbox = get_pose_details(i, pose_data)
        
        total_poses = []
        for i in range(len(pose_details)):
            pose_dict = {
                        '3d_world_coords': [],
                        'bbox_2d': [],
                        'class_name': "",
                        'orientation': [],
                        'scale': [],
                        'score': 0.0,
                        'state_label': "",
                        'avg_velocity': [],
                        'track_id': 0,
                        'pose_path': ""
                    }
            obj_path = pose_details[i]
            obj_bbox = pose_bbox[i]
            obj_class = "person"
            centroid = (obj_bbox[0], obj_bbox[1])
            z_val = temp_results['depth'][int(centroid[1]), int(centroid[0])]
            xyz = form2_conv_image_world(R,K,centroid, z_val)
            center = [xyz[0], xyz[2], 0]
            euler_angles = Blender_rot_scale['person']['orientation']
            scale = np.array(Blender_rot_scale['person']['scale'])  
            mesh_obj_path = '/'.join(obj_path.split("/")[-2:])
            pose_dict['3d_world_coords'] = center
            pose_dict['bbox_2d'] = obj_bbox
            pose_dict['class_name'] = obj_class
            pose_dict['orientation'] = euler_angles
            pose_dict['scale'] = scale
            pose_dict['score'] = 0.0
            pose_dict['state_label'] = "standing"
            pose_dict['pose_path'] = mesh_obj_path
            
            total_poses.append(pose_dict)
        
            
        results_f_3d['persons'] = total_poses  
        result_dict[img2_path] = results_f_3d
    
    
    video_name = "video{}".format(vid_id)
    # Dump the results to pickle
    with open(BASE_PATH+ "P3Data/results_{}.pkl".format(video_name), 'wb') as f:
        pickle.dump(result_dict, f)
    
    

if __name__ == '__main__':
    failed_cases = []
    vid_id_list = [1,2,3,4,7,8,9,10,11,12,13,5,6]
    for vid_id in vid_id_list:
        print("===================================="*3)
        print("Running for Image Folder: ", vid_id)
        print("===================================="*3)
        try:
            main(vid_id)
        except Exception as e:
            print("Failed for Image Folder: ", vid_id)
            print("Error: ", e)
            failed_cases.append(vid_id)
            pass
        print("===================================="*3)
    
    
    print("===================================="*3)
    print("Failed Cases: ", failed_cases)
    print("===================================="*3)