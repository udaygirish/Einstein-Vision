import bpy
import sys 
import os

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"

sys.path.append(BASE_PATH + "Einstein-Vision")
sys.path.append(BASE_PATH + "Einstein-Vision/lib")
sys.path.append(BASE_PATH + "Einstein-Vision/utilities")
sys.path.append(BASE_PATH + "Einstein-Vision/config")
sys.path.append(BASE_PATH+"blender-4.0.2-linux-x64/blender_env/lib/python3.10/site-packages")
from mathutils import Matrix, Vector 
import numpy as np 
import pickle 
import joblib
from config.blender_config import YOLOCLASSES_TO_BLENDER, Blender_rot_scale
import cv2 
from utilities.three_d_utils import *
from utilities.blender_utils import open_pickle_file
from blender.new_renderer import Blender_Utils
from utilities.cv2_utilities import *
from utilities.blender_utils import *
from utilities.random_utils import *
from PIL import Image
from lib.load_3d_poses import load_pose_data, get_pose_details
from tqdm import tqdm 
import time 

#blender_utils = Blender_Utils()

video_name = "video5"

K,R = get_hardcoded_KR()




BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"

all_objects_data = BASE_PATH + "/P3Data/results_video5_old.pkl"
person_obj_data = BASE_PATH + "/P3Data/Pose_Outputs/scene_5/output.pkl"

vehicle_types = ['car', "suv", "truck", "pickup_truck", "sedan", "motorcycle", "bicycle"]
traffic_light_types = ["traffic_light", "green traffic light", "red traffic light",
                           "yellow traffic light"]

def load_pickle_joblib(file_path):
    data = joblib.load(file_path)
    return data

def main():
    all_obj_data = load_pickle_joblib(all_objects_data)
    
    all_obj_data_frames = list(all_obj_data.keys())
    #print("All frames inside", all_obj_data_frames)
    #print("Keys inside the object data: ", all_obj_data[all_obj_data_frames[0]].keys()) 
    #print("Keys inside the sample object data: ", all_obj_data[all_obj_data_frames[0]]["objects"][0].keys())
    print("====================================="*3)
    BlenderUtils = Blender_Utils()
    CAMERA_LOC = (0,0,1.5)
    # Start Blender rendering 
    # Delete all objects
    
    hard_coded_scale_factor = 0.9     
    blend_obj_gen = Blender_Utils()
    # blend_obj = blend_obj_gen.objects
    # print object names
    #print("Object names: ", blend_obj_gen.objects)
    print("First frame: ", all_obj_data_frames[0])
    print("Last frame: ", all_obj_data_frames[-1])
    ob_f_no = 12
    #print("Current Frame: ", all_obj_data_frames[ob_f_no])
    lane_backup_list = None
    total_lane_classes_blist = None
    for i in range(len(all_obj_data_frames)):
        if i>5:
            continue
        # Delete all blender objects
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        blend_obj_gen.delete_all_objects()
        
        # Delete sun light and camera
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()
        
        bpy.ops.object.select_by_type(type='CAMERA')
        bpy.ops.object.delete()
        
        # Create Camera
        blend_obj_gen.setup_camera()
        
        
        # Create Road surface in Blender
        blend_obj_gen.create_road_surface()

        # Create lanes 
        lanes = all_obj_data[all_obj_data_frames[i]]["lane_3d_points"]
        total_lane_classes = all_obj_data[all_obj_data_frames[i]]["lane_classes"]
        if len(lanes) == 0:
            if lane_backup_list is not None:
                lanes = lane_backup_list
                total_lane_classes = total_lane_classes_blist
            else:
                lanes =  []
                total_lane_classes = []
        if len(lanes) > 0:
            lane_backup_list = lanes
            total_lane_classes_blist = total_lane_classes
        for l in range(len(lanes)):
            curve_obj = blend_obj_gen.create_bezier_curve_from_points(lanes[l])
            if total_lane_classes[l] == "solid-line":
                blend_obj_gen.create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=10, gap_length=0, num_lanes=15)
            else:
                blend_obj_gen.create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=3, gap_length=1, num_lanes=8)
        
        blendable_objects = all_obj_data[all_obj_data_frames[i]]["objects"]
        blendable_persons = all_obj_data[all_obj_data_frames[i]]["persons"]
        blendable_obj_names = [i.name for i in blend_obj_gen.objects]
        
        Objects = []
        Orientations = []
        Locations = []
        Scales = []
        ObjectStates = []
        
        for o in range(len(blendable_objects)):
            obj = blendable_objects[o]
            
            coords_3d = obj["3d_world_coords"]
            bbox_2d = obj["bbox_2d"]
            class_name = obj["class_name"]
            orientation = obj["orientation"]
            scale = obj["scale"]
            score = obj["score"]
            state_label = obj["state_label"]
            moving_label = obj["moving_label"]
            avg_velocity = obj["avg_velocity"]
            track_id = obj["track_id"]
            pose_path = obj["pose_path"]
            print("Class Name: ", class_name)
            if class_name in YOLOCLASSES_TO_BLENDER.keys():
                asset_name = YOLOCLASSES_TO_BLENDER[class_name]
                ind = next(i for i,s in enumerate(blendable_obj_names) if s.split('.')[0].startswith(asset_name.split('.')[0]))
                temp_obj = blend_obj_gen.objects[ind]
                Orientations.append(orientation)
                if state_label == "car_BrakeOn":
                    mod_class_name = class_name + "Red"
                    asset_name = YOLOCLASSES_TO_BLENDER[mod_class_name]
                    #print("Asset Name",asset_name)
                    #print("Blendable Object Names: ", blendable_obj_names)
                    ind = next(i for i,s in enumerate(blendable_obj_names) if s.split('.')[0].startswith(asset_name.split('.')[0]))
                if class_name in traffic_light_types:
                    Locations.append([coords_3d[0],  coords_3d[1], abs(coords_3d[2])+1.5])
                else:
                    Locations.append(coords_3d)
                #print("State Label: ", state_label)
                #print("Moving Label: ", moving_label)
                if moving_label == "Stationary" and state_label  != "car_BrakeOn":
                    blend_obj_gen.add_texture(temp_obj, color=(0.04,0.04, 0.04, 1))
                if class_name == "trash can" or class_name == "dust bin":
                    blend_obj_gen.add_texture(temp_obj)
                Objects.append(temp_obj)
                Scales.append(scale)
                ObjectStates.append(moving_label)
        
        blend_obj_gen.load_objects_to_blender(Objects, Orientations, Locations, Scales, ObjectState= ObjectStates)
        
        
        for k in range(len(blendable_persons)):
            print("Frame Number: ", i)
            person = blendable_persons[k]
            coords_3d = person["3d_world_coords"]
            bbox_2d = person["bbox_2d"]
            class_name = person["class_name"]
            orientation = person["orientation"]
            scale = person["scale"]
            score = person["score"]
            state_label = person["state_label"]
            # moving_label = person["moving_label"]
            # avg_velocity = person["avg_velocity"]
            base_pose_path = BASE_PATH + "/P3Data/Pose_Outputs/scene_5/meshes/"
            pose_path = person["pose_path"]
            
            bpy.ops.wm.obj_import(filepath=base_pose_path + pose_path)
            imported_obj = bpy.context.selected_objects[0]
            imported_obj.location = (coords_3d[0], coords_3d[1] , coords_3d[2]+0.5)
            imported_obj.scale = (1, 1, 1)
            imported_obj.rotation_euler = (90, 0, -170)
            
        # Load the Camera and save the render to save into a folder of images
        blend_obj_gen.render_cam_frame(f"Outputs/BLENDER/{video_name}", frame_name= "frame_{}.png".format(i), frame_number= 1)
        
    
    # Make a video from the images
    make_video_from_images(f"Outputs/BLENDER/{video_name}", f"Outputs/BLENDER/{video_name}.mp4")
    
    print("Video Created")


# Function to make a video from images

def make_video_from_images(image_folder, video_name):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    print("Length of Images: ", len(images))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()
    
    print("Video Created")
    
# def side_by_side_video(input_folder, output_folder, video_name):
    
    
    
    
    
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    
    print("Time Taken for Processing and Rendering the frames: ", t2-t1)


