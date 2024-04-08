# Define a common configuration for placing the objects 

# Type of items we have

#  Lanes
#  Vehicles - Wiht Subclasses
# Or is it better to have different items have different generator functions 

# 

import bpy 
import numpy as np 
import pickle 
import sys 



# Delete all blender objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
DATA_PATH = BASE_PATH + "P3Data/"
ASSETS_PATH = DATA_PATH + "Assets/"

sys.path.append(BASE_PATH+"Einstein-Vision")
sys.path.append(BASE_PATH+"Einstein-Vision/lib")
sys.path.append(BASE_PATH+"Einstein-Vision/blender")

sys.path.append(BASE_PATH+"blender-4.0.2-linux-x64/blender_env/lib/python3.10/site-packages")

from utilities.three_d_utils import *
from utilities.blender_utils import open_pickle_file
from lib.load_3d_poses import load_pose_data, get_pose_details
import cv2 
from blender.new_renderer import Blender_Utils 
from utilities.three_d_utils import *
from utilities.blender_utils import *


blender_utils = Blender_Utils()

K = np.array([[1622.30674706393,0.0,681.0156669556608],
        [0.0,1632.8929856491513,437.0195537829288],
        [0.0,0.0,1.0]])

R = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]])

blender_utils = Blender_Utils()


Mesh_path = "Pose_Detection/PyMAF/output/scene_10"
pose_data_path = BASE_PATH + Mesh_path+ "/output.pkl"

data = load_pose_data(pose_data_path)

pose_details , pose_bbox = get_pose_details(112, data)

print(pose_bbox)
print(pose_details)

# Fow now taking the BBOX Center based on Random depth this should be modifed

# Write a logic to match bounding box with location 
# Get Depth of that location 

for i in range(len(pose_details)):
    obj_path = pose_details[i]
    obj_bbox = pose_bbox[i]
    
    # Find the center of the bounding box
    center = (obj_bbox[0], obj_bbox[1])
    depth = 1.5
    print(center)
    xyz = form2_conv_image_world(R,K,center, depth)
    print("Object Path: ", obj_path)    
    obj_path = BASE_PATH + Mesh_path + "/" + obj_path
    print("Object Path: ", obj_path)
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    imported_obj = bpy.context.selected_objects[0]
    
    imported_obj.location = (xyz[0], xyz[2], xyz[1])
    
    imported_obj.scale = (1, 1, 1)
    
    imported_obj.rotation_euler = (90, 0, -160)
    

print("Pose Data Loaded and Objects are placed in the scene")





