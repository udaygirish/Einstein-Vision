import bpy
import sys 

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

blender_utils = Blender_Utils()

video_name = "video1"

K,R = get_hardcoded_KR()


BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"

all_objects_data = BASE_PATH + "/P3Data/results_video1.pkl"

def load_pickle_joblib(file_path):
    data = joblib.load(file_path)
    return data

def main():
    all_obj_data = load_pickle_joblib(all_objects_data)
    
    print(all_obj_data[list(all_obj_data.keys())[10]])
    print("====================================="*3)
    BlenderUtils = Blender_Utils()
    CAMERA_LOC = (0,-3, 3)
    # Start Blender rendering 
    # Delete all objects
    
    
    
    
    
if __name__ == "__main__":
    main()


