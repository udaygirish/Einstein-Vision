import bpy
import pickle
import random
import os 
import sys 
import numpy as np

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
sys.path.append(BASE_PATH+"Einstein-Vision")
sys.path.append(BASE_PATH+"Einstein-Vision/lib")
sys.path.append(BASE_PATH+"Einstein-Vision/blender")

sys.path.append(BASE_PATH+"blender-4.0.2-linux-x64/blender_env/lib/python3.10/site-packages")

# Setup 
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def open_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
DATA_PATH = BASE_PATH + "P3Data/"
ASSETS_PATH = DATA_PATH + "Assets/"
class Blender_Utils:
    # For now first view rendering is fine 
    
    # ToDo:
    # 1. Get all the objects in all files in blender assets
    # 2. Make it a available variable
    # 3. Modify the functions create road lanes and Spawn Object
    # 4. Add function to delete or clear assets
    # 5. Add function to switch between cameras
    # 6. Add Support to render both cameras
    
    def __init__(self, object_names=None):
        self.description = "Blender Utilities to play with objects and cameras"
        self.K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

        self.R = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1.5],
            [0, 0, 0, 1]])
        self.first_person_camera_location = (0, -5,0.5) 
        self.first_person_camera_rotation = (1.57, 0, 0)
        
        self.third_person_camera_location = (3, -4, 1)
        self.third_person_camera_rotation = (1.57,0, 0.75)
        
        if object_names is None:
            self.vehicles = ["Bicycle", "PickupTruck", "SedanAndHatchback", "SUV", "Truck"]
            self.entities = ["Dustbin", "Pedestrain", "SpeedLimitSign", "StopSign", "TrafficAssets", "TrafficConeAndCylinder", "TrafficSignal"]
        
        
    # Unrealted function below - Yet to modify
    def create_road_lanes(self):
        bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        road_surface = bpy.context.object
        road_surface.name = "Road_Surface"
        
#         Assign material to road surface
        road_material = bpy.data.materials.new(name="Road_Material")
        road_material.diffuse_color = (0, 0, 0, 1)  # Black color
        road_surface.data.materials.append(road_material)
        
#         Create lane markings
        bpy.ops.mesh.primitive_cube_add(size=0.1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        lane_marking = bpy.context.object
        lane_marking.name = "Lane_Marking"
        lane_marking.scale = (1, 4, 0.6)
        
#        Assign material to lane markings
        lane_material = bpy.data.materials.new(name="Lane_Material")
        lane_material.diffuse_color = (1, 1, 1, 1)  # White color
        lane_marking.data.materials.append(lane_material)
        
#         Create multiple lane markings
        for i in range(-999, 1000):
            new_lane = lane_marking.copy()
            new_lane.data = lane_marking.data.copy()
            bpy.context.collection.objects.link(new_lane)
            new_lane.location = (0, i * 1.0, 0)  # Adjust the spacing between lanes
    
    def setup_camera_first_person_view(self):
        # Create Camera Object - First Person View / Frontal
        bpy.ops.object.camera_add(enter_editmode=False,align='VIEW',location=(0,0,0))
        camera =  bpy.context.object
        camera.name = "Camera_First_Person"
        
        camera.location = self.first_person_camera_location
        camera.rotation_euler = self.first_person_camera_rotation
        
    def setup_camera_third_person_view(self):
        # Create Camera Object - Third Person View /Oblique
        bpy.ops.object.camera_add(enter_editmode=False,align='VIEW',location=(0,0,0))
        camera =  bpy.context.object    
        camera.name = "Camera_Third_Person"
        
        camera.location = self.third_person_camera_location
        camera.rotation_euler = self.third_person_camera_rotation
        
    # Unrelated function definition below 
    def SpawnObject(self, file_path, object_names, location, scale):
        
        with bpy.data.libraries.load(file_path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for i, obj in enumerate(data_to.objects):
            bpy.context.collection.objects.link(obj)
            obj.location = location
            obj.scale = scale
        return data_to.objects

blender_utils = Blender_Utils()

blender_utils.setup_camera_first_person_view()

blender_utils.create_road_lanes()
#vehicles = blender_utils.SpawnObject(ASSETS_PATH+"Vehicles/SedanAndHatchback.blend", ["Sedan"], (-2,3,0), (1,1,1))


