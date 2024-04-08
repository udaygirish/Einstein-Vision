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
from mathutils import Matrix, Vector
# Setup 
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

from utilities.three_d_utils import form2_conv_image_world, get_scale_factor
from utilities.blender_utils import open_pickle_file

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
DATA_PATH = BASE_PATH + "P3Data/"
ASSETS_PATH = DATA_PATH + "Assets/"

from utilities.three_d_utils import * 
from utilities.blender_utils import open_pickle_file
import cv2 
# assets objects to YOLO class matches

# Extra Traffic assets class - TrafficAssets
# barrel, cone, no parking, fire, trash can, iron pole 
# SpeedLimitSign - sign_25mph_sign_25mph

YOLO_CLASSES_TO_BLENDER = {
	"car": {
		"car_1": "Car",
		"car_2": "jeep_3_"
	},
	"truck": {
		"truck_1": "PickupTruck",
		"truck_2": "Truck"
	},
	"motorcycle": "B_Wheel",
	"bicycle": "roadbike 2.0.1",
	"dustbin": "Bin_Mesh.072",
	"stop sign": "StopSign_Geo",
	"parking meter": "sign_25mph_sign_25mph",
	"traffic light": "TrafficSignal",
	"person": "BaseMesh_Man_Simple",
	"fire": "fire hydrant",
	"traffic cone": "absperrhut"
}

    
    
class Blender_Utils:
    # For now first view rendering is fine 
    
    # ToDo:
    # 1. Get all the objects in all files in blender assets
    # 2. Make it a available variable
    # 3. Modify the functions create road lanes and Spawn Object
    # 4. Add function to delete or clear assets
    # 5. Add function to switch between cameras
    # 6. Add Support to render both cameras
    
    def __init__(self, assets_path = ASSETS_PATH, object_names=None):
        self.description = "Blender Utilities to play with objects and cameras"
        self.K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

        self.R = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1.5],
            [0, 0, 0, 1]])
        self.first_person_camera_location = (0, 0,1.5) 
        self.first_person_camera_rotation = (1.57, 0, 0)
        
        self.third_person_camera_location = (3, 0, 1)
        self.third_person_camera_rotation = (1.57,0, 0.75)
        
        self.render_width = 1920
        self.render_height = 1080
        
        self.veh_path = assets_path + "Vehicles/"
        self.ent_path = assets_path + "Entities/"
        self.mod_ent_path = assets_path + "Mod_Entities/"
        self.mod_veh_path = assets_path + "Vehicles_RedLight/"   
        self.global_exclude_objects = ['Camera', 'Sun', 'Light', 'Cube']
        # Delete object when initialzing the class to 
        # clear  the scene with all the objects 
        self.delete_all_objects()
        
        # Setup the camera when the render loads 
        self.setup_camera_first_person_view()
        
        
        # Load Object when initializing the class 
        self.objects =  self.load_blend_objects()
        
        
    def setup_light_source(self):
        if not bpy.data.objects.get("SUN"):
            bpy.ops.object.light_add(type='SUN', location = self.light_location)
            light = bpy.context.object
            light.name = "Sun"
            light.data.energy = 2.0
            light.data.use_shadow = True
        self.light_source = bpy.data.objects["Sun"]
        
            
    def delete_all_objects(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
    def delete_all_objects_except_camera(self):
        bpy.ops.object.select_all(action='DESELECT')
        if self.camera:
            self.camera.select_set(True)
        bpy.ops.object.delete()
        
    def delete_from_scene(self):
        scene = bpy.context.scene 
        for obj in scene.objects:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.delete()
    
        
    def load_blend_objects(self):
        objects = []
        exclude_Objects = self.global_exclude_objects
        
        for fpath in sorted(os.listdir(self.veh_path)):
            if fpath.endswith(".blend"):
                with bpy.data.libraries.load(self.veh_path + "/" + fpath) as (data_from, vehicle_data):
                    for obj_from in data_from.objects:
                        if obj_from not in exclude_Objects:
                            vehicle_data.objects.append(obj_from)
                for veh in vehicle_data.objects:
                    objects.append(veh)
                    
        for fpath in sorted(os.listdir(self.ent_path)):
            if fpath.endswith(".blend"):
                with bpy.data.libraries.load(self.ent_path + "/" + fpath) as (data_from, entity_data):
                    for obj_from in data_from.objects:
                        if obj_from not in exclude_Objects:
                            entity_data.objects.append(obj_from)
                for ent in entity_data.objects:
                    objects.append(ent)
                    
        for fpath in sorted(os.listdir(self.mod_ent_path)):
            if fpath.endswith(".blend"):
                with bpy.data.libraries.load(self.mod_ent_path + "/" + fpath) as (data_from, mod_entity_data):
                    for obj_from in data_from.objects:
                        if obj_from not in exclude_Objects:
                            mod_entity_data.objects.append(obj_from)
                for mod_ent in mod_entity_data.objects:
                    objects.append(mod_ent)
                    
        for fpath in sorted(os.listdir(self.mod_veh_path)):
            if fpath.endswith(".blend"):
                with bpy.data.libraries.load(self.mod_veh_path + "/" + fpath) as (data_from, mod_vehicle_data):
                    for obj_from in data_from.objects:
                        if obj_from not in exclude_Objects:
                            mod_vehicle_data.objects.append(obj_from)
                for mod_veh in mod_vehicle_data.objects:
                    objects.append(mod_veh)
                    
        return objects
    
    
    def create_bezier_curve(self, points, name="Bezier_Curve"):
        curve_data = bpy.data.curves.new(name="Bezier_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(len(points[0])-1)
        for i, (x, y, z) in enumerate(zip(points[0], points[1], sorted(points[2]))):
            polyline.points[i].co = (x, z,0, 1) # Aligning Y along z and setting z=0 to ensure the points are snapped to Ground
            # This errors are because of issues in finding the depth (Many inaccuracies) 
            # to reduce we do a direct snapping 
        
        curve_object = bpy.data.objects.new(name="Bezier_Curve_Object", object_data=curve_data)
        bpy.context.collection.objects.link(curve_object)
        return curve_object
        
    def create_bezier_curve_from_points(self, points, name="Bezier_Curve"):
    # Create curve object in Blender without using Poly
        curve_data = bpy.data.curves.new(name="Bezier_Curve", type='CURVE')
        curve_data.dimensions = '3D'
        polyline = curve_data.splines.new('BEZIER')
        polyline.bezier_points.add(len(points)-1)
        for i, (x, y, z) in enumerate(points):
            polyline.bezier_points[i].co = (x, z, 0)
            polyline.bezier_points[i].handle_left = (x, z, 0)
            polyline.bezier_points[i].handle_right = (x, z, 0)
        
        curve_object = bpy.data.objects.new(name="Bezier_Curve_Object", object_data=curve_data)
        bpy.context.collection.objects.link(curve_object)
        return curve_object
        
    def create_road_surface(self):
        bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        road_surface = bpy.context.object
        road_surface.name = "Road_Surface"
        
        # Assign material to road surface
        road_material = bpy.data.materials.new(name="Road_Material")
        road_material.diffuse_color = (0, 0, 0, 1)  # Black color
        road_surface.data.materials.append(road_material)
        
    def create_lane_markings(self, curve_object, lane_width=4, lane_length=10, gap_length=1, num_lanes=10):
        bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.context.object.scale[1] = 0.1 * lane_width
        bpy.context.object.scale[0] = 0.1* lane_length
        bpy.context.object.scale[2] = 0.05
        
        # Add modifier 
        bpy.ops.object.modifier_add(type='ARRAY')
        bpy.context.object.modifiers["Array"].count = num_lanes
        
        # Position lane markings along the curve object
        bpy.context.object.modifiers["Array"].use_constant_offset = True
        bpy.context.object.modifiers["Array"].constant_offset_displace[0] = gap_length
        bpy.ops.object.modifier_add(type='CURVE')
        bpy.context.object.modifiers["Curve"].object = curve_object
        
        # Currently deforming along Position X as it is the current fit 
        bpy.context.object.modifiers["Curve"].deform_axis = 'POS_X'
        
    def create_lane_markings_by_curve_length(self, curve_object, lane_width=4, lane_length=10, gap_length=1, num_lanes=10):
        bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.context.object.scale[1] = 0.1 * lane_width
        bpy.context.object.scale[0] = 0.1* lane_length
        bpy.context.object.scale[2] = 0.05
        
        # Add modifier 
        bpy.ops.object.modifier_add(type='ARRAY')
        # Length of the lane markings array should be the length of the curve object
        bpy.context.object.modifiers["Array"].fit_type = 'FIT_CURVE'
        bpy.context.object.modifiers["Array"].curve = curve_object
        
        # Position lane markings along the curve object
        bpy.context.object.modifiers["Array"].use_constant_offset = True
        bpy.context.object.modifiers["Array"].constant_offset_displace[0] = gap_length
        bpy.ops.object.modifier_add(type='CURVE')
        bpy.context.object.modifiers["Curve"].object = curve_object
        
        # Currently deforming along Position X as it is the current fit 
        bpy.context.object.modifiers["Curve"].deform_axis = 'POS_X'
        
        
    # Unrealted function below - Yet to modify
    def create_road_lanes(self):
        bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        road_surface = bpy.context.object
        road_surface.name = "Road_Surface"
        road_surface.scale = (.15, 2, .15)
        
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
        return camera 
        
    def setup_camera_third_person_view(self):
        # Create Camera Object - Third Person View /Oblique
        bpy.ops.object.camera_add(enter_editmode=False,align='VIEW',location=(0,0,0))
        camera =  bpy.context.object    
        camera.name = "Camera_Third_Person"
        
        camera.location = self.third_person_camera_location
        camera.rotation_euler = self.third_person_camera_rotation
        return camera 
        
    # Unrelated function definition below 
    def SpawnObject(self, file_path, object_names, location, rotation, scale):
        
        with bpy.data.libraries.load(file_path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for i, obj in enumerate(data_to.objects):
            bpy.context.collection.objects.link(obj)
            obj.location = location
            obj.rotation_euler = rotation
            obj.scale = scale
        return data_to.objects
    
    def setup_camera_loc(self, camera_loc):
        self.camera.location = camera_loc
        
    def add_texture(self,obj, color=(0, 0.1,0,0)):
        mat_green = bpy.data.materials.new(name="Black")
        mat_green.diffuse_color = color
        obj.active_material = mat_green
        return obj

    def setup_camera(self):
        # Check if camera exists, if not create a new camera
        if not bpy.data.objects.get("Camera"):
            bpy.ops.object.camera_add()
        if not bpy.data.objects.get("SUN"):
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        self.camera = bpy.data.objects.get("Camera")
        self.camera.location = (0, 0, 1.5)
        self.camera.rotation_euler = (1.57, 0, 0)
    
    def load_objects_to_blender(self, objects, orientations, locations, scales, frame_number=1, camera_loc=(0,0,1.5), ObjectState = None):
        bpy.context.scene.frame_set(frame_number)
        self.setup_camera()
        self.setup_camera_loc(camera_loc)
        
        for obj_c, obj in enumerate(objects):
            obj = obj.copy()
            obj.data = obj.data.copy()
            obj.rotation_euler = orientations[obj_c]
            obj.location = locations[obj_c]
            obj.scale = scales[obj_c]
            obj.keyframe_insert(data_path="location", frame =frame_number, index=-1)
            obj.keyframe_insert(data_path="rotation_euler", frame =frame_number, index=-1)
            obj.keyframe_insert(data_path="scale", frame =frame_number, index=-1)
            
            bpy.context.collection.objects.link(obj)    
        del objects, orientations, locations, scales
            
        
    def setup_cameras(self):
        # Setup First Person Camera
        self.first_camera = self.setup_camera_first_person_view()
        # Setup Third Person Camera
        self.third_camera = self.setup_camera_third_person_view()
        
    def render_cam_frame(self, output_path, frame_name, frame_number, width =1920, height=1080):
        # Add light to the scenes on top of the camera 
        bpy.ops.object.light_add(type='SUN', location=(0,0,5))
        bpy.context.scene.camera = self.camera 
        bpy.context.scene.render.filepath = os.path.join(output_path, f"{frame_name}")
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.ops.render.render(write_still=True)
        self.delete_from_scene()