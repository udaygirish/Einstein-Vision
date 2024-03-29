import bpy
import os
from mathutils import Matrix , Vector
import numpy as np
import json
import copy

class BlenderSceneGenerator:
    def __init__(self):
        self.delete_all_objects()
        # self.camera = None  # Initialize camera as None
        self.setup_camera()  # Setup camera
        self.v_objects, self.en_objects = self.load_blend_files()

    def setup_camera(self):
        # Check if camera exists, otherwise create one
        if not bpy.data.objects.get('Camera'):
            bpy.ops.object.camera_add()
        self.camera = bpy.data.objects.get('Camera')
        self.camera.location = (0, -5, 1.5)
        self.camera.rotation_euler = (1.57, 0, 0)

    def delete_all_objects(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def delete_all_objects_except_camera(self):
        bpy.ops.object.select_all(action='DESELECT')
        if self.camera:
            self.camera.select_set(True)
        bpy.ops.object.delete()
    
    def load_blend_files(self):
        vPath = '/home/abven/CV_Course/badari_p3/P3Data/Assets/Vehicles/'
        ePath = '/home/abven/CV_Course/badari_p3/P3Data/Assets/Entities/'
        v_objects, en_objects = [], []
        exclude_Objects = ['Camera', 'Sun', 'Light']
        
        for bFile in sorted(os.listdir(vPath)):
            with bpy.data.libraries.load(vPath + '/' + bFile) as (data_from, vehicle_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        vehicle_data.objects.append(obj_from)
            v_objects.append(vehicle_data.objects)

        for bFile in sorted(os.listdir(ePath)) :
            with bpy.data.libraries.load(ePath + '/' + bFile) as (data_from, ent_data):
                for obj_from in data_from.objects:
                    if obj_from not in exclude_Objects:
                        ent_data.objects.append(obj_from)
            en_objects.append(ent_data.objects)

        return v_objects, en_objects
    
    
    def load_objects_into_frame(self, objects, orientations=None, locations=None, scales=None, frame_number=1,camera_loc=(0,0,1.5)):
        
        bpy.context.scene.frame_set(frame_number)
        self.setup_camera()
        self.setup_camera_location(camera_loc)
        for obj_c, obj in enumerate(objects):
            
            # self.load_object([obj], orientations[obj_c], locations[obj_c], scales[obj_c], frame_number)
            obj = obj[0]
            obj = obj.copy()
            obj.data = obj.data.copy()
            obj.rotation_euler = orientations[obj_c]
            obj.location = locations[obj_c]
            obj.scale = scales[obj_c]
            obj.keyframe_insert(data_path="location", frame = frame_number ,index=-1)
            obj.keyframe_insert(data_path="rotation_euler", frame = frame_number , index=-1)
            obj.keyframe_insert(data_path="scale", frame = frame_number , index=-1)
            bpy.context.collection.objects.link(obj)

    def setup_camera_location(self,camera_loc):
        self.camera.location = camera_loc

    def render_frame(self, output_path, frame_number, width=1920, height=1080):
        bpy.context.scene.camera = self.camera
        bpy.context.scene.render.filepath = os.path.join(output_path, f"frame_{frame_number:04d}.png")
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.ops.render.render(write_still=True)

generator = BlenderSceneGenerator()
#generator.load_object(generator.v_objects[4] , location = (0,0,0))
#generator.load_object(generator.en_objects[0])
# car - 4
# vehicles = ['Bicycle', 'Motorcycle', 'PickupTruck', 'SUV', 'SedanAndHatchback', 'Truck']
vehicles = ['Bicycle', 'Motorcycle', 'PickupTruck', 'Jeep', 'car', 'Truck']
entities = ['Dustbin', 'Pedestrain', 'SpeedLimitSign', 'StopSign', 'TrafficAssets', 'TrafficConeAndCylinder', 'TrafficSignal']

def find_xyz(R, K, pts, depth):
    
    u ,v = pts
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    xyz = np.array([x, y, z , 1]).T
    xyz = np.dot(R, xyz)
    xyz = xyz[:3]

    return xyz

K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

R = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]])


with open('/home/abven/CV_Course/badari_p3/scene9_frames.json', 'r') as f: # YOLO3D preds
    json_data = json.load(f)
    
with open('/home/abven/CV_Course/badari_p3/xy_data.json', 'r') as f: # YOLOv8 2D predictions data
    xy_json_data = json.load(f)
    
with open('/home/abven/CV_Course/badari_p3/scene9_depth.json', 'r') as f: # Depth at every frame
    depth_data = json.load(f)

file_name = "/home/abven/CV_Course/badari_p3/P3Data/Assets/Vehicles/SUV.blend"

for (c,i),d_ in zip(enumerate(json_data),depth_data):

    # create a list of items and send to load_objects_into_frame
    Objects = []
    Orientations = []
    Locations = []
    Scales = []
    Cam_Locs = []

    print(f'Frame {c+1} Loading , {len(i["Objects"])} objects')

    for obj_c,obj_det in enumerate(i['Objects']) :

        c_name = obj_det['Class']
        box_2d = obj_det['Box_2d']
        box_3d = obj_det['Box_3d']
        orien = obj_det['Orientation']
        loc = obj_det['Location']

        cent = np.mean(box_3d,axis=0)
        dep = np.array(d_['depth'])
        z_val = dep[int(cent[1]), int(cent[0])]
        cent = find_xyz(R,K,cent,z_val)
        cent = [cent[0], cent[2], 0]
        scale_fac = (max(dep.ravel()) - min(dep.ravel()))/max(dep.ravel())
        # print(scale_fac)

        scale = np.array([.01,.01,.01]) * scale_fac
        orien , rot = obj_det['Orientation'] , obj_det['R']
        bird_view_orien = Matrix(((1, 0, 0),
                                    (0, 1, 0),
                                    (orien[0], orien[1], 0)))
        relative_view = bird_view_orien.transposed() @ Matrix(rot)
        euler_angles = relative_view.to_euler()

        if c_name in vehicles:
            Objects.append(generator.v_objects[vehicles.index(c_name)])
            Orientations.append(euler_angles)
            Locations.append(cent)
            Scales.append(scale)
            Cam_Locs.append((0,-5,1.5))

        elif c_name in entities:
            Objects.append(generator.en_objects[entities.index(c_name)])
            Orientations.append(euler_angles)
            Locations.append(cent)
            Scales.append(scale)
            Cam_Locs.append((0,-5,1.5))

    generator.load_objects_into_frame(Objects, Orientations, Locations, Scales, c+1, Cam_Locs[0])
    # bpy.ops.render.render(write_still=True)
    # output_path = f"/home/abven/CV_Course/badari_p3/Blender_outputs/render_{c}.png"
    # bpy.ops.image.save_as({"image": bpy.data.images['Render Result']}, filepath=output_path)
    # delete_all_objects_except_camera()
    generator.render_frame(f"/home/abven/CV_Course/badari_p3/Blender_outputs/", frame_number=c+1)
    #generator.delete_all_objects_except_camera()

    generator.delete_all_objects()
