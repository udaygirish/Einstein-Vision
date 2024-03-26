import bpy
import math
import json
import numpy as np
from mathutils import Matrix , Vector

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

delete_all_objects()

# Create a new camera if none exists
if not bpy.data.objects.get('Camera'):
    bpy.ops.object.camera_add()

# Get the camera object
camera = bpy.data.objects.get('Camera')  # Assuming the camera's name is 'Camera'

with open('/home/abven/CV_Course/badari_p3/frames_2.json', 'r') as f:
    json_data = json.load(f)

with open('/home/abven/CV_Course/badari_p3/frames.json', 'r') as f:
    xy_json_data = json.load(f)

file_name = "/home/abven/CV_Course/badari_p3/P3Data/Assets/Vehicles/SUV.blend"
depth_file = "/home/abven/CV_Course/badari_p3/frame_583.txt"
depth = np.loadtxt(depth_file)

scale_fac = (max(depth.ravel()) - min(depth.ravel()))/max(depth.ravel())


def conv_im_world(R,K,pts) :
    u,v = pts
    
    xyz = np.dot(np.array([u, v, 1]), np.linalg.inv(K).dot(np.linalg.inv(R)[:3,:4]))

    # Convert to a C-Array (reshape if necessary)
    xyz = np.array(xyz, dtype=np.float32)
    
    return xyz[:3]

def convert_to_3d(R,K,pt) :
    r_inv = np.linalg.inv(R[:3,:3])
    k_inv = np.linalg.inv(K)
    
    xyz = np.dot(k_inv , r_inv)
    xyz = np.dot(np.array([pt[0] , pt[1] , 1]) , xyz)
    
    return xyz


def find_xyz(R, K, pts, depth):
    
    # Get the pixel coordinates
    u ,v = pts

    # Get the intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Calculate the x, y, z coordinates
#    scale = 2.5
#    depth = depth *1.5
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
#    x = scale_fac *  (u - cx) * depth / fx
#    y = scale_fac *  (v - cy) * depth / fy
#    z = depth*scale_fac
    xyz = np.array([x, y, z , 1]).T
    xyz = np.dot(R, xyz)
    xyz = xyz[:3]

    return xyz

# Road Rendering

#######
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
cube = bpy.context.object

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.transform.resize(value=(10, 1000, 0.05))  # Adjust the Z scale to make it thinner or thicker
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.shade_smooth()
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

cube.name = "Road"

material = bpy.data.materials.new(name="Road_Material")
material.diffuse_color = (0.2, 0.2, 0.2, 1)  # Adjust color as needed
material.specular_intensity = 0.1  # Adjust specular intensity as needed

if cube.data.materials:
    cube.data.materials[0] = material
else:
    cube.data.materials.append(material)
########


for c, i in enumerate(json_data):
         
    K = np.array([[1622.30674706393,0.0,681.0156669556608],
             [0.0,1632.8929856491513,437.0195537829288],
             [0.0,0.0,1.0]])

    R = np.array([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1.5],
        [0, 0, 0, 1]])
    
#    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, z_val))
#    camera = bpy.context.object
#    camera.name = "Camera"
    camera.location = (0, -5, 1.5)
    camera.rotation_euler = (1.57, 0, 0)
    bpy.context.scene.frame_set(c)
    for obj_c,obj_det in enumerate(i['Objects']) :
        with bpy.data.libraries.load(file_name) as (data_from, data_to):
            data_to.objects = data_from.objects
        bpy.context.scene.frame_set(obj_c)
        for obj, obj_fro in zip(data_to.objects, data_from.objects):
            bpy.context.collection.objects.link(obj)
            obj = bpy.data.objects.get(obj_fro.name)
            
            if obj.name[:4] == 'Jeep' :
            
                box_3d = obj_det['Box_3d']
                cent = np.mean(box_3d,axis=0)
                z_val = depth[int(cent[1]), int(cent[0])]
                cent = find_xyz(R,K,cent,z_val)
#                cent = convert_to_3d(R,K,cent)*z_val*scale_fac
                cent = [cent[0], cent[2], 0]
                obj.location = cent
    #            obj.location =  #(0,obj_det['Location'][1],obj_det['Location'][2])
    #            obj.location = [cent[1],cent[2],-cent[0]]
                di = obj_det['Dim'] # (di[0]*scale_fac , di[1] , di[2]) #
#                sca = [obj_fro.scale[0] * scale_fac,obj_fro.scale[1] * scale_fac,obj_fro.scale[2]* scale_fac]
#                obj.scale = sca
                obj.scale = np.array([1,1,1]) * scale_fac #* obj_det['Dim']
                orien , rot = obj_det['Orientation'] , obj_det['R']
                bird_view_orien = Matrix(((1, 0, 0),
                                            (0, 1, 0),
                                            (orien[0], orien[1], 0)))
                relative_view = bird_view_orien.transposed() @ Matrix(rot)
                euler_angles = relative_view.to_euler()
                obj.rotation_euler = euler_angles
    break