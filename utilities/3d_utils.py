import bpy
import math
import json
import numpy as np
from mathutils import Matrix , Vector

def get_scale_factor(depth):
    scale_factor = (max(depth.ravel()) - min(depth.ravel()))/max(depth.ravel())
    return scale_factor

def form1_conv_image_world(R,K,pts) :
    u,v = pts
    
    xyz = np.dot(np.array([u, v, 1]), np.linalg.inv(K).dot(np.linalg.inv(R)[:3,:4]))
    
    return xyz[:3]

def form2_conv_image_world(R,K,pts, depth) :
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


    
# #    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, z_val))
# #    camera = bpy.context.object
# #    camera.name = "Camera"
#     camera.location = (0, -1, 1.5)
#     camera.rotation_euler = (1.57, 0, 0)
#     bpy.context.scene.frame_set(c)
#     for obj_c,obj_det in enumerate(i['Objects']) :
#         with bpy.data.libraries.load(file_name) as (data_from, data_to):
#             data_to.objects = data_from.objects
#         bpy.context.scene.frame_set(obj_c)
#         for obj, obj_fro in zip(data_to.objects, data_from.objects):
#             bpy.context.collection.objects.link(obj)
#             obj = bpy.data.objects.get(obj_fro.name)
            
#             if obj.name[:4] == 'Jeep' :
            
#                 box_3d = obj_det['Box_3d']
#                 cent = np.mean(box_3d,axis=0)
#                 z_val = depth[int(cent[1]), int(cent[0])]
#                 cent = find_xyz(R,K,cent,z_val)
#     #            cent = convert_to_3d(R,K,np.array(cent))
#                 cent = [cent[0], cent[2], 0]
#                 obj.location = cent
#     #            obj.location =  #(0,obj_det['Location'][1],obj_det['Location'][2])
#     #            obj.location = [cent[1],cent[2],-cent[0]]
#                 di = obj_det['Dim'] # (di[0]*scale_fac , di[1] , di[2]) #
# #                sca = [obj_fro.scale[0] * scale_fac,obj_fro.scale[1] * scale_fac,obj_fro.scale[2]* scale_fac]
# #                obj.scale = sca
#                 obj.scale = np.array([1,1,1]) * scale_fac #* obj_det['Dim']
#                 orien , rot = obj_det['Orientation'] , obj_det['R']
#                 bird_view_orien = Matrix(((1, 0, 0),
#                                             (0, 1, 0),
#                                             (orien[0], orien[1], 0)))
#                 relative_view = bird_view_orien.transposed() @ Matrix(rot)
#                 euler_angles = relative_view.to_euler()
#                 obj.rotation_euler = euler_angles
#     break

#cam = bpy.data.objects.get("Camera_F")
#bpy.context.scene.camera = camera