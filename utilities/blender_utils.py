import bpy 
import math 
import json 
import numpy as np 
from mathutils import Matrix, Vector 

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_camera():
    # Create a new camera if none exists
    if not bpy.data.objects.get('Camera'):
        bpy.ops.object.camera_add()
    # Get the camera object
    camera = bpy.data.objects.get('Camera')  # Assuming the camera's name is 'Camera'
    return camera

def transform_camera_world(pts, R, K):
    # Return World Coordinates of the points
    # R is a 4*4 matrix 
    # K is a 3*3 matrix
    # pts is a tuple of (u, v)
    
    u, v = pts
    xyz = np.dot(np.array([u, v, 1]), np.linalg.inv(K).dot(np.linalg.inv(R)[:3, :4]))
    