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

