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

# Deselect all objects (optional but recommended)
bpy.ops.object.select_all(action='DESELECT')

# Select all objects in the scene
bpy.ops.object.select_all(action='SELECT')
# Delete selected objects
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
import cv2 
from blender.new_renderer import Blender_Utils 
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use('Agg')

blender_utils = Blender_Utils()

K = np.array([[1622.30674706393,0.0,681.0156669556608],
        [0.0,1632.8929856491513,437.0195537829288],
        [0.0,0.0,1.0]])

R = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1.5],
    [0, 0, 0, 1]])


# Make Lanes 

load_pickle_data = open_pickle_file(DATA_PATH + "results.pkl")
img_no = 1
print("Keys: ", load_pickle_data.keys())

data_keys = list(load_pickle_data.keys())
output_data = load_pickle_data[data_keys[img_no]]

final_lanes = output_data['final_lanes']
depth = output_data['depth']
print(data_keys)
# print("Final Lanes: ", final_lanes)

img = cv2.imread(DATA_PATH + data_keys[img_no])
img1 = img.copy()
print(img.shape)

for i in range(len(final_lanes)):
    lane = final_lanes[i]
    print("====================================")
    lane_points = lane[0]
    lane_bbox = lane[1]
    lane_class = lane[2]
    
    # Print the Lane class 
    print("Lane Class: ", lane_class)
    if lane_class == "solid-line":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)  
    print("Lane points shape: ", lane_points.shape)
    for j in range(lane_points.shape[0]):
        # Draw point on image 
        point = lane_points[j]
        
        cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
    
# Save the image
cv2.imwrite("lane_points.png", img)

print("Length of Lane Points: ", len(final_lanes))

# print("Length of Lane Points BBX: ", len(final_lanes[1][0]))
blender_utils.create_road_surface()
scale_factor = get_scale_factor(depth)
# Create a empty image x 
lane_total_3d_pts = []



def func_filter_3d_points(lane_3d_pts):
    filtered_lane_3d_pts = []
    for i in range(1, len(lane_3d_pts)):
        if lane_3d_pts[i][2] - lane_3d_pts[i - 1][2] < 0:
            break
        else:
            filtered_lane_3d_pts.append(lane_3d_pts[i])
    return filtered_lane_3d_pts

for i in final_lanes[:3]:
    single_lane = i 
    lane_points = single_lane[0]
    lane_bbox = single_lane[1]
    lane_class = single_lane[2]
    
    lane_subsampled_points = sub_sample_points(lane_points, 1)
    
    
    lane_3d_pts, lane_3d_bbox = get_3d_lane_pts(R, K, lane_subsampled_points, depth, lane_bbox, scale_factor)
    # print("===================================="*3)
    # print(lane_subsampled_points)
    # print("===================================="*3)
    # print(lane_3d_pts)
    # print("===================================="*3)
    
    print("Lane 3d points before filtering", len(lane_3d_pts))
    sorted(lane_3d_pts, key=lambda x: x[2])
    lane_3d_pts = func_filter_3d_points(lane_3d_pts)
    print("Lane 3d points after filtering", len(lane_3d_pts))
    
    temp_x, temp_z = [], []
    for i in range(len(lane_3d_pts)):
        # Plot the x and z points
        point = lane_3d_pts[i]
        temp_x.append(point[0])
        temp_z.append(point[2])
    #     # Plot a lane marking in matplotlib
    lane_total_3d_pts.append({"x": temp_x, "z": temp_z})
        # Sghow the plot
    # Plot with matplotlib
    
    
    
    
    
    # for i in range(len(lane_3d_pts)):
    #     point = lane_3d_pts[i]
    #     # Plot a lane marking in blender 
    #     bpy.ops.mesh.primitive_cube_add(size=0.1, location=point)
    #     obj = bpy.context.active_object
    #     obj.name = "Lane_Marking"
    #     obj.scale = (0.1, 0.1, 0.1)
    #     obj.location = (point[0], point[2], 0)
    
    
    filtered_lane_3d_pts = np.array(lane_3d_pts)
    
    bezier_curve_points = fit_bezier_curve(filtered_lane_3d_pts, smooth=0.5)
    print("Bezier Curve Points: ", bezier_curve_points)
    
    bezier_curve_points = bezier_curve_filter(bezier_curve_points)
    bezier_curve_points = lane_3d_pts
    # Create curve object in Blender
    curve_obj = blender_utils.create_bezier_curve_from_points(bezier_curve_points)
    if lane_class == "solid-line":
        blender_utils.create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=10, gap_length=0, num_lanes=15)
    else:
        blender_utils.create_lane_markings_by_curve_length(curve_obj, lane_width=1, lane_length=10, gap_length=1, num_lanes=8)
        
print("Lane Generated")
    

colors = ["red", "blue", "green"]
print("Length of Lane Total 3D Points: ", len(lane_total_3d_pts))
for i in range(len(lane_total_3d_pts)):
    temp_lane = lane_total_3d_pts[i]
    plt.scatter(temp_lane["x"], temp_lane["z"], color= colors[i])


    
plt.xlabel("X")
plt.ylabel("Z")
plt.savefig("lane_points_plot.png")

    
    
    
    
    
    
    
    
    
    
    
    
    
    

# lane_subsampled_points = sub_sample_points(final_lanes[1][0], 5)

# # PLot the sub sampled points
# for i in range(len(lane_subsampled_points)):
#     point = lane_subsampled_points[i]
#     cv2.circle(img1, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)


# print("Lane Subsampled Points: ", len(lane_subsampled_points))
# cv2.imwrite("lane_subsampled_points.png", img1)
# print("Lane Subsampled Points After: ", len(lane_subsampled_points))

# lane_3d_pts, lane_3d_bbox = get_3d_lane_pts(R, K, lane_subsampled_points, depth, final_lanes[1][1])

# # print("Lane 3D Points: ", lane_3d_pts)

# # filtered_lane_3d_pts = remove_outliers(np.array(lane_3d_pts))
# filtered_lane_3d_pts = np.array(lane_3d_pts)




# # Fit a Bézier curve to the filtered points
# bezier_curve_points = fit_bezier_curve(filtered_lane_3d_pts)

# bezier_curve_points = bezier_curve_filter(bezier_curve_points)
# # Create curve object in Blender
# curve_data = bpy.data.curves.new(name="BezierCurve", type='CURVE')
# curve_data.dimensions = '3D'
# polyline = curve_data.splines.new('POLY')
# polyline.points.add(len(bezier_curve_points[0]))
# for i, (x, y, z) in enumerate(zip(bezier_curve_points[0], bezier_curve_points[1], bezier_curve_points[2])):
#     polyline.points[i].co = (x, z, 0, 1)  # Align in the Blender aixs and set 0 so that z becomes zero relatively all the lanes stay on
#     # the ground 

# # Create a new object with the curve
# curve_object = bpy.data.objects.new(name="BezierCurveObject", object_data=curve_data)
# bpy.context.collection.objects.link(curve_object)



