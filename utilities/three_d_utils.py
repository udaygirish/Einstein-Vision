import bpy
import math
import json
import numpy as np
from mathutils import Matrix , Vector
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

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

# Function to fit a Bezier curve to the points
def fit_bezier_curve(points, smooth=0.2):
    # Fit a Bézier curve to the points
    points = np.array(points)
    tck, _ = splprep(points.T, s=smooth)
    # Evaluate the Bézier curve at a higher resolution
    u_new = np.linspace(0, 1, num=100)
    curve_points = splev(u_new, tck)
    return curve_points

def sub_sample_points(points, n=5):
    sub_sampled_points = []
    # Sub sample the points
    n = min(n, len(points)//2)
    for i in range(0, len(points), n):
        sub_sampled_points.append(points[i])
    return sub_sampled_points

def get_3d_lane_pts(R, K, lane_points, depth, lane_bbox):
    lane_points = np.array(lane_points)
    lane_bbox = np.array(lane_bbox)
    lane_3d_pts = []
    lane_3d_bbox = []
    # Take first 75 percent only 
    for i in range(lane_points.shape[0]):
        point = lane_points[i]
        x = point[0]
        y = point[1]
        z = depth[int(y), int(x)]
        (x, y, z) = form2_conv_image_world(R, K, (x, y), z)
        lane_3d_pts.append((x, y, z))
        
    for i in range(len(lane_bbox)):
        point = lane_bbox[i]
        x = point[0]
        y = point[1]
        z = depth[int(y), int(x)]
        (x, y, z) = form2_conv_image_world(R, K, (x, y), z)
        lane_3d_bbox.append((x, y, z))
    
    return lane_3d_pts, lane_3d_bbox

# Function to filter the bezier curve points
def bezier_curve_filter(bezier_curve_points):
    filtered_bezier_curve_points = [[], [], []]
    for i in range(1, len(bezier_curve_points[0])):
        print("Difference: ", bezier_curve_points[2][i] - bezier_curve_points[2][i - 1])
        if bezier_curve_points[2][i] - bezier_curve_points[2][i - 1] < -0.05:
            for j in range(3):
                filtered_bezier_curve_points[j].append(bezier_curve_points[j][i])
    return filtered_bezier_curve_points
