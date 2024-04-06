import numpy as np 
import cv2 

def get_filter_boxes(boxes, classes, scores, classes_names, filter_classes):
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []
    filtered_classes_names = []
    classes = list(classes)
    scores = list(scores)
    print("Type of classes: ", type(classes))
    print("Type of classes_names: ", type(classes_names))
    print("Type of filter_classes: ", type(filter_classes))
    print("Type of scores: ", type(scores))
    
    print("Class names: ", classes_names)
    for i in range(len(boxes)):
        print("Class: ", classes[i])
        if classes_names[int(classes[i])] in filter_classes:
            filtered_boxes.append(boxes[i])
            filtered_classes.append(classes[i])
            filtered_scores.append(scores[i])
            filtered_classes_names.append(classes_names[int(classes[i])])
            
    return filtered_boxes, filtered_classes, filtered_scores, filtered_classes_names

def calculate_bbox_closeness(bbox1, bbox2):
    
    # here x,y, w, h are the center coordinates of the bounding box and width and height of the bounding box
    x, y, w, h = bbox1
    x1, y1, w1, h1 = bbox2
    
    # Calculate the distance between the centers of the bounding boxes
    distance = np.sqrt((x-x1)**2 + (y-y1)**2)
    
    # Calculate the closeness of the bounding boxes
    return distance 

def euc_distance(p1, p2):
    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance

# def associate_bboxes(bbox1_list, bbox2_list, threshold = 10):
#     # Associate bounding boxes


def func_filter_3d_points(lane_3d_pts):
    filtered_lane_3d_pts = []
    for i in range(1, len(lane_3d_pts)):
        if lane_3d_pts[i][2] - lane_3d_pts[i - 1][2] < 0:
            break
        else:
            filtered_lane_3d_pts.append(lane_3d_pts[i])
    return filtered_lane_3d_pts