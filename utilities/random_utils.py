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
    
    for i in range(len(boxes)):
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

# def associate_bboxes(bbox1_list, bbox2_list, threshold = 10):
#     # Associate bounding boxes