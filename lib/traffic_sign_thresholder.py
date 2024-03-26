import numpy as np 

def get_traffic_signs(boxes_total, classes_total, scores_total, classes_names):
    
    traffic_signs = []
    for i in range(len(boxes_total)):
        if classes_names[classes_total[i]] == "traffic sign":
            traffic_signs.append(boxes_total[i])
    
    return traffic_signs

def traffic_sign_thresholder(image, boxes_total, classes_total, scores_total, classes_names):
    traffic_sign_bbox = get_traffic_signs(boxes_total, classes_total, scores_total, classes_names)
    