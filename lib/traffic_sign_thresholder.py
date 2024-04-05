import numpy as np 
import cv2

def get_traffic_signs(boxes_total, classes_total, scores_total, classes_names):
    
    traffic_signs = []
    print("Classes Total:", classes_total)
    print("Classes Names: ", classes_names)
    for i in range(len(boxes_total)):
        if classes_names[i] == "traffic sign":
            traffic_signs.append(boxes_total[i])
    return traffic_signs

def traffic_sign_threshold(image, boxes_total, classes_total, scores_total, classes_names):
    traffic_sign_bbox = get_traffic_signs(boxes_total, classes_total, scores_total, classes_names)
    print("Length of Traffic Signs: ", len(traffic_sign_bbox))
    for i in range(len(traffic_sign_bbox)):
        x1, y1, x2, y2 = traffic_sign_bbox[i]
        crop_image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite("temp_outputs/traffic_sign_.jpg".format(i), crop_image)
        
def main():
    print("Doing Nothing")

if __name__ == "__main__":
    main()
        