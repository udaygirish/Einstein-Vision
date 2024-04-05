import numpy as np 
import cv2 
import os 
import sys 
sys.path.append("../")
sys.path.append("../Einstein-Vision")
sys.path.append("../Einstein-Vision/utilities")
from utilities.cv2_utilities import *
from ultralytics import YOLO
import imutils

def load_model_veh_ind():
    # classes
    #classes = ["car brake light on"]
    model = YOLO('lib/best.pt')
    #model.set_classes(classes)
    return model

def predict_image_ind_classes(model, img_path):
    results = model.predict(img_path)
    #results[0].show()
    boxes_total = results[0].boxes.xywh.cpu().numpy()
    classes_total = results[0].boxes.cls.cpu().numpy()
    scores_total = results[0].boxes.conf.cpu().numpy()
    total_labels = results[0].names
    classes_names = []
    for i in range(len(classes_total)):
        classes_names.append(total_labels[classes_total[i]])
        
    # print("====================================")
    # print("Predictions")
    # print("Boxes: ", boxes_total)
    # print("Classes: ", classes_total)
    # print("Scores: ", scores_total)
    # print("Classes Names: ", classes_names)
    # print("====================================")
    return results, boxes_total, classes_total, scores_total, classes_names

# parent_dir = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/new_test/"

# test_file = parent_dir + "frame_405.jpg"

# model = load_model()
# results, boxes_total, classes_total, scores_total, classes_names = predict_image(model, test_file)

# print(results)

# # Save the image with bounding boxes
# img = cv2.imread(test_file)

# print("Length of boxes: ", len(boxes_total))
# for i in range(len(boxes_total)):
#     x, y, w, h = boxes_total[i]
#     print("Class: ", classes_names[i])
#     print("x: ", x, "y: ", y, "w: ", w, "h: ", h)
#     cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
#     cv2.putText(img, classes_names[i], (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

# cv2.imwrite("output_brake.jpg", img)
# # Get cars 
# cars = []
# for i in range(len(classes_names)):
#     if classes_names[i] == "car":
#         cars.append(boxes_total[i])
        

# sample_bbox = cars[0]
# img = cv2.imread(test_file)
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# def crop_bbox(img, bbox):
#     # Here x, y is center of the bbox and w, h are width and height of the bbox
#     x, y, w, h = bbox
#     x1 = int(x - w/2)
#     y1 = int(y - h/2)
#     x2 = int(x + w/2)
#     y2 = int(y + h/2)
#     cropped_img = img[y1:y2, x1:x2]
#     return cropped_img

# crop_sample = crop_bbox(img, sample_bbox)

# # Get different thresholds 
# startRedLower = (0 , 50 , 50)
# startRedUpper = (10 , 255, 255)
# endRedLower = (160 , 100 , 100)
# endRedUpper = (180 , 255 , 255)

# blackLower = (0 , 0 , 0)
# blackUpper = (180 , 255 , 35)

# def confirm_day_or_night(frame , flag_night_counter):
#     mask = cv2.inRange(hsv, blackLower , blackUpper)
#     #mask = cv2.erode(mask, None, iterations=2)
#     #mask = cv2.dilate(mask , None, iterations=2)
#     # Write temporary image
#     #cv2.imwrite('black_temp.jpg',imutils.resize(mask,width=250))
#     pixel_ct = 0
#     pixel_len = 0
#     for i in mask:
#       pixel_ct = pixel_ct + np.sum(i==0)
#       pixel_len = pixel_len + len(i)
#     ratio = pixel_ct / pixel_len
#     print("ratio = ",ratio)
#     if ratio < 0.68:
#         flag_night_counter = flag_night_counter + 1
#         return flag_night_counter
#     else:
#         flag_night_counter = flag_night_counter - 1 
#         return flag_night_counter
    
# day_night_flag = confirm_day_or_night(img, 1)
# print(day_night_flag)

# # Save crop sameple 
# cv2.imwrite("crop_sample.jpg", crop_sample)

# # Apply Gaussian Blur to the cropped image
# crop_blur = cv2.GaussianBlur(crop_sample, (7,7), 0)

# crop_hsv = cv2.cvtColor(crop_blur, cv2.COLOR_BGR2HSV)

# cv2.imwrite("crop_hsv.jpg", crop_hsv)

# # mask1 = cv2.inRange(crop_hsv, startRedLower, startRedUpper)
# # mask2 = cv2.inRange(crop_hsv, endRedLower, endRedUpper)
# # maskRed = mask1 + mask2
# # maskRed = cv2.erode(maskRed, None, iterations=2)
# # maskRed = cv2.dilate(maskRed, None, iterations=2)

# # print(crop_hsv.shape)

# # mask1 = np.stack((mask1,)*3, axis=-1)

# # # Save maskRed
# # cv2.imwrite("maskRed.jpg", mask1)

# # Threshold red color
# # mask1 = cv2.inRange(crop_hsv, startRedLower, startRedUpper)
# # mask2 = cv2.inRange(crop_hsv, endRedLower, endRedUpper)
# # maskRed = mask1 + mask2
# # low_red = np.array([161, 155, 84])
# # high_red = np.array([179, 255, 255])
# # maskRed = cv2.inRange(crop_hsv, low_red, high_red)
# # red = cv2.bitwise_and(crop_hsv, crop_hsv, mask=maskRed)

# # # Save maskRed
# # cv2.imwrite("maskRed.jpg", maskRed)

# # Threshold - Adaptive Thresholding on the HSV
# crop_gray = cv2.cvtColor(crop_hsv, cv2.COLOR_BGR2GRAY)
# crop_blur_gray = cv2.GaussianBlur(crop_gray, (7,7), 0)
# crop_thresh = cv2.adaptiveThreshold(crop_blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Find contours to get blobs with more than 100 pixels
# contours, _ = cv2.findContours(crop_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     if cv2.contourArea(contour) > 100:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(crop_thresh, (x, y), (x+w, y+h), (255, 255, 255), 2)


# # Save crop_thresh
# cv2.imwrite("crop_thresh.jpg", crop_thresh)

