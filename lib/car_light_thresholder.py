import cv2
import numpy as np

parent_dir = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/Sequences/scene10/Undist/"
cap = cv2.VideoCapture(parent_dir + "2023-03-06_19-48-30-front_undistort.mp4")
success, frame = cap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",0,255,nothing)
cv2.createTrackbar("U-S","Trackbars",0,255,nothing)
cv2.createTrackbar("U-V","Trackbars",0,255,nothing)

while True:
    success, frame = cap.read()
    
    frame = cv2.resize(frame, (640,480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H","Trackbars")
    l_s = cv2.getTrackbarPos("L-S","Trackbars")
    l_v = cv2.getTrackbarPos("L-V","Trackbars")
    u_h = cv2.getTrackbarPos("U-H","Trackbars")
    u_s = cv2.getTrackbarPos("U-S","Trackbars")
    u_v = cv2.getTrackbarPos("U-V","Trackbars")

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    
    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1)
    if key == 27:
        break
    