import cv2 
import numpy as np


# Traditional Edge Detector 
def edge_detector(img):
    # Apply GaussianBlur to the image
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate and erode the image
    kernel = np.ones((3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def dilate_image_with_mask(img, mask):
    # Dilate the image using the mask
    print("Mask shape", mask.shape)
    print("Image shape", img.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(img, kernel, iterations=1)
    mask = mask.astype('uint8')
    # Apply the mask on the dilated image
    masked = cv2.bitwise_and(dilated, dilated, mask=mask)
    return masked

def find_contours(img):
    # Find the contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def single_contour_processing(contour):
    contour_edges = cv2.approxPolyDP(contour, 0.03*cv2.arcLength(contour, True), True)
    
    hull = cv2.convexHull(contour)
    hull_edges = cv2.approxPolyDP(hull, 0.03*cv2.arcLength(hull, True), True)
    
    return contour_edges, hull_edges, hull

# Function to filter contours and hulls based on area threshold
def filter_contours_hulls(contours, hulls, threshold=20):
    filtered_contours = []
    filtered_hulls = []
    for contour, hull in zip(contours, hulls):
        if cv2.contourArea(contour) < threshold:
            filtered_contours.append(contour)
            filtered_hulls.append(hull)
    return filtered_contours, filtered_hulls

def multiple_contours_processing(contours, threshold=2):
    contour_edges_list = []
    hull_edges_list = []
    hull_list = []
    contour_list = []
    for contour in contours:
        contour_edges, hull_edges, hull = single_contour_processing(contour)
        
        edge_differece = abs(len(contour_edges) - len(hull_edges))
        if edge_differece < threshold:
            contour_edges_list.append(contour_edges)
            hull_edges_list.append(hull_edges)
            hull_list.append(hull)
            contour_list.append(contour)
            
    return contour_edges_list, hull_edges_list, hull_list, contour_list

def draw_contours(img, contours, hulls,thickness=2):
    #cv2.drawContours(img, contours, -1, (0,255,0), thickness)
    cv2.drawContours(img, hulls, -1, (0,0,255), thickness)
    return img

def check_point_inside_bbox(point, bbox):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x+w, y+h
    if point[0] > x1 and point[1] > y1 and point[0] < x2 and point[1] < y2:
        return True
    return False

def filter_contours_by_area(contours, img, bbox=None):
    filtered_contours = []
    # Bounding box of lower 40 percent of the image
    bbox = (0, int(img.shape[0]*0.6), img.shape[1], int(img.shape[0]*0.4))
    for contour in contours:
        if cv2.contourArea(contour) > 40 and check_point_inside_bbox(contour[0][0], bbox):
            filtered_contours.append(contour)
            
    return filtered_contours
    
def find_arrow_direction(hull_list):
    directions = []
    # Find the direction of sharpest angle of convex hull
    for hull in hull_list:
        if len(hull) >=4:
            direction = find_arrow_orientation(hull)
            directions.append(direction)
        else:
            directions.append("None")
    return directions
                
                
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def find_arrow_orientation(contour):
    # Compute the centroid of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    centroid = (centroid_x, centroid_y)

    # Find the endpoints of the longest line segment from the centroid to the contour points
    max_distance = 0
    start_point = end_point = None
    for point in contour:
        pt = (point[0][0], point[0][1])
        distance = np.linalg.norm(np.array(pt) - np.array(centroid))
        if distance > max_distance:
            max_distance = distance
            start_point = pt

    # Calculate the angle of the line segment relative to the y-axis
    angle = np.arctan2(start_point[0] - centroid[0], start_point[1] - centroid[1]) * 180 / np.pi

    print(angle)
    # Determine the orientation of the arrow based on the angle
    if -30 <= angle <= 30:
        return "Up"
    elif 30 < angle <= 120:
        return "Up"
    elif -120 <= angle < -30:
        return "Down"
    else:
        return "Left"

# Save the image
def save_image(img, path):
    cv2.imwrite(path, img)
    print(f" The image is saved in: {path}")
    

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])