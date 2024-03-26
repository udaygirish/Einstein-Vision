import numpy as np

def check_point_in_box(point, box):
    x, y = point
    x1, y1 = box[0]
    x2, y2 = box[1]
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return True
    return False

def check_number_of_points_in_box(points, box):
    
    count = 0
    for point in points:
        if check_point_in_box(point, box):
            count += 1
    return count

def lane_class_matcher(results):
    lanes_orig = results['lanes']
    lane_masks = results['lane_masks']
    lane_boxes = results['lane_boxes']
    lane_labels = results['lane_labels']
    
    print("Lengths of the Lanes")
    print("Lanes: ", len(lanes_orig))
    print("Lane Masks: ", len(lane_masks))
    print("Lane Boxes: ", len(lane_boxes))
    print("Lane Labels: ", len(lane_labels))
    
    # Lane BBOX shape: [(806,547),(1279,880)]
    get_lane_indices = dict()
    for lane_id in range(len(lanes_orig)):
        get_lane_indices[lane_id] = []
        
    for lane_b_id in range(len(lane_boxes)):
        for lane_id in range(len(lanes_orig)):
            count = check_number_of_points_in_box(lanes_orig[lane_id], lane_boxes[lane_b_id])
            get_lane_indices[lane_id].append(count)
    
    get_max_indices = dict()
    for lane_id in range(len(lanes_orig)):
        get_max_indices[lane_id] = get_lane_indices[lane_id].index(max(get_lane_indices[lane_id]))
    
    print("Max Indices: ", get_max_indices)
    
    final_lanes = []
    for lane_id in range(len(lanes_orig)):
        temp_lane = (lanes_orig[lane_id], lane_boxes[get_max_indices[lane_id]], lane_labels[get_max_indices[lane_id]])
        final_lanes.append(temp_lane)
        
    return final_lanes