import numpy as np
import sys 
sys.path.append("../../Einstein-Vision")
sys.path.append("../../Einstein-Vision/lib")
sys.path.append("../../Einstein-Vision/utilities")

from lib.ocrInference import *
from utilities.random_utils import *
from utilities.cv2_utilities import *
from utilities.blender_utils import *
from utilities.three_d_utils import *
from utilities.frame_extractor import *

from mathutils import Matrix, Vector 
from config.blender_config import YOLOCLASSES_TO_BLENDER, Blender_rot_scale
import cv2



def data_processor(results, image_path):
    '''
    Structure of the results
    results = {
        'lanes': lanes,
        'object_detection': {
            'boxes': boxes_total,
            'classes': classes_total,
            'scores': scores_total,
            'classes_names': classes_names
        },
        'pose_detection': pose_res,
        'depth': depth,
        'segmentation': seg_res,
        'lane_masks': lane_masks,
        'lane_boxes': lane_boxes,
        'lane_labels': lane_labels,
        'yolo3d': out_Objects_3d,
        'optical_flow': {
            'moving_labels': moving_labels_list,
            'filtered_boxes': filtered_boxes,
        },
        'object_detection_yworld': {
            'boxes': world_boxes,
            'classes': world_classes,
            'scores': world_scores,
            'classes_names': world_classes_names
        },
        'vehicle_indicators': {
            'vehicle_results': vehicle_results,
            'vehicle_boxes': vehicle_boxes,
            'vehicle_classes': vehicle_classes,
            'vehicle_scores': vehicle_scores,
            'vehicle_classes_names': vehicle_classes_names
        }
    }'''
    results_3d = {}
    # Get Lane 3d points 
    depth = results['depth']
    final_lanes = results['final_lanes']
    lane_bbox  = results['lane_boxes']
    scale_factor = get_scale_factor(depth)
    K, R = get_hardcoded_KR()
    
    total_lane_3d_points = []
    
    for i in range(len(final_lanes)):
        lane = final_lanes[i]
        lane_points = lane[0]
        lane_bbox = lane[1]
        lane_class = lane[2]
        
        lane_subsampled_points = sub_sample_points(lane_points, 1)
        lane_3d_pts, lane_3d_bbox = get_3d_lane_pts(R, K, lane_subsampled_points, depth, lane_bbox, scale_factor)

        lane_3d_pts = func_filter_3d_points(lane_3d_pts)
    
        total_lane_3d_points.append(lane_3d_pts)
    
    
    results_3d['lane_3d_points'] = total_lane_3d_points
    
    # Each object detection class has the following structure
    '''
    {
        '3d_world_coords': []
        'bbox_2d': []
        'class_name': ""
        'orientation': []
        'scale': []
        'score': 0.0
        'state_label': ""
        'avg_velocity': [],
        'track_id': 0,
        'pose_path': ""
    }
    
    List of the above dictionaries for each object detected
    '''
    # Object Detection Classes
    ob_bbox = results['object_detection']['boxes']
    ob_classes = results['object_detection']['classes']
    ob_scores = results['object_detection']['scores']
    ob_classes_names = results['object_detection']['classes_names']
    
    ob_poses = results['pose_detection']
    ob_seg = results['segmentation']
    ob_yolo3d = results['yolo3d']
    
    ob_world_boxes = results['object_detection_yworld']['boxes']
    ob_world_classes = results['object_detection_yworld']['classes']
    ob_world_scores = results['object_detection_yworld']['scores']
    ob_world_classes_names = results['object_detection_yworld']['classes_names']
    
    ob_vehicle_results = results['vehicle_indicators']['vehicle_results']
    ob_vehicle_boxes = results['vehicle_indicators']['vehicle_boxes']
    ob_vehicle_classes = results['vehicle_indicators']['vehicle_classes']
    ob_vehicle_scores = results['vehicle_indicators']['vehicle_scores']
    
    ob_vehicle_classes_names = results['vehicle_indicators']['vehicle_classes_names']
    
    moving_labels = results['optical_flow']['moving_labels']
    filtered_boxes = results['optical_flow']['filtered_boxes']
    
    print("Length of the Yolo3d: ", len(ob_yolo3d))
    print("Lenght of the Yolo world boxes: ", len(ob_world_boxes))
    print("Length of the vehicle boxes: ", len(ob_vehicle_boxes))
    print("Length of the filtered boxes: ", len(filtered_boxes))
    
    # For now neglect using the Yolov8 
    # Let us yolo world to make it fast 
    vehicle_types = ['car', "suv", "truck", "pickup_truck", "sedan", "motorcycle", "bicycle"]
    traffic_light_types = ["traffic_light", "green traffic light", "red traffic light",
                           "yellow traffic light"]
    
    total_objects = []
    visited_locations = []  ## Avoid duplicate objects or objects with same locations
    for box, class_, score, class_name in zip(ob_world_boxes, ob_world_classes, ob_world_scores, ob_world_classes_names):
        # box - x, y, w,h (x,y is the center and w and h are width and height)
        temp_obj_dict = {
        '3d_world_coords': [],
        'bbox_2d': [],
        'class_name': "",
        'orientation': [],
        'scale': [],
        'score': 0.0,
        'state_label': "",
        "moving_label": "",
        'avg_velocity': [],
        'track_id': 0,
        'pose_path': ""
        }
        if class_name in vehicle_types:
            distances = []
            obj_counter = []
            state_label = "NA"
            for i in range(len(ob_yolo3d)):
                c_name = ob_yolo3d[i]['Class']
                if c_name in vehicle_types:
                    temp_box_2d  = ob_yolo3d[i]['Box_2d']
                    x,y = np.mean(temp_box_2d, axis = 0)
                    distances.append(euc_distance([x,y], [box[0], box[1]]))
                    obj_counter.append(i)
            if len(distances) == 0:
                centroid = (box[0], box[1])
                z_val = depth[int(centroid[1]), int(centroid[0])]
                centroid = form2_conv_image_world(R, K, (centroid[0], centroid[1]), z_val)
                centroid = [centroid[0], centroid[2], 0]
                if centroid in visited_locations:
                    continue
                visited_locations.append(centroid)
                euler_angles = Blender_rot_scale[class_name]['orientation']
                scale = np.array(Blender_rot_scale[class_name]['scale'])
            else:
                min_index = np.argmin(distances)
                obj_det = ob_yolo3d[obj_counter[min_index]]
                box_2d = obj_det['Box_2d']
                class_name = obj_det['Class']
                orientation = obj_det['Orientation']
                cent = np.mean(box_2d, axis = 0)
                z_val = depth[int(cent[1]), int(cent[0])]
                centroid = form2_conv_image_world(R, K, (cent[0], cent[1]), z_val)
                centroid = [centroid[0], centroid[2], 0]
                
                if centroid in visited_locations:
                    continue
                visited_locations.append(centroid)
                orien, rot = obj_det['Orientation'], obj_det['R']
                
                bird_view_orien = Matrix([[1, 0, 0], [0, 1, 0], [orien[0], orien[1], 0]])
                relative_view = bird_view_orien.transposed() @ Matrix(rot)
                euler_angles = relative_view.to_euler()
                euler_angles += np.array(Blender_rot_scale[class_name]['orientation'])
                scale = np.array(Blender_rot_scale[class_name]['scale'])
            veh_class_distance = []
            if class_name in vehicle_types[:4]:
                for k in range(len(ob_vehicle_boxes)):
                    temp_x1, temp_y1 = ob_vehicle_boxes[k][0], ob_vehicle_boxes[k][1]
                    temp_x2, temp_y2 = box[0], box[1]
                    veh_class_distance.append(euc_distance([temp_x1, temp_y1], [temp_x2, temp_y2]))
                if len(veh_class_distance) == 0:
                    state_label = "NA"
                else:
                    min_index = np.argmin(veh_class_distance)
                    state_label = ob_vehicle_classes_names[min_index]
            
            mov_label_distance = []
            moving_label = "Moving"  # Default label
            # Checking for the moving non moving label
            for k in range(len(filtered_boxes)):
                temp_x1, temp_y1 = filtered_boxes[k][0], filtered_boxes[k][1]
                temp_x2, temp_y2 = box[0], box[1]
                mov_label_distance.append(euc_distance([temp_x1, temp_y1], [temp_x2, temp_y2]))
                if len(mov_label_distance) == 0:
                    moving_label = "Moving"
                else:
                    min_index = np.argmin(mov_label_distance)
                    moving_label = moving_labels[min_index]
            temp_obj_dict['3d_world_coords'] = centroid
            temp_obj_dict['bbox_2d'] = box
            temp_obj_dict['class_name'] = class_name
            temp_obj_dict['orientation'] = euler_angles
            temp_obj_dict['scale'] = scale
            temp_obj_dict['score'] = score
            temp_obj_dict['state_label'] = state_label
            temp_obj_dict['moving_label'] = moving_label
            total_objects.append(temp_obj_dict)
            
        elif class_name in traffic_light_types:
            distances = []
            for i in range(len(ob_yolo3d)):
                c_name = ob_yolo3d[i]['Class']
                temp_box_2d  = ob_yolo3d[i]['Box_2d']
                x,y = np.mean(temp_box_2d, axis = 0)
                distances.append(euc_distance([x,y], [box[0], box[1]]))
            min_index = np.argmin(distances)
            obj_det = ob_yolo3d[min_index]
            box_2d = obj_det['Box_2d']
            class_name_ = obj_det['Class']
            orientation = obj_det['Orientation']
            cent = np.mean(box_2d, axis = 0)
            z_val = depth[int(cent[1]), int(cent[0])]
            centroid = form2_conv_image_world(R, K, (cent[0], cent[1]), z_val)
            centroid = [centroid[0], centroid[2], centroid[1]]
            if centroid in visited_locations:
                continue
            visited_locations.append(centroid)
            euler_angles = Blender_rot_scale[class_name]['orientation']
            scale = np.array(Blender_rot_scale[class_name]['scale'])
            
            temp_obj_dict['3d_world_coords'] = centroid
            temp_obj_dict['bbox_2d'] = box
            temp_obj_dict['class_name'] = class_name
            temp_obj_dict['orientation'] = euler_angles
            temp_obj_dict['scale'] = scale
            temp_obj_dict['score'] = score
        
        elif class_name == "person":
            # This is person class usually taken from the PyMAF
            continue
    
        elif class_name == "road sign":
            image = cv2.imread(image_path)
            bbox_img = image[int(box[1] - box[3]/2):int(box[1] + box[3]/2), int(box[0] - box[2]/2):int(box[0] + box[2]/2)]
            DetText, DetNumBool = ocr_run(bbox_img)
            if DetNumBool:
                centroid = [box[0], box[1]]
                z_val = depth[int(centroid[1]), int(centroid[0])]
                centroid = form2_conv_image_world(R, K, (centroid[0], centroid[1]), z_val)
                centroid = [centroid[0], centroid[2], 0]
                if centroid in visited_locations:
                    continue
                visited_locations.append(centroid)
                class_name = f'parking meter'
                euler_angles = Blender_rot_scale[class_name]['orientation']
                scale = np.array(Blender_rot_scale[class_name]['scale'])
                temp_obj_dict['3d_world_coords'] = centroid
                temp_obj_dict['bbox_2d'] = box
                temp_obj_dict['class_name'] = class_name
                temp_obj_dict['orientation'] = euler_angles
                temp_obj_dict['scale'] = scale
                temp_obj_dict['score'] = score
            elif 'hump' in DetText: # DetTex.lower - Check later
                class_name = 'hump'
                centroid = [box[0], box[1]]
                z_val = depth[int(centroid[1]), int(centroid[0])]
                centroid = form2_conv_image_world(R, K, (centroid[0], centroid[1]), z_val)
                centroid = [centroid[0]-2, centroid[2]-2, 0]
                if centroid in visited_locations:
                    continue
                visited_locations.append(centroid)
                euler_angles = Blender_rot_scale[class_name]['orientation']
                scale = np.array(Blender_rot_scale[class_name]['scale'])
                temp_obj_dict['3d_world_coords'] = centroid
                temp_obj_dict['bbox_2d'] = box
                temp_obj_dict['class_name'] = class_name
                temp_obj_dict['orientation'] = euler_angles
                temp_obj_dict['scale'] = scale
                temp_obj_dict['score'] = score
            # else:
            #     centroid = [box[0], box[1]]
            #     z_val = depth[int(centroid[1]), int(centroid[0])]
            #     centroid = form2_conv_image_world(R, K, (centroid[0], centroid[1]), z_val)
            #     centroid = [centroid[0], centroid[2], 0]
            #     if centroid in visited_locations:
            #         continue
            #     visited_locations.append(centroid)
            #     euler_angles = Blender_rot_scale[class_name]['orientation']
            #     scale = np.array(Blender_rot_scale[class_name]['scale'])
            #     temp_obj_dict['3d_world_coords'] = centroid
            #     temp_obj_dict['bbox_2d'] = box
            #     temp_obj_dict['class_name'] = class_name
            #     temp_obj_dict['orientation'] = euler_angles
            #     temp_obj_dict['scale'] = scale
            #     temp_obj_dict['score'] = score
        else:
            centroid = [box[0], box[1]]
            z_val = depth[int(centroid[1]), int(centroid[0])]
            centroid = form2_conv_image_world(R, K, (centroid[0], centroid[1]), z_val)
            centroid = [centroid[0], centroid[2]-3, 0]
            if centroid in visited_locations:
                continue
            visited_locations.append(centroid)
            euler_angles = Blender_rot_scale[class_name]['orientation']
            scale = np.array(Blender_rot_scale[class_name]['scale'])
            temp_obj_dict['3d_world_coords'] = centroid
            temp_obj_dict['bbox_2d'] = box
            temp_obj_dict['class_name'] = class_name
            temp_obj_dict['orientation'] = euler_angles
            temp_obj_dict['scale'] = scale
            temp_obj_dict['score'] = score
        total_objects.append(temp_obj_dict)
        
    results_3d['objects'] = total_objects
    return results_3d

    