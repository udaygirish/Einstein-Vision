import argparse
import time
from pathlib import Path
import cv2
import torch
import sys 
import os 
import shutil
import numpy as np


sys.path.append("../Drivable_Area/YOLOPv2")
sys.path.append("../Drivable_Area/YOLOPv2/utils")
sys.path.append("../Einstein-Vision/")
from utilities.cv2_utilities import *
# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages
    

    
base_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/Drivable_Area/YOLOPv2"

default_weights_path = base_path + "/data/weights/yolopv2.pt"


def detect(img_path, weights_path, img_size=640, conf_thres=0.3, iou_thres=0.45, device='cuda:0', save_conf=False, save_txt=False, nosave=True, classes=None, agnostic_nms=False):
    # setting and directories
    results_json  = {}
    bbox_list = []
    labels_list = []
    mask_green_list = []
    mask_red_list = []
    temp_img = cv2.imread(img_path)
    temp_img_shape = temp_img.shape 
    source, weights,  save_txt, imgsz =  img_path, weights_path, save_txt, img_size
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    model = model.eval()
    
    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        im0s = cv2.resize(im0s, (temp_img_shape[1], temp_img_shape[0]),  interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).to(device)
        print("Model Image Shape", img.shape)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        
        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        
        print("Im0s shape", im0s.shape)
        print("Seg Mask Shape", seg.shape)
        print("LL Mask Shape", ll.shape)
        print("Da Seg Mask Shape", da_seg_mask.shape)
        print("LL Seg Mask Shape", ll_seg_mask.shape)
        print("Temp Image Shape", temp_img_shape)
        print("Type of Da Seg Mask", type(da_seg_mask))
        da_seg_mask = cv2.resize(da_seg_mask, (temp_img_shape[1], temp_img_shape[0]),  interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = cv2.resize(ll_seg_mask, (temp_img_shape[1], temp_img_shape[0]),  interpolation=cv2.INTER_NEAREST)
        
        save_dir = Path("/home/udaygirish/Projects/WPI/computer_vision/project3/Drivable_Area/YOLOPv2/data/output")
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            print("Im0 shape", im0.shape)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        bbox_list.append(xyxy)
                        labels_list.append(cls)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            mask_green_list.append(da_seg_mask)  # Drivable area mask
            mask_red_list.append(ll_seg_mask) # Lane lines seg mask 
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    
            results_json["bbox"] = bbox_list
            results_json["labels"] = labels_list
            results_json["drivable_area_mask"] = mask_green_list
            results_json["lane_line_mask"] = mask_red_list
            

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    return results_json 


def main():
    img_size = 640
    conf_thres = 0.3
    iou_thres = 0.45
    device = '0'
    save_conf = False
    save_txt = True
    nosave = False  
    classes = None
    img_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/P3_road_arrow_frames/scene7/frame_2046.jpg"
    agnostic_nms = False
    img1 = cv2.imread(img_path)
    print("Original Image Shape", img1.shape)
    with torch.no_grad():
            results_json = detect(img_path, default_weights_path, img_size, conf_thres, iou_thres, device, save_conf, save_txt, nosave, classes, agnostic_nms)
    print("===================================="*3)
    print("Output - Results")
    print(results_json)
    print("===================================="*3)
    drivable_area_mask = results_json["drivable_area_mask"][0]
    lane_line_mask = results_json["lane_line_mask"][0]
    print("Lane Line Mask Shape", lane_line_mask.shape)
    img = cv2.imread(img_path)
    img_mask = dilate_image_with_mask(img, drivable_area_mask)
    edges = edge_detector(img_mask)
    output_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/Drivable_Area/YOLOPv2/data/output"
    save_image(edges, output_path+ "/" + "edges.jpg")
    
    contours = find_contours(edges)
    print("Length of Contours", len(contours))
    img = cv2.imread(img_path)
    contour_edges_list, hull_edges_list, hull_list, contour_list = multiple_contours_processing(contours, threshold=2)
    
    print("Length of filtered contours", len(contour_list))
    print("Length of filtered hulls", len(hull_list))

    # Filtering contours
    # contour_list, hull_list = filter_contours_hulls(contour_list, hull_list, threshold=100)
    
    contour_img = draw_contours(img, contour_list, hull_list, thickness=2)
    
    save_image(contour_img, output_path+ "/" + "contours.jpg")
    
    # img1 = cv2.imread(img_path)
    # contours, hierarchy = cv2.findContours(preprocess(img1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for cnt in contours:
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    #     hull = cv2.convexHull(approx, returnPoints=False)
    #     sides = len(hull)

    #     if 6 > sides > 3 and sides + 2 == len(approx):
    #         arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
    #         if arrow_tip:
    #             cv2.drawContours(img1, [cnt], -1, (0, 255, 0), 3)
    #             cv2.circle(img1, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
    
    # save_image(img1, output_path+ "/" + "arrow.jpg")

    

    
if __name__ == '__main__':
    main()