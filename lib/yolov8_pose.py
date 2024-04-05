from ultralytics import YOLO
# Output keypoints are in normalized format

def load_model(model_path= 'yolov8x-pose.pt'):
    model = YOLO(model_path)
    return model


def predict_image(model, img_path):
    total_results = {}
    results = model(img_path) 
    boxes_total = results[0].boxes.xywh.cpu().numpy()
    classes_total = results[0].boxes.cls.cpu().numpy()
    scores_total = results[0].boxes.conf.cpu().numpy()
    total_labels = results[0].names
    keypoints_total = results[0].keypoints.xy.cpu().numpy()
    classes_names = []
    for i in range(len(classes_total)):
        classes_names.append(total_labels[classes_total[i]])
    total_results['boxes'] = boxes_total
    total_results['classes'] = classes_total
    total_results['scores'] = scores_total
    total_results['classes_names'] = classes_names
    total_results['keypoints'] = keypoints_total
    return total_results

def main():
    model = load_model()
    image_path= "../../P3Data/test_video_frames/frame_0001.png"
    results = predict_image(model, image_path)  
    print(results)
    
if __name__ == '__main__':
    main()
