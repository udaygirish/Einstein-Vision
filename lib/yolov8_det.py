from ultralytics import YOLO


def load_model():
    model = YOLO('yolov8x.pt')
    return model    

def predict_image(model, img_path):
    results = model(img_path) 
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
        
    
def main():
    model = load_model()
    image_path= "../../P3Data/test_video_frames/frame_0001.png"
    total_new = predict_image(model, image_path)  
    
if __name__ == '__main__':
    main()
