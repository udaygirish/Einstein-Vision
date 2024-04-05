from ultralytics import YOLO

def load_model(model_path= 'yolov8n-seg.pt'):
    model = YOLO(model_path)
    return model


def predict_image(model, img_path):
    results = model(img_path) 
    return results


def main():
    model = load_model()
    image_path= "../../P3Data/test_video_frames/frame_0001.png"
    results = predict_image(model, image_path)  
    print(results)
    
    
if __name__ == '__main__':
    main()
