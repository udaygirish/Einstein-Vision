from ultralytics import YOLO



# # Initialize a YOLO-World model
# model = YOLO('yolov8x-worldv2.pt')  # or choose yolov8m/l-world.pt


# # Define custom classes
# model.set_classes(["car", "suv", "pickup truck" , "truck", "sedan", "person", "green traffic light", "red traffic light", "yellow traffic light", "traffic cone", "speed limit sign", "bicycle", "road sign", "stop sign", "speed breaker", "speed hump", "traffic cylinder"])
# # model.set_classes(["tail light"])
# # Execute prediction for specified categories on an image
# results = model.predict('../../P3Data/Images_1_9_10/Images_9/frame_830.jpg')

# # Show results
# results[0].show()


def load_model_yworld(classes = ["car", "suv", "pickup truck" , "dust bin", "trash can", "truck", "sedan", "person", "green traffic light", "red traffic light", "yellow traffic light", "traffic cone", "speed limit sign", "bicycle", "road sign", "stop sign", "speed breaker", "speed hump", "traffic cylinder", "parking meter"]):
    # classes
    classes = classes
    model = YOLO('yolov8x-worldv2.pt')
    model.set_classes(classes)
    return model

def predict_image_yworld(model, img_path):
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

def main():
    model = load_model_yworld()
    image_path= "../../P3Data/test_video_frames/frame_0289.png"
    total_new = predict_image_yworld(model, image_path)
    
if __name__ == '__main__':
    main()