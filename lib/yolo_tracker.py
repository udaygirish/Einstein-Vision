import cv2
from ultralytics import YOLO
import pickle

def load_model_yworld(classes = ["car", "suv", "pickup truck" , "truck", "sedan", "person", "bicycle", "motorcycle", "green traffic light", "red traffic light", "yellow traffic light", "traffic cone", "road sign", "stop sign", "speed hump", "hump", "traffic cylinder", "parking meter", "dust bin", "trash can"]):
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
    print("YOLO WORLD MAIN Class Names: ", classes_names)
    # print("====================================")
    # print("Predictions")
    # print("Boxes: ", boxes_total)
    # print("Classes: ", classes_total)
    # print("Scores: ", scores_total)
    # print("Classes Names: ", classes_names)
    # print("====================================")
    return results, boxes_total, classes_total, scores_total, classes_names


def main():
    total_results = []
    model = load_model_yworld()
    video_path= "../../P3Data/2023-02-14_11-51-54-front_undistort.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        results_track_json = {}
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            results_track_json['bbox'] = results[0].pandas().xyxy[0].to_json(orient="records")
            results_track_json['results_total'] = results
            total_results.append(results_track_json)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    
    pickle.dump(total_results, open("yolo_track_results.pkl", "wb"))
    
if __name__ == '__main__':
    main()
    
    