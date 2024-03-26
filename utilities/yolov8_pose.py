from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model

# Predict with the model
image_path = "../../P3Data/test_video_frames/frame_0001.png"
image_path = "https://ultralytics.com/images/bus.jpg"
results = model(image_path)  # predict on an image
print(results)
