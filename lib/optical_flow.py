from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
import os.path as osp
import cv2 
import sys 
sys.path.append("../")
sys.path.append("../Einstein-Vision")
sys.path.append("../Einstein-Vision/utilities")
from utilities.cv2_utilities import *
from ultralytics import YOLO


def load_model():
    # classes
    classes = ["car", "suv", "pickup truck", "truck", "sedan", "person", "bicycle"]
    model = YOLO('yolov8x-worldv2.pt')
    model.set_classes(classes)
    return model

def predict_image(model, img_path):
    results = model.predict(img_path)
    results[0].show()
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



config_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.py'
checkpoint_file = 'models/raft_8x2_100k_flyingthings3d_sintel_368x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
parent_dir = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/test_video_frames/"
file1 = parent_dir + "frame_0231.png"
file2 = parent_dir + "frame_0244.png"
output = inference_model(model,file1, file2)

result_out = "flow_output"

print("Result Shape: ", output.shape)
# visualize the optical flow
visualize_flow(output, osp.join("./", f'{result_out}.png'))
write_flow(output, osp.join("./", f'{result_out}.flo'))

flow_out = cv2.imread(f'{result_out}.png')
print(flow_out.shape)

model_yolo = load_model()

results, boxes_total, classes_total, scores_total, classes_names = predict_image(model_yolo, file2)

imgk = cv2.imread(file2)

for i in range(len(boxes_total)):
    x = int(boxes_total[i][0])
    y = int(boxes_total[i][1])
    w = int(boxes_total[i][2])
    h = int(boxes_total[i][3])
    cv2.rectangle(imgk, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), (0, 255, 0), 2)
    cv2.putText(imgk, classes_names[i],  (int(x-w//2), int(y-h//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
cv2.imwrite("output.png", imgk)


# Find point correspondences between the two images
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)

# convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fundamental_matrix  = compute_fundamental_matrix_sift(img1, img2)
print("Fundamental Matrix: ", fundamental_matrix)

for i in range(len(boxes_total)):
    x = int(boxes_total[i][0])
    y = int(boxes_total[i][1])
    w = int(boxes_total[i][2])
    h = int(boxes_total[i][3])
    x_min = int(x-w//2)
    y_min = int(y-h//2)
    x_max = int(x+w//2)
    y_max = int(y+h//2)
    bbox = (x_min, y_min, x_max, y_max)
    flow_field = flow_out
    sampson_distance_out = calculate_movement(img1_gray, img2_gray, bbox, flow_field, fundamental_matrix)
    print("Sampson Distance: ", sampson_distance_out)
    # if sampson_distance_val > 30000:
    #     sampson_distance_out = True
    # else:
    #     sampson_distance_out = False
    if sampson_distance_out == True:
        label_moving = "Moving"
    else:
        label_moving = "Stationary"
    #label_moving = str(round(sampson_distance_val, 2))
    #label_moving = sampson_distance_val
    cv2.rectangle(img2, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), (0, 255, 0), 2)
    cv2.putText(img2, label_moving,  (int(x-w//2), int(y-h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

# Save the image
cv2.imwrite("output_sampson.png", img2)


