from transformers import pipeline
import cv2 
import numpy as np

def load_pipe():
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    return pipe

def get_depth(pipe,image_path):
    image = cv2.imread(image_path)
    depth = pipe(image)['depth']
    return depth

def main():
    pipe = load_pipe()
    image_path = "../../P3Data/test_video_frames/frame_0001.png"
    depth = get_depth(pipe,image_path)
    print(depth)
    
    return depth

if __name__ == '__main__':
    main()
