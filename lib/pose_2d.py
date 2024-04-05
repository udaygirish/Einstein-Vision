from mmpose.apis import MMPoseInferencer
import cv2

def get_pose_2d(image_path, model_name='hrnet_w32_coco_256x192', device='cuda:0'):
    # Load the model
    pose_inferencer = MMPoseInferencer('human')

    # Load the image
    image = cv2.imread(image_path)

    # Perform inference
    pose_results = pose_inferencer.inference(image)
    
    pose_results =  next(pose_results)

    return pose_results


def main():
    image_path = "../../P3Data/test_video_frames/frame_0001.png"
    pose_results = get_pose_2d(image_path)
    print(pose_results)

if __name__ == '__main__':
    main()
