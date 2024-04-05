import pickle 
import os 
import joblib 



def load_pose_data(data_path):
    data = joblib.load(data_path)
    return data

def get_pose_details(frame_no, out_data):
    pose_details = []
    pose_bbox = []
    for i in list(out_data.keys()):
        if frame_no in out_data[i]['frame_ids']:
            frame_ids_list = list(out_data[i]['frame_ids'])
            frame_index = frame_ids_list.index(frame_no)
            pose_bbox.append(out_data[i]['bboxes'][frame_index])
            file_obj_path = "meshes/{0}/{1}.obj".format(str(i).zfill(4), str(frame_no).zfill(6))
            pose_details.append(file_obj_path)
    return pose_details, pose_bbox


def main():
    base_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/Pose_Detection/PyMAF"
    out_data = load_pose_data(base_path + './output/test_vid/output.pkl')
    print("Out Data Keys: ", out_data[1].keys())
    for i in list(out_data.keys()):
        print(i)
        print(out_data[i]['frame_ids'])
    pose_details , pose_bbox = get_pose_details(2035, out_data)
    print("Pose Details: ", pose_details)
    print("Pose BBox: ", pose_bbox)

if __name__ == "__main__":
    main()
    