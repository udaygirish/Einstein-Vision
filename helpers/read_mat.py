# Script to read a Mat 

import scipy.io as sio
import numpy as np

def read_mat(file_path):
    """
    Read a .mat file.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        dict: Dictionary containing the data from the .mat file.
    """
    data = sio.loadmat(file_path)
    return data

# Example usage

abs_path_data = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/"
calibration_folder = "Calib/"
camera_type = "back/"
file_path = abs_path_data + calibration_folder + camera_type + "calibration.mat"
data = read_mat(file_path)

print(data)