# General CV2 Camera Calibration Function Implementation
import sys
sys.path.append("../")
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from logger import setup_logger


class CamcalibCV2:

    def __init__(
        self,
        ImagesPath="Calibration_Imgs/*.jpg",
    ):
        """
        Initialize the CamcalibCV2 class.

        Args:
            ImagesPath (str): Path to the calibration images. Default is "Calibration_Imgs/*.jpg".
        """
        self.description = "Camera Calibration Class"
        # Arrays to store object points and image points
        self.objpoints = []
        self.imgpoints = []

        # Read and Make a List of Calibration images
        self.images = glob.glob(ImagesPath)

        # Prepare object points - Ex: (0,0,0), (1,0,0), (2,0,0),...
        self.objp = np.zeros((6 * 9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    def calib(self, images_list=[]):
        """
        Perform camera calibration using the given list of images.

        Args:
            images_list (list): List of image file paths. If empty, uses the images specified during initialization.

        Returns:
            tuple: Camera matrix (mtx) and distortion coefficients (dst).
        """
        if len(images_list) == 0:
            images_list = self.images
        else:
            pass

        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for fname in images_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the Chessboard Corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If corners found add object and image points

            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
            else:
                continue

            # Get Camera Matrix, distortion Coefficients, Rotational and Translation vectors,
            # for Image Undistortion

        ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )

        self.mtx = mtx
        self.dst = dst
        self.rvecs = rvecs
        self.tvecs = tvecs

        return mtx, dst

    def undistort(self, img, mtx, dst):
        """
        Undistort the given image using the camera matrix and distortion coefficients.

        Args:
            img (numpy.ndarray): Input image.
            mtx (numpy.ndarray): Camera matrix.
            dst (numpy.ndarray): Distortion coefficients.

        Returns:
            tuple: Undistorted image, new camera matrix, and region of interest (roi).
        """
        width = img.shape[0]
        height = img.shape[1]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dst, (width, height), 0, (width, height)
        )
        undistorted_image = cv2.undistort(img, mtx, dst, None)  # , newcameramatrix)
        return undistorted_image, newcameramatrix, roi

    def calculate_projection_err(self):
        """
        Calculate the projection error.

        Returns:
            float: Total projection error.
        """
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dst
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                imgpoints2
            )

            mean_error += error

        mean_error = round((mean_error / len(self.objpoints)), 6)
        return mean_error
    

def write_calib_txt(mtx, dst, path="calib.txt"):
    """
    Write the camera matrix and distortion coefficients to a text file.

    Args:
        mtx (numpy.ndarray): Camera matrix.
        dst (numpy.ndarray): Distortion coefficients.
        path (str): Path to save the text file. Default is "calib.txt".
    """
    with open(path, "w") as f:
        f.write("Camera Matrix\n")
        # dont write as string Write individual elements
        for i in mtx:
            for j in i:
                f.write(str(j) + " ")
        f.write("\n\nDistortion Coefficients\n")
        for i in dst:
            for j in i:
                f.write(str(j) + " ")
        f.close()
    print(f"Camera Matrix and Distortion Coefficients saved to {path}")
    
def read_calib_txt(path="calib.txt"):
    """
    Read the camera matrix and distortion coefficients from a text file.

    Args:
        path (str): Path to the text file. Default is "calib.txt".

    Returns:
        tuple: Camera matrix and distortion coefficients.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        # Read Camera Matrix
        mtx = np.array([float(i) for i in lines[1].strip().split()])
        dst = np.array([float(i) for i in lines[4].strip().split()])
        # Reshape the arrays
        mtx = mtx.reshape(3, 3)
        dst = dst.reshape(1, 5)
        f.close()
    return mtx, dst
    
def main():
    # Initialize the class
    abs_path_data = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/"
    calibration_folder = "Calib/"
    camera_type = "front/"
    image_path = abs_path_data + calibration_folder + camera_type + "frame*.jpg"
    output_path = abs_path_data + calibration_folder + camera_type + "calibration.txt"
    cam_calib = CamcalibCV2(ImagesPath=image_path)
    # Perform Camera Calibration
    mtx, dst = cam_calib.calib()
    print("Original Camera Matrix: ", mtx)
    print("Original Distortion Coefficients: ", dst)
    # Calculate the projection error
    proj_err = cam_calib.calculate_projection_err()
    print(f"Projection Error: {proj_err}")
    # Save the camera matrix and distortion coefficients
    write_calib_txt(mtx, dst, path=output_path)
    
    read_mtx, read_dst = read_calib_txt(path=output_path)
    print(f"Camera Matrix: {read_mtx}")
    print(f"Distortion Coefficients: {read_dst}")

if __name__ == "__main__":
    main()
    