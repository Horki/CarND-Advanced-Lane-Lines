import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


if __name__ == '__main__':
    # prepare object points
    nx = 9 # number of inside corners in x
    ny = 6 # number of inside corners in y
    objpoints = []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objpoints.append(objp)

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration3.jpg')
    for idx, image in enumerate(images):
        imgpoints = []
        img = cv2.imread(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # # If found, draw corners
        if ret == True:
            # Draw and display the corners
            imgpoints.append(corners)
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            undistorted = cal_undistort(img, objpoints, imgpoints)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            # ax1.show()
            ax2.imshow(undistorted)
            ax2.set_title('Undistorted Image', fontsize=50)
            # ax2.show()
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
            print("../output_images/chess_{}".format(os.path.basename(image)))
            plt.savefig("../output_images/chess_{}".format(os.path.basename(image)))
        else:
            print("Ret: False", image)