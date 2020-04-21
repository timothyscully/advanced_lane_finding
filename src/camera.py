import cv2
import numpy as np
import glob


class Camera:
    def __init__(self):
        self._dist = None
        self._cam_mtx = None

    def calibrate(self, img_fname_pattern, chess_size, sub_pixel_search_window=(11, 11)):
        img_points = []  # 2D image points
        obj_points = []  # 3D points of chessboard

        # Generate 3D chess grid of corners
        chess_objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
        chess_objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

        sub_pixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        img_size = None  # Store image size, assume all images are the same size

        for img_fname in glob.glob(img_fname_pattern):
            img = cv2.imread(img_fname)  # Read image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray scale
            img_size = gray.shape if img_size is None else img_size

            found, corners = cv2.findChessboardCorners(gray, chess_size, None)

            if found:
                obj_points.append(chess_objp)

                corners = cv2.cornerSubPix(gray, corners, sub_pixel_search_window, (-1, -1),
                                           sub_pixel_criteria)  # Refine sub-pixel accuracy
                img_points.append(corners)

        _, self._cam_mtx, self._dist, _, _ = cv2.calibrateCamera(obj_points,
                                                                 img_points,
                                                                 img_size,
                                                                 None,
                                                                 None)

    def undistort(self, img):
        return cv2.undistort(img, self._cam_mtx, self._dist, None, self._cam_mtx)
