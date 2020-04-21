import cv2
import numpy as np

class PerspectiveTransform:
    def __init__(self, src_pts, dst_pts):
        self._M = cv2.getPerspectiveTransform(np.array(src_pts, dtype="float32"),
                                                  np.array(dst_pts, dtype="float32"))
        self._Minv = cv2.getPerspectiveTransform(np.array(dst_pts, dtype="float32"),
                                                  np.array(src_pts, dtype="float32"))

    def transform(self, img):
        return cv2.warpPerspective(img, self._M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)