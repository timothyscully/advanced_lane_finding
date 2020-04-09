import cv2
import matplotlib.image as mpimg
import numpy as np


class Thresholder:
    def __init__(self, abs_sobel_thresh_x, abs_sobel_thresh_y, mag_thresh, dir_thresh):
        self._abs_sobel_thresh_x = abs_sobel_thresh_x
        self._abs_sobel_thresh_y = abs_sobel_thresh_y
        self._mag_thresh = mag_thresh
        self._dir_thresh = dir_thresh

    def _threshold(self, img, thresh):
        img = img / np.max(img)
        return ((img >= thresh[0]) & (img <= thresh[1]))

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 1)):
        img = np.copy(img)

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        else:
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        return self._threshold(abs_sobel, thresh)

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 1)):
        img = np.copy(img)

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        return self._threshold(np.sqrt(sobelx ** 2 + sobely ** 2), thresh)

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        img = np.copy(img)

        abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        return self._threshold(np.arctan2(abs_sobely, abs_sobelx), thresh)

    def process(self, img_fname):
        img = mpimg.imread(img_fname)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img_s = hls[:, :, 2]

        mask = (self.abs_sobel_thresh(img_s, orient='x', thresh=self._abs_sobel_thresh_x) &
                self.abs_sobel_thresh(img_s, orient='y', thresh=self._abs_sobel_thresh_y) &
                self.mag_thresh(img_s, thresh=self._mag_thresh) &
                self.dir_threshold(img_s, thresh=self._dir_thresh))

        result = np.zeros_like(img_s)
        result[mask] = 1

        return result