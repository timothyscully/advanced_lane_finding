import cv2
import numpy as np
import matplotlib.pyplot as plt

class Thresholder:
    def __init__(self, abs_sobel_thresh_x=None, abs_sobel_thresh_y=None, mag_thresh=None, dir_thresh=None):
        self._abs_sobel_thresh_x = abs_sobel_thresh_x
        self._abs_sobel_thresh_y = abs_sobel_thresh_y
        self._mag_thresh = mag_thresh
        self._dir_thresh = dir_thresh

    def _threshold(self, img, thresh):
        mask = np.zeros_like(img)
        mask[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return mask

    def abs_sobel_thresh(self, img_in, orient='x', sobel_kernel=3, thresh=(0, 1)):
        img = np.copy(img_in)

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        else:
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        abs_sobel = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8)

        return abs_sobel, self._threshold(abs_sobel, thresh)

    def mag_thresh(self, img, sobel_kernel=9, thresh=(0, 1)):
        img = np.copy(img)

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        mag_img = np.sqrt(sobelx ** 2 + sobely ** 2)
        mag_img = ((255 * mag_img) / np.max(mag_img)).astype(np.uint8)

        return mag_img, self._threshold(mag_img, thresh)

    def dir_threshold(self, img, sobel_kernel=15, thresh=(0, np.pi / 2)):
        img = np.copy(img)

        abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        dir_img = np.arctan2(abs_sobely, abs_sobelx)

        return dir_img, self._threshold(dir_img, thresh)

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        #plt.imshow(255 - hls[:, :, 0])
        #plt.show()

        mask = (cv2.inRange(hls, (90, 0, 80), (100, 255, 255)) == 255)

        #import ipdb
        #ipdb.set_trace()
        #mask = mask | (cv2.inRange(hls, (180, 0.4, 0.05), (290, 1, 0.7)) == 255)

        #img_s = hls[:, :, 2]

        #result = self.mag_thresh(hls[:,:,1], thresh=self._mag_thresh)
        #mask = self.abs_sobel_thresh(hls[:, :, 2], orient='x', thresh=self._abs_sobel_thresh_x)
        #(self.abs_sobel_thresh(gray, orient='x', thresh=self._abs_sobel_thresh_x) &
        #        self.abs_sobel_thresh(gray, orient='y', thresh=self._abs_sobel_thresh_y)) |
        #        (self.mag_thresh(gray, thresh=self._mag_thresh) &
        #        self.dir_threshold(gray, thresh=self._dir_thresh)))

        input_img = gray #hls[:, :, 1]
        gradx_img, gradx = self.abs_sobel_thresh(input_img, orient='x', thresh=self._abs_sobel_thresh_x)
        grady_img, grady = self.abs_sobel_thresh(input_img, orient='y', thresh=self._abs_sobel_thresh_y)
        mag_img, mag_binary = self.mag_thresh(input_img, thresh=self._mag_thresh)
        dir_img, dir_binary = self.dir_threshold(input_img, thresh=self._dir_thresh)

        #import ipdb
        #ipdb.set_trace()

        mask = mask | (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))

        result = np.zeros_like(gray)
        result[mask] = 1

        return result