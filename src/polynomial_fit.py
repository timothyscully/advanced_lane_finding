import cv2
import numpy as np

class PolynomialFit:
    def __init__(self, mx, my):
        self._nwindows = 9
        self._margin = 100
        self._minpix = 50

        self._mx = mx
        self._my = my

        self._left_fit = None
        self._right_fit = None

    def _eval_poly(self, fit, y):
        return fit[0] * (y ** 2) + fit[1] * y + fit[2]

    def _fit_polynomial(self, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def _full_histogram_search(self, img):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(img.shape[0] // self._nwindows)

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self._nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height

            win_xleft_low = int(leftx_current - self._margin)
            win_xleft_high = int(leftx_current + self._margin)
            win_xright_low = int(rightx_current - self._margin)
            win_xright_high = int(rightx_current + self._margin)

            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                              (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                               (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self._minpix:
                leftx_current = np.mean(nonzerox[good_left_inds])

            if len(good_right_inds) > self._minpix:
                rightx_current = np.mean(nonzerox[good_right_inds])

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self._left_fit, self._right_fit = self._fit_polynomial(leftx, lefty, rightx, righty)

    def _search_around_poly(self, img):
        margin = 100

        # Grab activated pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        right_min = (self._eval_poly(self._right_fit, nonzeroy) - margin)
        right_max = (self._eval_poly(self._right_fit, nonzeroy) + margin)

        left_min = (self._eval_poly(self._left_fit, nonzeroy) - margin)
        left_max = (self._eval_poly(self._left_fit, nonzeroy) + margin)

        left_lane_inds = ((nonzerox > left_min) & (nonzerox < left_max))
        right_lane_inds = ((nonzerox > right_min) & (nonzerox < right_max))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self._left_fit, self._right_fit = self._fit_polynomial(leftx, lefty, rightx, righty)

    def draw_result(self, warped_img, img, Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_img).astype(np.float32)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(200, img.shape[0] - 1, img.shape[0])

        leftx = self._eval_poly(self._left_fit, ploty)
        rightx = self._eval_poly(self._right_fit, ploty)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 1, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        newwarp = (newwarp * 255).astype(np.uint8)

        # Combine the result with the original image
        return cv2.addWeighted(img, 0.7, newwarp, 0.3, 0)

    def estimate_curvature(self, img):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        leftx = self._left_fit[0] * ploty ** 2 + self._left_fit[1] * ploty + self._left_fit[2]
        rightx = self._right_fit[0] * ploty ** 2 + self._right_fit[1] * ploty + self._right_fit[2]

        left_fit_cr = np.polyfit(ploty * self._my, leftx * self._mx, 2)
        right_fit_cr = np.polyfit(ploty * self._my, rightx * self._mx, 2)

        y_eval = np.max(ploty)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self._my + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self._my + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def calculate_distance(self, img):
        y_eval = img.shape[0] - 1

        left_max_x = self._left_fit[0] * y_eval ** 2 + self._left_fit[1] * y_eval + self._left_fit[2]
        right_max_x = self._right_fit[0] * y_eval ** 2 + self._right_fit[1] * y_eval + self._right_fit[2]

        return right_max_x - left_max_x



    def _sanity_check(self, img):
        left_cr, right_cr = self.estimate_curvature(img)

        if (abs(left_cr/right_cr) > 1.1) or (abs(left_cr/right_cr) < 0.9):
            return False

        return True

    def process(self, img):
        if self._left_fit is None or not self._sanity_check(img):
            self._full_histogram_search(img)
        else:
            self._search_around_poly(img)

        return self._left_fit, self._right_fit

