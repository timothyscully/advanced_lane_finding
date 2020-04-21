import glob

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.camera import Camera
from src.pespective_transform import PerspectiveTransform
from src.polynomial_fit import PolynomialFit
from src.thresholder import Thresholder

cam = Camera()
cam.calibrate('camera_cal/calibration*.jpg', (9, 6))

thresholder = Thresholder(abs_sobel_thresh_x=(20, 90),
                          abs_sobel_thresh_y=(20, 90),
                          mag_thresh=(90, 100),
                          dir_thresh=(0.9, 1.0))

src_pts = [[560, 500], [800, 500], [1190, 720], [240, 720]]
dst_pts = [[200, 0], [1200, 0], [1200, 720], [200, 720]]
transformer = PerspectiveTransform(src_pts, dst_pts)
poly_fit = PolynomialFit(3.7 / 700, 30 / 720)

#img_fname = 'test_images/test1.jpg'
img_fname = 'challenge_images/out-001.jpg'
img = mpimg.imread(img_fname)  # Read image


# img_fname = 'camera_cal/calibration1.jpg'

def process_image(img):
    orig_img = cam.undistort(img)  # Undistort image
    img = thresholder.process(orig_img)  # Process image
    #img = transformer.transform(img)  # Perspective transform image
    #poly_fit.process(img)  # Fit polynomial
    #import ipdb; ipdb.set_trace()

    #img = poly_fit.draw_result(img, orig_img, transformer._Minv)

    return img


# pts = np.array(src_pts).reshape((-1, 1, 2))
# cv2.polylines(img, [pts], True, (0, 255, 255), thickness=2)
# i = 0
# for img_fname in glob.glob('challenge_images/*.png'):
#     print(f'{i}')
#     img = mpimg.imread(img_fname)
#     out_img = process_image(img)
#     mpimg.imsave(f'out/out-{i}.png', out_img)
#
#     i += 1

img = process_image(img)

#clip1 = VideoFileClip("challenge_video.mp4")
#white_clip = clip1.fl_image(process_image)
#white_output = 'challenge_video_out.mp4'
#white_clip.write_videofile(white_output, audio=False)


# print(curv_estimator.process(img, left_fit, right_fit))
#cv2.polylines(img, [np.array(src_pts).reshape((-1, 1, 2))], True, (0, 255, 255), thickness=12)
plt.imshow(img)
plt.show()
