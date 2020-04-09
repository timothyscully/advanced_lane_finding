import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.camera import Camera
from src.thresholder import Thresholder

cam = Camera()
thresholder = Thresholder(abs_sobel_thresh_x=(0.08, 0.4),
                          abs_sobel_thresh_y=(0.08, 0.4),
                          mag_thresh=(0.12, 0.4),
                          dir_thresh=(0, np.pi / 2))

#cam.calibrate('camera_cal/*.jpg', (9,6))

img = thresholder.process('test_images/test5.jpg')

plt.imshow(img)
plt.show()
