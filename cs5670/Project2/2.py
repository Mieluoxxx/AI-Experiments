import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('new.png')
# 缩小采样率
scale_percent = 10 # 缩小50%
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# 显示新图像
plt.imshow(img_resized)
plt.show()