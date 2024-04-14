import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(img):
    # 找到图像的最小值和最大值
    min_val = np.min(img)
    max_val = np.max(img)

    # 对图像进行归一化
    img_normalized = (img - min_val) * (255 / (max_val - min_val))

    # 确保值在 0 到 255 之间
    img_normalized = np.clip(img_normalized, 0, 255)

    # 转换数据类型为 uint8
    img_normalized = img_normalized.astype(np.uint8)

    return img_normalized


# 读取两张图像
imgA = cv2.imread('dog.png')
imgB = cv2.imread('cat.png')

# 使两张图像大小一致
imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

# 将图像转换为灰度图
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# 计算图像A的频谱
fA = np.fft.fft2(grayA)
fshiftA = np.fft.fftshift(fA)
magnitude_spectrumA = 20 * np.log(np.abs(fshiftA))

# 显示图像A及其频谱
plt.figure(1)
plt.subplot(121)
plt.imshow(grayA, 'gray')
plt.subplot(122)
plt.imshow(magnitude_spectrumA, 'gray')

# 获取图像大小和频谱中心点
rows, cols = grayA.shape
crow, ccol = int(rows/2), int(cols/2)

# 构造A图像的低通滤波器
maskA = np.zeros((rows, cols), np.uint8)
maskA[crow-30:crow+30, ccol-30:ccol+30] = 1
md = fshiftA * maskA
epsilon = 1e-10
magnitude_spectrumA_md = 20 * np.log(np.abs(md) + epsilon)

# 低通滤波并显示结果
ishiftA_low = np.fft.ifftshift(md)
iimgA_low = np.fft.ifft2(ishiftA_low)
iimgA_low = np.abs(iimgA_low)
iimgA_low = cv2.normalize(iimgA_low, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

plt.figure(2)
plt.subplot(121)
plt.imshow(magnitude_spectrumA_md, 'gray')
plt.subplot(122)
plt.imshow(iimgA_low, 'gray')

# 构造A图像的高通滤波器
fshiftA[crow-10:crow+10, ccol-10:ccol+10] = 0
ishiftA_high = np.fft.ifftshift(fshiftA)
iimgA_high = np.fft.ifft2(ishiftA_high)
iimgA_high = np.abs(iimgA_high)
iimgA_high = normalize_image(iimgA_high)
magnitude_spectrumA_fshiftA = 20 * np.log(np.abs(fshiftA)+1)

plt.figure(3)
plt.subplot(121)
plt.imshow(magnitude_spectrumA_fshiftA, 'gray')
plt.subplot(122)
plt.imshow(iimgA_high, 'gray')

# 计算图像B的频谱
fB = np.fft.fft2(grayB)
fshiftB = np.fft.fftshift(fB)
magnitude_spectrumB = 20 * np.log(np.abs(fshiftB))

# 显示图像B及其频谱
plt.figure(4)
plt.subplot(121)
plt.imshow(grayB, 'gray')
plt.subplot(122)
plt.imshow(magnitude_spectrumB, 'gray')

# 构造B图像的低通滤波器
maskB = np.zeros((rows, cols), np.uint8)
maskB[crow-30:crow+30, ccol-30:ccol+30] = 1
md_B = fshiftB * maskB
epsilon = 1e-10
magnitude_spectrumB_md = 20 * np.log(np.abs(md_B) + epsilon)

# 低通滤波并显示结果
ishiftB_low = np.fft.ifftshift(md_B)
iimgB_low = np.fft.ifft2(ishiftB_low)
iimgB_low = np.abs(iimgB_low)
iimgB_low = normalize_image(iimgB_low)

plt.figure(5)
plt.subplot(121)
plt.imshow(magnitude_spectrumB_md, 'gray')
plt.subplot(122)
plt.imshow(iimgB_low, 'gray')

# 构造B图像的高通滤波器
fshiftB[crow-10:crow+10, ccol-10:ccol+10] = 0
ishiftB_high = np.fft.ifftshift(fshiftB)
iimgB_high = np.fft.ifft2(ishiftB_high)
iimgB_high = np.abs(iimgB_high)
iimgB_low = normalize_image(iimgB_high)
magnitude_spectrumB_fshiftB = 20 * np.log(np.abs(fshiftB)+1)

plt.figure(6)
plt.subplot(121)
plt.imshow(magnitude_spectrumB_fshiftB, 'gray')
plt.subplot(122)
plt.imshow(iimgB_high, 'gray')

# A图像低频与B图像高频融合
fshiftB[crow-20:crow+20, ccol-20:ccol+20] = md[crow-20:crow+20, ccol-20:ccol+20]
ishiftC = np.fft.ifftshift(fshiftB)
iimgC = np.fft.ifft2(ishiftC)
iimgC = np.abs(iimgC)
iimgC = normalize_image(iimgC)
magnitude_spectrumC = 20 * np.log(np.abs(fshiftB))

# 显示融合结果
plt.figure(7)
plt.imshow(iimgC, 'gray')
plt.show()
