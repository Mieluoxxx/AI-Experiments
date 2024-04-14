import numpy as np


def create_gaussian_kernel(k, sigma):
    center = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(
                -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma**2)
            )
    kernel /= np.sum(kernel)
    return kernel


def conv_2d(img, k, sigma=None):
    assert k % 2 != 0, "卷积核应为奇数"
    pad = k // 2

    # 创建卷积核
    if sigma is None:  # 均值滤波
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    else:  # 高斯滤波
        kernel = create_gaussian_kernel(k, sigma)

    # 执行卷积操作
    if len(img.shape) == 2:  # 灰度图像
        H, W = img.shape
        # img_padded = np.pad(img, pad_width=pad, mode='constant')
        img_padded = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2))
        img_padded[pad:-pad, pad:-pad] = img
        out = np.zeros_like(img, dtype=np.float32)
        for h in range(H):
            for w in range(W):
                out[h, w] = (img_padded[h : h + k, w : w + k] * kernel).sum()
    else:  # 彩色图像
        H, W, C = img.shape
        # img_padded = np.pad(img, pad_width=(pad, pad, (0, 0)), mode='constant')
        img_padded = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2, C))
        img_padded[pad:-pad, pad:-pad] = img
        out = np.zeros_like(img, dtype=np.float32)
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    out[h, w, c] = (img_padded[h : h + k, w : w + k, c] * kernel).sum()

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


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

def my_filters(img, D=30, tag=None):
    assert tag=='low-pass' or tag=='high-pass', "please choose a tag in 'low-pass' or 'high-pass'"
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    H, W = img.shape
    crow, ccol = int(H/2), int(W/2)
    if tag=='low-pass':
        # 构造图像的低通滤波器
        mask = np.zeros((H, W), np.uint8)
        mask[crow-D:crow+D, ccol-D:ccol+D] = 1
        md = fshift * mask
        epsilon = 1e-10
        magnitude_spectrum_md = 20 * np.log(np.abs(md) + epsilon)

        # 低通滤波并显示结果
        ishift_low = np.fft.ifftshift(md)
        iimg_low = np.fft.ifft2(ishift_low)
        iimg_low = np.abs(iimg_low)
        iimg_low = normalize_image(iimg_low)

        return iimg_low, magnitude_spectrum_md
    
    else:
        # 构造A图像的高通滤波器
        fshift[crow-D:crow+D, ccol-D:ccol+D] = 0
        ishift_high = np.fft.ifftshift(fshift)
        iimg_high = np.fft.ifft2(ishift_high)
        iimg_high = np.abs(iimg_high)
        iimg_high = normalize_image(iimg_high)
        magnitude_spectrum_fshift = 20 * np.log(np.abs(fshift)+1)

        return iimg_high, magnitude_spectrum_fshift



def blend_images(image1, image2, d1=30, d2=10, d3=20):
    assert image1.shape == image2.shape, "image1 shape should equal to image2 shape"
    
    # 计算图像A的频谱
    fA = np.fft.fft2(image1)
    fshiftA = np.fft.fftshift(fA)

    # 计算图像B的频谱
    fB = np.fft.fft2(image2)
    fshiftB = np.fft.fftshift(fB)

    # 获取图像大小和频谱中心点
    rows, cols = image1.shape
    crow, ccol = int(rows/2), int(cols/2)

    # 构造A图像的低通滤波器
    maskA = np.zeros((rows, cols), np.uint8)
    maskA[crow-d1:crow+d1, ccol-d1:ccol+d1] = 1
    md = fshiftA * maskA

    # 构造B图像的高通滤波器
    fshiftB[crow-d2:crow+d2, ccol-d2:ccol+d2] = 0

    # A图像低频与B图像高频融合
    fshiftB[crow-d3:crow+d3, ccol-d3:ccol+d3] = md[crow-d3:crow+d3, ccol-d3:ccol+d3]
    ishiftC = np.fft.ifftshift(fshiftB)
    iimgC = np.fft.ifft2(ishiftC)
    iimgC = np.abs(iimgC)
    iimgC = normalize_image(iimgC)

    return iimgC


def gaussian_pyramid(image, levels=6):
    pyramid_images = [image]
    for _ in range(levels-1):
        image = image[::2, ::2]
        pyramid_images.append(image)
    return pyramid_images