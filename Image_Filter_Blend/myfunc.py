import cv2
import numpy as np
import matplotlib.pyplot as plt

class project1:
    def __init__(self):
        print("计算机视觉 Project 1: Image Filtering and Hybrid Images")

    # 均值滤波
    def mean_filter(self, img, K_size=5):
        assert K_size % 2 != 0, "卷积核应为奇数"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, C = img.shape

        # 创建均值滤波器
        kernel = np.ones((K_size, K_size), dtype=np.float32) / (K_size * K_size)

        # 图片零填充
        pad = K_size // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

        # 使用广播进行卷积
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[y, x, c] = (padded_img[y:y+K_size, x:x+K_size, c] * kernel).sum()

        # 裁剪边缘
        out = out[pad:pad+H, pad:pad+W]

        # 克隆超出范围的值
        out = np.clip(out, 0, 255).astype(np.uint8)

        self.img_show(img, out)

    # 高斯滤波
    def gaussian_filter(self, img, K_size=3, sigma=1.3):
        assert K_size % 2 != 0, "卷积核应为奇数"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, C = img.shape
        ## 图片零填充
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
    
        ## 构造高斯卷积核
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    
        K /= (2 * np.pi * sigma * sigma)    
        K /= K.sum()    # 归一化
        tmp = out.copy()
    
        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    
        out = np.where(out < 0, 0, np.where(out > 255, 255, out))
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
        self.img_show(img, out)

    # 高通滤波器
    def myfilter():
        pass

    # 低通滤波器
    def myfilter():
        pass

    @staticmethod
    def img_show(img1, img2):
        plt.subplot(121), plt.imshow(img1), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(img2), plt.title('Mean Filter Image')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    # 使用示例
    project = project1()
    image = cv2.imread('dog.png')
    project.mean_filter(image)