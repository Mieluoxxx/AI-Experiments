import cv2
import numpy as np

def gaussian_smooth(img, sigma=1.3, kernel_size=5):
    """对图像应用高斯平滑"""
    # 计算填充的宽度
    padding_width = kernel_size // 2
    
    # 创建高斯核
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_kernel[i, j] = np.exp(-((i - padding_width) ** 2 + \
                            (j - padding_width) ** 2) / (2 * sigma ** 2))
    
    # 归一化高斯核
    gaussian_kernel /= np.sum(gaussian_kernel)
    
    # 获取图像的尺寸
    height, width = img.shape
    
    # 对图像进行零填充
    padded_img = np.zeros((height + 2 * padding_width, width + 2 * padding_width))
    padded_img[padding_width:height + padding_width, padding_width:width + padding_width] = img
    
    # 应用高斯滤波
    smoothed_img = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            smoothed_img[i, j] = np.sum(padded_img[i:i + kernel_size, j:j + kernel_size] * gaussian_kernel)
    
    return np.uint8(smoothed_img)

def get_gradient_and_direction(img):
    """计算图像的梯度和方向"""
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # 获取图像的尺寸
    height, width = img.shape
    
    # 初始化梯度和方向矩阵
    gradients = np.zeros((height - 2, width - 2))
    directions = np.zeros((height - 2, width - 2))
    
    # 计算梯度和方向
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(img[i:i+3, j:j+3] * sobel_x)
            gy = np.sum(img[i:i+3, j:j+3] * sobel_y)
            gradients[i - 1, j - 1] = np.sqrt(gx ** 2 + gy ** 2)
            if gx == 0:
                directions[i - 1, j - 1] = np.pi / 2
            else:
                directions[i - 1, j - 1] = np.arctan(gy / gx)
    
    return np.uint8(gradients), directions

def non_maximum_suppression(gradients, directions):
    """非极大值抑制"""
    # 获取梯度图像的尺寸
    height, width = gradients.shape
    
    # 初始化NMS结果矩阵
    nms_result = np.copy(gradients[1:-1, 1:-1])
    
    # 非极大值抑制处理
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            theta = directions[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                delta_i, delta_j = 0, 1
                weight = 1 / weight
            elif 0 <= theta <= np.pi / 4:
                delta_i, delta_j = 1, 1
            elif -np.pi / 4 <= theta < 0:
                delta_i, delta_j = 1, 0
            else:
                delta_i, delta_j = 1, -1
            
            gradient_top = gradients[i + delta_i, j + delta_j]
            gradient_diagonal = gradients[i + delta_i * -1, j + delta_j * -1]
            
            if gradient_top > gradients[i, j] or gradient_diagonal > gradients[i, j]:
                nms_result[i - 1, j - 1] = 0
    
    return nms_result

def double_thresholding(nms_result, threshold_low, threshold_high):
    """执行双阈值处理，并对高于高阈值的区域执行深度优先搜索标记。"""
    height, width = nms_result.shape
    visited = np.zeros_like(nms_result, dtype=bool)
    output_img = np.zeros_like(nms_result, dtype=np.uint8)

    def dfs(x, y):
        "“执行深度优先搜索，标记连通区域。”""
        if (0 <= x < width) and (0 <= y < height) and not visited[x, y] and nms_result[x, y] > threshold_low:
            visited[x, y] = True
            output_img[x, y] = 255
            dfs(x - 1, y)
            dfs(x + 1, y)
            dfs(x, y - 1)
            dfs(x, y + 1)

    # 标记高于高阈值的区域
    for i in range(width):
        for j in range(height):
            if nms_result[i, j] >= threshold_high and not visited[i, j]:
                dfs(i, j)

    # 将低于低阈值的区域设置为0
    output_img[nms_result <= threshold_low] = 0

    return output_img

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 应用高斯平滑
    smoothed_image = gaussian_smooth(image)
    
    # 计算梯度和方向
    gradients, directions = get_gradient_and_direction(smoothed_image)
    
    # 应用非极大值抑制
    nms_result = non_maximum_suppression(gradients, directions)
    
    # 执行双阈值处理
    final_output = double_thresholding(nms_result, 40, 100)
    
    # 显示结果
    plt.imshow(final_output, cmap='gray')
    plt.axis('off')
    plt.show()