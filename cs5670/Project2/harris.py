import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from typing import Any, Tuple


def BGR2GRAY(img: np.ndarray, weights: list[float]) -> np.ndarray:
    """
    将BGR图像转换为灰度图像。

    Args:
        img: numpy.ndarray
            输入的BGR图像，形状为 (height, width, 3)。
        weights: list[float]
            包含三个浮点数的列表，表示灰度转换中红色、绿色和蓝色通道的权重。

    Returns:
        gray: numpy.ndarray
            灰度图像，数据类型为 uint8。
    """
    # 通过加权求和计算灰度值
    gray = (
        weights[0] * img[:, :, 2]
        + weights[1] * img[:, :, 1]
        + weights[2] * img[:, :, 0]
    )

    # 将灰度图像转换为8位无符号整数类型（uint8）
    return gray.astype(np.uint8)



def Sobel_filtering(gray: np.ndarray, args: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Sobel算子对灰度图像进行滤波。

    Args:
        gray: numpy.ndarray
            输入的灰度图像，形状为 (H, W)。
        args: Any
            包含Sobel算子参数的对象。

    Returns:
        Ix: numpy.ndarray
            x方向的Sobel滤波结果，数据类型为 float32。
        Iy: numpy.ndarray
            y方向的Sobel滤波结果，数据类型为 float32。
    """
    # 获取图像形状
    H, W = gray.shape

    # Sobel算子
    sobely = np.array(args.sobelx, dtype=np.float32)
    sobelx = np.array(args.sobely, dtype=np.float32)

    # 填充
    tmp = np.pad(gray, (1, 1), "edge")

    # 准备
    Ix = np.zeros_like(gray, dtype=np.float32)
    Iy = np.zeros_like(gray, dtype=np.float32)

    # 计算差分
    for y in range(H):
        for x in range(W):
            Ix[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobelx)
            Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)

    return Ix, Iy



def gaussian_filtering(I: np.ndarray, args: Any) -> np.ndarray:
    """
    使用高斯滤波器对图像进行滤波。

    Args:
        I: numpy.ndarray
            输入的图像，形状为 (H, W)。
        args: Any
            包含高斯滤波器参数的对象。

    Returns:
        I: numpy.ndarray
            滤波后的图像，形状与输入相同。
    """
    # 获取图像形状
    H, W = I.shape

    ## 高斯
    I_t = np.pad(I, (args.K_size // 2, args.K_size // 2), "edge")

    # 高斯核
    K = np.zeros((args.K_size, args.K_size), dtype=np.float32)
    for x in range(args.K_size):
        for y in range(args.K_size):
            _x = x - args.K_size // 2
            _y = y - args.K_size // 2
            K[y, x] = np.exp(-(_x**2 + _y**2) / (2 * (args.sigma**2)))
    K /= args.sigma * np.sqrt(2 * np.pi)
    K /= K.sum()

    # 滤波
    for y in range(H):
        for x in range(W):
            I[y, x] = np.sum(I_t[y : y + args.K_size, x : x + args.K_size] * K)

    return I



def corner_detect(
    img: np.ndarray, Ix2: np.ndarray, Iy2: np.ndarray, Ixy: np.ndarray, args: Any
) -> np.ndarray:
    """
    使用Harris角点检测算法检测图像中的角点。

    Args:
        img: numpy.ndarray
            输入的图像，形状为 (H, W)。
        Ix2: numpy.ndarray
            x方向梯度的平方，形状与输入图像相同。
        Iy2: numpy.ndarray
            y方向梯度的平方，形状与输入图像相同。
        Ixy: numpy.ndarray
            x和y方向梯度的乘积，形状与输入图像相同。
        args: Any
            包含Harris角点检测算法参数的对象。

    Returns:
        R_th: numpy.ndarray
            Harris响应的二值图像，形状与输入图像相同。
    """
    # 准备输出图像
    H, W = img.shape

    # 计算Harris角点响应矩阵
    R = np.zeros_like(img, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            # 计算Harris矩阵的各个分量
            M = np.array(
                [[Ix2[y, x], Ixy[y, x]], [Ixy[y, x], Iy2[y, x]]], dtype=np.float32
            )
            # 计算行列式和迹
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            if trace_M != 0:
                # 计算Harris响应
                R[y, x] = det_M / trace_M

    # 使用阈值比率阈值化Harris响应
    R_th = (R > R.max() * args.th_ratio) + 0
    return R_th



def threshold_and_nms(R: np.ndarray, args: Any) -> np.ndarray:
    """
    对角点响应图进行阈值化和非极大值抑制。

    Args:
        R: numpy.ndarray
            输入的角点响应图像，形状为 (H, W)。
        args: Any
            包含阈值化和非极大值抑制参数的对象。

    Returns:
        R_final: numpy.ndarray
            经过阈值化和非极大值抑制处理后的角点响应图，形状与输入图像相同。
    """
    # 阈值化处理
    R_th = (R > R.max() * args.th_nms) + 0
    # 使用膨胀操作对角点响应图进行处理
    R_dilate = dilate(R, args.ks)
    # 根据膨胀操作得到的结果进行非极大值抑制
    R_nms = R >= R_dilate
    # 最终的角点响应图
    R_final = R_th * R_nms

    return R_final



def dilate(R: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """
    使用矩形结构元素对图像进行膨胀操作。

    Args:
        R: numpy.ndarray
            输入的图像，形状为 (H, W)。
        kernel_size: Tuple[int, int]
            结构元素的大小，形状为 (kh, kw)。

    Returns:
        R_dilate: numpy.ndarray
            膨胀后的图像，形状与输入图像相同。
    """
    # 使用矩形结构元素进行膨胀操作
    R_dilate = np.zeros_like(R)

    H, W = R.shape
    kh, kw = kernel_size

    # 对图像进行遍历
    for y in range(H):
        for x in range(W):
            # 在当前位置应用结构元素
            for ky in range(kh):
                for kx in range(kw):
                    ny = y + ky - kh // 2
                    nx = x + kx - kw // 2
                    # 确保结构元素不超出图像边界
                    if 0 <= ny < H and 0 <= nx < W:
                        R_dilate[y, x] = max(R_dilate[y, x], R[ny, nx])

    return R_dilate



def DrawKeypoints(img: np.ndarray, R_final: np.ndarray, args: Any) -> None:
    """
    在图像上标记检测到的角点。

    Args:
        img: numpy.ndarray
            输入的图像，可以是灰度图或RGB图，形状为 (H, W) 或 (H, W, 3)。
        R_final: numpy.ndarray
            角点响应图，形状与输入图像相同。
        args: Any
            包含标记角点参数的对象，例如标记点的大小等。

    Returns:
        None
    """
    # 将图像转换为RGB格式（如果是灰度图的话）
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    # 找到角点的坐标
    corner_y, corner_x = np.where(R_final > 0)

    # 在图像上绘制角点
    plt.imshow(img_rgb)
    plt.scatter(corner_x, corner_y, c="r", s=args.s)  # 绘制红色的点作为角点
    plt.axis("off")
    plt.show()



def cornerHarris(img, args: Any) -> np.ndarray:
    """
    使用 Harris 角点检测算法检测图像中的角点，并标记在图像上。

    Args:
        args: Any
            包含 Harris 角点检测算法参数的对象。

    Returns:
        R_final: numpy.ndarray
    """

    gray = BGR2GRAY(img, args.gray)
    Ix, Iy = Sobel_filtering(gray, args)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    Ix2 = gaussian_filtering(Ix2, args)
    Iy2 = gaussian_filtering(Iy2, args)
    Ixy = gaussian_filtering(Ixy, args)

    R = corner_detect(gray, Ix2, Iy2, Ixy, args)
    R_final = threshold_and_nms(R, args)

    return R_final
    


if __name__ == "__main__":
    args = Namespace(
        img="image2.png",   # 输入图像的路径
        gray=[0.299, 0.114, 0.587], # 灰度转换的权重
        sobelx=[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],    # Sobel算子的x方向
        sobely=[[1, 0, -1], [2, 0, -2], [1, 0, -1]],    # Sobel算子的y方向
        K_size=3,   # 高斯滤波器的大小
        sigma=3,    # 高斯滤波器的标准差
        th_ratio=0.1,   # Harris角点检测算法的阈值比率
        th_nms=0.,    # 非极大值抑制的阈值
        ks=(3, 3),  # 膨胀操作的结构元素大小
        s=0.2,   # 标记角点的大小
    )

    img = cv2.imread(args.img)
    R_final = cornerHarris(img, args)

    DrawKeypoints(img.copy(), R_final, args)
