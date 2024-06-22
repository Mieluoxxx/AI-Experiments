import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams['figure.figsize'] = [15, 15]

# 读取图像并转换为灰度图
def read_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 转换为RGB格式
    return img_gray, img, img_rgb

# SIFT算法提取关键点和描述符
def SIFT(img):
    siftDetector = cv2.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

# 绘制SIFT关键点
def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# 特征点匹配
def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    bf = cv2.BFMatcher()  # 使用默认参数的BFMatcher
    matches = bf.knnMatch(des1, des2, k=2)  # 使用knnMatch进行特征点匹配

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])  # 应用比值测试进行筛选

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

# 绘制匹配的特征点
def plot_matches(matches, total_img, filename=None):
    match_img = total_img.copy()
    offset = total_img.shape[1] / 2

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8'))

    ax.plot(matches[:, 0], matches[:, 1], 'xr')  # 标记左图像中的特征点
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')  # 标记右图像中的特征点

    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)  # 连接匹配的特征点对

    plt.savefig(f'{filename}.png')

# 计算单应性矩阵H
def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2]  # 标准化，使得H[2,2] = 1
    return H

# 从匹配集合中随机选择四个点
def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx]
    return np.array(point)

# 计算重投影误差
def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp / temp[2])[0:2]  # 归一化坐标
    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2  # 计算误差
    return errors

# RANSAC算法寻找最优单应性矩阵和内点集合
def ransac(matches, threshold, iters, min_error_prob=0.75):
    num_best_inliers = 0
    current_error_prob = 1.0

    for i in range(iters):
        points = random_point(matches)
        H = homography(points)

        if np.linalg.matrix_rank(H) < 3:  # 避免奇异矩阵
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

        # 更新当前错误概率
        current_error_prob = 1.0 - (num_best_inliers / len(matches))

        # 判断是否达到最小错误概率要求
        if current_error_prob < min_error_prob:
            break

    print("内点数/总匹配数: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

if __name__ == '__main__':
    # 读取图像
    left_gray, left_origin, left_rgb = read_image('image1.jpg')
    right_gray, right_origin, right_rgb = read_image('image2.jpg')

    # 确定目标尺寸，这里以左图像的尺寸为准
    target_height, target_width = left_gray.shape[:2]

    # 调整右图像的大小以匹配左图像的尺寸
    right_gray = cv2.resize(right_gray, (target_width, target_height))
    right_rgb = cv2.resize(right_rgb, (target_width, target_height))

    # 使用灰度图进行SIFT特征提取
    kp_left, des_left = SIFT(left_gray)
    kp_right, des_right = SIFT(right_gray)

    # 绘制SIFT关键点图像
    kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
    kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
    total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    plt.imshow(total_kp)
    plt.savefig('keypoints.png')
    plt.clf()

    # 进行特征点匹配
    matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

    # 将左右两幅图像拼接起来并绘制匹配的特征点
    total_img = np.concatenate((left_rgb, right_rgb), axis=1)
    plot_matches(matches, total_img, 'matches')

    # 使用RANSAC算法找出最优的单应性矩阵和内点集合，并绘制内点匹配
    inliers, H = ransac(matches, 0.5, 2000)
    plot_matches(inliers, total_img, 'inliers')
